"""StyleGAN-prior inversion for the Poland 2026 face-colorization task.

Copy this class into the solution cell of the official notebook.  It uses only
the supplied generator and PyTorch; no image weights or internet access are
needed beyond the checkpoint loaded by the starter code.
"""

import torch
import torch.nn.functional as F


class YourModel(torch.nn.Module):
    """Invert each gray face into the supplied StyleGAN2 generator's W space.

    The generator in the task exposes `style_to_image(W)`, which applies one W
    vector at every synthesis layer and injects fresh random noise.  Inversion
    is more stable with layer-wise W+ codes and fixed noise, so `_synthesize`
    follows the same module sequence while using its registered noise buffers.
    """

    def __init__(self, generator):
        super().__init__()
        self.generator = generator.eval()
        for parameter in self.generator.parameters():
            parameter.requires_grad_(False)
        self.register_buffer("_w_mean", torch.zeros(1, 512), persistent=False)
        self.register_buffer("_w_std", torch.ones(1, 512), persistent=False)
        self.register_buffer(
            "_luma", torch.tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1),
            persistent=False,
        )
        self._fitted = False

    def fit(self):
        """Estimate the center and scale of the generator's mapped W prior."""
        device = next(self.generator.parameters()).device
        with torch.no_grad():
            z = torch.randn(1024, self.generator.style_dim, device=device)
            mapped = self.generator.get_latent(z)
            self._w_mean = mapped.mean(dim=0, keepdim=True).detach()
            self._w_std = mapped.std(dim=0, unbiased=False, keepdim=True).clamp_min(0.05).detach()
        self._fitted = True
        return self

    def _synthesize(self, latent):
        """Synthesize from W+ codes [B, n_latent, 512] with fixed noise."""
        g = self.generator
        if latent.ndim == 2:
            latent = latent[:, None, :].expand(-1, g.n_latent, -1)

        # `stylegan.py` registers the canonical noise maps but its public helper
        # omits them, causing new random noise at every optimization step.
        noise = [getattr(g.noises, f"noise_{i}") for i in range(g.num_layers)]
        out = g.input(latent)
        out = g.conv1(out, latent[:, 0], noise=noise[0])
        skip = g.to_rgb1(out, latent[:, 1])

        style_idx = 1
        noise_idx = 1
        for conv_a, conv_b, to_rgb in zip(g.convs[::2], g.convs[1::2], g.to_rgbs):
            out = conv_a(out, latent[:, style_idx], noise=noise[noise_idx])
            out = conv_b(out, latent[:, style_idx + 1], noise=noise[noise_idx + 1])
            skip = to_rgb(out, latent[:, style_idx + 2], skip)
            style_idx += 2
            noise_idx += 2
        return skip

    def _gray(self, rgb):
        return (rgb * self._luma.to(dtype=rgb.dtype, device=rgb.device)).sum(dim=1, keepdim=True)

    def _reconstruction_loss(self, rgb, gray):
        pred_gray = self._gray(rgb)
        # Coarse structure matters most for matching the face; full-resolution
        # and edge terms retain hair, eyes, and other fine features.
        loss = F.l1_loss(pred_gray, gray)
        for factor, weight in ((2, 0.25), (4, 0.25)):
            loss = loss + weight * F.l1_loss(
                F.avg_pool2d(pred_gray, factor), F.avg_pool2d(gray, factor)
            )
        dx_pred = pred_gray[:, :, :, 1:] - pred_gray[:, :, :, :-1]
        dy_pred = pred_gray[:, :, 1:, :] - pred_gray[:, :, :-1, :]
        dx_gray = gray[:, :, :, 1:] - gray[:, :, :, :-1]
        dy_gray = gray[:, :, 1:, :] - gray[:, :, :-1, :]
        loss = loss + 0.08 * (F.l1_loss(dx_pred, dx_gray) + F.l1_loss(dy_pred, dy_gray))
        return loss, pred_gray

    def _invert(self, gray):
        batch = gray.shape[0]
        w0 = self._w_mean.expand(batch, -1).clone()
        std = self._w_std

        # First find a semantic W code shared across layers.
        w = torch.nn.Parameter(w0.clone())
        opt = torch.optim.Adam([w], lr=0.055)
        for step in range(8):
            opt.zero_grad(set_to_none=True)
            rgb = self._synthesize(w)
            loss, _ = self._reconstruction_loss(rgb, gray)
            loss = loss + 0.006 * (((w - w0) / std) ** 2).mean()
            loss.backward()
            opt.step()

        # W+ improves local alignment while a prior penalty prevents the code
        # from drifting too far from the plausible face manifold.
        anchor = w.detach()[:, None, :].expand(-1, self.generator.n_latent, -1).clone()
        w_plus = torch.nn.Parameter(anchor.clone())
        opt = torch.optim.Adam([w_plus], lr=0.025)
        for step in range(10):
            opt.zero_grad(set_to_none=True)
            rgb = self._synthesize(w_plus)
            loss, _ = self._reconstruction_loss(rgb, gray)
            loss = loss + 0.012 * (((w_plus - anchor) / std[:, None, :]) ** 2).mean()
            loss.backward()
            opt.step()

        with torch.no_grad():
            rgb = self._synthesize(w_plus)
            fit_error = (self._gray(rgb) - gray).square().mean(dim=(1, 2, 3), keepdim=True).sqrt()
        return rgb.detach(), fit_error.detach()

    def predict(self, x_gray):
        if not self._fitted:
            self.fit()
        outputs = []
        # The official validator wraps prediction in no_grad; enable gradients
        # locally because only latent codes, never generator weights, are fitted.
        with torch.enable_grad():
            for gray in x_gray.split(8, dim=0):
                rgb_gan, fit_error = self._invert(gray)
                gray_rgb = gray.expand(-1, 3, -1, -1)
                chroma = rgb_gan - self._gray(rgb_gan)
                # Preserve the observed luminance exactly. Shrink chroma when
                # the generator could not match the input's gray structure.
                strength = (0.72 / (1.0 + 5.0 * fit_error)).clamp(0.30, 0.72)
                outputs.append((gray_rgb + strength * chroma).clamp(-1.0, 1.0).detach())
        return torch.cat(outputs, dim=0)
