import numpy as np
import torch


class YourSolution:
    """Fit the supplied 10-circle renderer with CPU-only continuation + Adam."""

    SHARPNESS_STAGES = (0.12, 0.2, 0.35, 0.6, 1.0, 2.0, 4.0, 8.0, 20.0)
    MAX_RESTARTS = 6
    STOP_MSE = 0.0075
    ADAM_STEPS = 100
    ADAM_LR = 0.15

    def __init__(self):
        self._grid_shape = None
        self._x_grid = None
        self._y_grid = None

    def _ensure_grid(self, height, width):
        if self._grid_shape != (height, width):
            self._y_grid, self._x_grid = np.mgrid[0:height, 0:width]
            self._grid_shape = (height, width)

    def _render_and_grad(self, params, target, sharpness):
        """Match the official alpha compositing and return MSE plus its gradient."""
        p = params.reshape(10, 7)
        x_grid, y_grid = self._x_grid, self._y_grid
        height, width = target.shape[1:]
        q_list, mask_list, before_list = [], [], []
        image = np.ones((3, height, width), dtype=np.float64)

        for row in p:
            x, y, radius, red, green, blue, alpha = row
            z = (radius * radius - (x_grid - x) ** 2 - (y_grid - y) ** 2) * sharpness
            mask = 1.0 / (1.0 + np.exp(-np.clip(z, -60.0, 60.0)))
            pixel_alpha = mask * alpha
            before_list.append(image)
            q_list.append(pixel_alpha)
            mask_list.append(mask)
            color = row[3:6, None, None]
            image = color * pixel_alpha[None] + image * (1.0 - pixel_alpha[None])

        residual = image - target
        loss = np.mean(residual * residual)
        image_grad = (2.0 / residual.size) * residual
        param_grad = np.zeros((10, 7), dtype=np.float64)

        # Backpropagate through the compositing sequence from top circle to bottom.
        for i in range(9, -1, -1):
            row = p[i]
            x, y, radius, red, green, blue, alpha = row
            pixel_alpha = q_list[i]
            mask = mask_list[i]
            before = before_list[i]
            color = row[3:6, None, None]
            dloss_dalpha = np.sum(image_grad * (color - before), axis=0)

            param_grad[i, 3:6] = np.sum(image_grad * pixel_alpha[None], axis=(1, 2))
            dmask_dz = mask * (1.0 - mask)
            dloss_dz = dloss_dalpha * alpha * dmask_dz
            param_grad[i, 0] = np.sum(dloss_dz * (2.0 * sharpness * (x_grid - x)))
            param_grad[i, 1] = np.sum(dloss_dz * (2.0 * sharpness * (y_grid - y)))
            param_grad[i, 2] = np.sum(dloss_dz * (2.0 * sharpness * radius))
            param_grad[i, 6] = np.sum(dloss_dalpha * mask)
            image_grad *= 1.0 - pixel_alpha[None]

        return loss, param_grad.ravel()

    def _fit_restart(self, target, seed):
        rng = np.random.default_rng(seed)
        params = np.empty((10, 7), dtype=np.float64)
        params[:, 0:2] = rng.uniform(6.0, 58.0, size=(10, 2))
        params[:, 2] = rng.uniform(5.0, 17.0, size=10)
        params[:, 3:6] = rng.uniform(0.02, 0.98, size=(10, 3))
        params[:, 6] = rng.uniform(0.5, 1.0, size=10)

        lower = np.array([0, 0, 3, 0, 0, 0, 0.5] * 10, dtype=np.float64)
        upper = np.array([63, 63, 22, 1, 1, 1, 1] * 10, dtype=np.float64)

        for sharpness in self.SHARPNESS_STAGES:
            first_moment = np.zeros(70, dtype=np.float64)
            second_moment = np.zeros(70, dtype=np.float64)
            best_params = params.ravel().copy()
            best_loss = float("inf")

            for step in range(1, self.ADAM_STEPS + 1):
                loss, grad = self._render_and_grad(params.ravel(), target, sharpness)
                if loss < best_loss:
                    best_loss = loss
                    best_params = params.ravel().copy()

                first_moment = 0.9 * first_moment + 0.1 * grad
                second_moment = 0.999 * second_moment + 0.001 * grad * grad
                corrected_first = first_moment / (1.0 - 0.9**step)
                corrected_second = second_moment / (1.0 - 0.999**step)
                params = (params.ravel() - self.ADAM_LR * corrected_first /
                          (np.sqrt(corrected_second) + 1e-8)).reshape(10, 7)
                params = np.clip(params.ravel(), lower, upper).reshape(10, 7)

            params = best_params.reshape(10, 7)

        final_loss, _ = self._render_and_grad(params.ravel(), target, 20.0)
        return params, final_loss

    def solve(self, target_image: torch.Tensor) -> torch.Tensor:
        """Return ten [x, y, radius, red, green, blue, alpha] circle records."""
        height, width = target_image.shape[-2:]
        self._ensure_grid(height, width)
        target = target_image.detach().cpu().numpy().astype(np.float64, copy=False)

        best_params = None
        best_loss = float("inf")
        for seed in range(self.MAX_RESTARTS):
            params, loss = self._fit_restart(target, seed)
            if loss < best_loss:
                best_loss = loss
                best_params = params
            if best_loss <= self.STOP_MSE:
                break

        result = torch.as_tensor(best_params.astype(np.float32), dtype=target_image.dtype)
        return result.to(target_image.device)
