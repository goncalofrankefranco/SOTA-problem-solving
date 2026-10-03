import torch
import torch.nn.functional as F

# Grad-CAM over the final ResNet block.  The returned map is normalized to [0, 1];
# the evaluator thresholds it at 0.5.
def your_solution(img, model):
    import torch.nn.functional as F

    _, H, W = img.shape
    device = next(model.parameters()).device
    model.eval()

    # The evaluator wraps this function in no_grad(), so explicitly re-enable
    # autograd. Requiring an input gradient also keeps the graph when model
    # parameters have been frozen.
    with torch.enable_grad():
        x = img.detach().unsqueeze(0).to(device).requires_grad_(True)
        feature_map = model.encoder(x)
        pooled = model.pool(feature_map).flatten(1)
        logit = model.head(pooled).reshape(-1)[0]
        gradient = torch.autograd.grad(logit, feature_map)[0]

        channel_weights = gradient.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((channel_weights * feature_map).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=(H, W), mode="bilinear", align_corners=False)[0, 0]
        cam = torch.nan_to_num(cam)
        lo, hi = cam.min(), cam.max()
        cam = (cam - lo) / (hi - lo + 1e-8)

    return cam.detach()
