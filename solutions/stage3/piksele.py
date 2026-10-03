import numpy as np
import torch

# Fit once from the labeled sparse training set, then reuse the models for all
# validation/test images. This combines spatial statistics with the supplied CNN.
_pixel_models_cache = None

def _extract_pixel_cnn_features(images, model):
    """Return the penultimate CNN activations for a batch of sparse images."""
    device = next(model.parameters()).device
    chunks = []
    with torch.no_grad():
        for start in range(0, len(images), 128):
            batch = torch.as_tensor(images[start:start + 128], dtype=torch.float32, device=device)
            batch = (batch - 0.1307) / 0.3081
            chunks.append(model.net[:9](batch).cpu().numpy())
    return np.concatenate(chunks, axis=0)

def _get_pixel_models():
    global _pixel_models_cache
    if _pixel_models_cache is not None:
        return _pixel_models_cache

    from scipy.ndimage import gaussian_filter, maximum_filter
    from sklearn.svm import SVC

    if hasattr(train_imgs, "detach"):
        train_x = train_imgs.detach().cpu().numpy().astype(np.float32)[:, 0]
        train_y = train_labels.detach().cpu().numpy().astype(int)
    else:
        train_x = np.asarray(train_imgs, dtype=np.float32)[:, 0]
        train_y = np.asarray(train_labels, dtype=int)

    # Class-conditional spatial density of visible pixels. Smooth the maps to
    # tolerate the one-pixel shifts common in handwritten digits.
    log_density = []
    for cls in range(10):
        class_images = train_x[train_y == cls]
        counts = class_images.sum(axis=0)
        counts = gaussian_filter(counts, sigma=0.9) + 1e-3
        density = counts / counts.sum()
        log_density.append(np.log(density).reshape(-1))
    log_density = np.asarray(log_density, dtype=np.float32)

    # An RBF SVC on blurred sparse masks captures local point geometry.
    blurred_train = np.stack([gaussian_filter(im, sigma=1.5) for im in train_x])
    pixel_svm = SVC(C=10, kernel="rbf", gamma="scale", decision_function_shape="ovr")
    pixel_svm.fit(blurred_train.reshape(len(blurred_train), -1), train_y)

    # The CNN was trained on complete digits. Max-filter views make the sparse
    # observations look more like connected strokes; train a small SVC on its
    # penultimate features from raw, 3x3-dilated, and 5x5-dilated views.
    dilated3 = np.stack([maximum_filter(im, size=3) for im in train_x])[:, None]
    dilated5 = np.stack([maximum_filter(im, size=5) for im in train_x])[:, None]
    cnn_views = [train_x[:, None], dilated3, dilated5]
    hidden_views = [_extract_pixel_cnn_features(view, model) for view in cnn_views]
    hidden_train = np.concatenate(hidden_views, axis=1)
    hidden_svm = SVC(C=1, kernel="rbf", gamma="scale", decision_function_shape="ovr")
    hidden_svm.fit(hidden_train, train_y)

    _pixel_models_cache = (log_density, pixel_svm, hidden_svm)
    return _pixel_models_cache

def _standardize_pixel_scores(scores):
    return (scores - scores.mean(axis=1, keepdims=True)) / (
        scores.std(axis=1, keepdims=True) + 1e-9
    )

def solution(img):
    """Classify a 28x28 MNIST image showing only ten visible pixels."""
    from scipy.ndimage import gaussian_filter, maximum_filter

    if hasattr(img, "detach"):
        image = img.detach().cpu().numpy().astype(np.float32)
    else:
        image = np.asarray(img, dtype=np.float32)
    image = np.squeeze(image)

    if "train_imgs" not in globals() or "train_labels" not in globals():
        batch = torch.as_tensor(image, dtype=torch.float32).reshape(1, 1, 28, 28)
        predicted, _ = predict(batch, model)
        return int(predicted[0].item())

    log_density, pixel_svm, hidden_svm = _get_pixel_models()
    flat = np.maximum(image, 0).reshape(1, -1)
    density_scores = np.sqrt(flat) @ log_density.T

    blurred = gaussian_filter(image, sigma=1.5).reshape(1, -1)
    pixel_svm_scores = pixel_svm.decision_function(blurred)

    dilated3 = maximum_filter(image, size=3)
    dilated5 = maximum_filter(image, size=5)
    views = np.stack([image, dilated3, dilated5])[:, None]
    device = next(model.parameters()).device
    with torch.no_grad():
        batch = torch.as_tensor(views, dtype=torch.float32, device=device)
        batch = (batch - 0.1307) / 0.3081
        hidden = model.net[:9](batch)
        cnn_scores = model.net[9](hidden[2:3]).cpu().numpy()
        hidden_features = hidden.cpu().numpy().reshape(1, -1)
    hidden_svm_scores = hidden_svm.decision_function(hidden_features)

    scores = (
        _standardize_pixel_scores(density_scores)
        + 0.4 * _standardize_pixel_scores(pixel_svm_scores)
        + 1.25 * _standardize_pixel_scores(hidden_svm_scores)
        + 0.3 * _standardize_pixel_scores(cnn_scores)
    )
    return int(np.argmax(scores[0]))
