"""Train-only 150-prototype construction for the 2025 OAI task."""

from __future__ import annotations

import torch


class YourSolution:
    """Allocate prototypes to broad/confusable classes, then run classwise k-means."""

    @staticmethod
    def _two_means(x: torch.Tensor, mean: torch.Tensor) -> torch.Tensor:
        """Small deterministic farthest-first k-means with two centers."""
        squared_from_mean = (x - mean).square().sum(dim=1)
        center0 = mean
        center1 = x[squared_from_mean.argmax()]
        centers = torch.stack((center0, center1))

        for _ in range(12):
            distances = torch.cdist(x, centers).square()
            assignment = distances.argmin(dim=1)
            updated = []
            for cluster_id in range(2):
                members = x[assignment == cluster_id]
                if members.shape[0] == 0:
                    other = centers[1 - cluster_id]
                    updated.append(x[((x - other) ** 2).sum(dim=1).argmax()])
                else:
                    updated.append(members.mean(dim=0))
            new_centers = torch.stack(updated)
            if torch.max(torch.abs(new_centers - centers)).item() < 1e-5:
                centers = new_centers
                break
            centers = new_centers
        return centers

    def get_prototypes(self, train_embeddings, train_labels):
        """Return 150 float32 centers and their long class labels.

        Only the released training embeddings/labels are read.  Every class
        receives one center; one extra center is assigned to the 50 classes
        with the largest within-class spread relative to centroid separation.
        """
        device = train_embeddings.device
        x = train_embeddings.detach().to(device=device, dtype=torch.float32)
        y = train_labels.detach().to(device=device, dtype=torch.long)
        classes = torch.unique(y, sorted=True)
        if classes.numel() != 100:
            raise ValueError(f"expected 100 classes, got {classes.numel()}")

        class_rows = []
        means = []
        radii = []
        for class_id in classes:
            class_x = x[y == class_id]
            if class_x.shape[0] == 0:
                raise ValueError(f"training embeddings are missing class {int(class_id)}")
            mean = class_x.mean(dim=0)
            means.append(mean)
            radii.append((class_x - mean).square().sum(dim=1).mean().sqrt())
            class_rows.append(class_x)

        means = torch.stack(means)
        center_distances = torch.cdist(means, means)
        center_distances.fill_diagonal_(float("inf"))
        separation = center_distances.min(dim=1).values.clamp_min(1e-6)
        spread_score = torch.stack(radii) / separation
        extra_class_indices = torch.topk(spread_score, k=50, largest=True).indices
        use_two = torch.zeros(100, dtype=torch.bool, device=device)
        use_two[extra_class_indices] = True

        prototypes = []
        prototype_labels = []
        for class_index, class_id in enumerate(classes):
            class_x = class_rows[class_index]
            if bool(use_two[class_index]):
                class_centers = self._two_means(class_x, means[class_index])
            else:
                class_centers = means[class_index].unsqueeze(0)
            prototypes.append(class_centers)
            prototype_labels.extend([int(class_id)] * class_centers.shape[0])

        prototypes = torch.cat(prototypes, dim=0).to(device=device, dtype=torch.float32)
        prototype_labels = torch.tensor(
            prototype_labels, device=device, dtype=torch.long
        )
        if prototypes.shape[0] != 150 or prototype_labels.shape[0] != 150:
            raise RuntimeError("prototype allocation did not produce exactly 150 rows")
        return prototypes, prototype_labels
