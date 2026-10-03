import numpy as np

######################### TUTAJ JEST MIEJSCE NA TWOJE FINALNE ROZWIĄZANIE ##########################

class YourSolution:
    """Cosine-neighbor multilabel classifier that respects taxonomy depth."""

    def __init__(self):
        self.n_neighbors = 80
        self.similarity_power = 4
        # A depth-specific STOP bias, tuned on the released validation set.
        self.stop_bias = np.array([0, 1, 1, 3, 2, 1, 1], dtype=np.float32)

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float32)
        self.X_train = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
        self.paths = [list(path) for path in y]
        self.max_depth = max((len(path) for path in self.paths), default=0)
        self.classes_by_depth = []
        self.train_codes_by_depth = []

        for depth in range(self.max_depth):
            classes = sorted({path[depth] for path in self.paths if len(path) > depth})
            class_to_code = {label: index for index, label in enumerate(classes)}
            stop_code = len(classes)
            codes = np.full(len(self.paths), stop_code, dtype=np.int32)
            for row_index, path in enumerate(self.paths):
                if len(path) > depth:
                    codes[row_index] = class_to_code[path[depth]]
            self.classes_by_depth.append(classes)
            self.train_codes_by_depth.append(codes)
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=np.float32)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
        similarities = X @ self.X_train.T
        k = min(self.n_neighbors, len(self.X_train))
        neighbor_indices = np.argpartition(similarities, -k, axis=1)[:, -k:]
        neighbor_similarities = np.take_along_axis(similarities, neighbor_indices, axis=1)
        order = np.argsort(neighbor_similarities, axis=1)[:, ::-1]
        neighbor_indices = np.take_along_axis(neighbor_indices, order, axis=1)
        neighbor_similarities = np.take_along_axis(neighbor_similarities, order, axis=1)
        weights = np.maximum(neighbor_similarities, 0.0) ** self.similarity_power

        predictions = [[] for _ in range(len(X))]
        row_indices = np.arange(len(X))
        for depth, classes in enumerate(self.classes_by_depth):
            votes = np.zeros((len(X), len(classes) + 1), dtype=np.float32)
            train_codes = self.train_codes_by_depth[depth]
            for rank in range(k):
                codes = train_codes[neighbor_indices[:, rank]]
                votes[row_indices, codes] += weights[:, rank]

            best_codes = votes[:, :-1].argmax(axis=1)
            best_votes = votes[row_indices, best_codes]
            stop_votes = votes[:, -1]
            bias = self.stop_bias[depth] if depth < len(self.stop_bias) else 1.0
            use_category = best_votes > bias * stop_votes
            for row_index in np.flatnonzero(use_category):
                label = classes[best_codes[row_index]]
                if label not in predictions[row_index]:
                    predictions[row_index].append(label)
        return predictions
