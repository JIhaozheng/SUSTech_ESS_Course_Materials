import numpy as np

class TreeNode:
    def __init__(self, feature_idx=None, threshold=None, left=None, right=None, *, value=None):
        self.feature_idx = feature_idx
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

class DecisionTree:
    def __init__(self, max_depth=None, min_samples_split=2):
        self.root = None
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split

    def fit(self, X, y):
        self.n_classes = len(np.unique(y))
        self.root = self._build_tree(X, y)
    
    def _entropy(self, y):
        hist = np.bincount(y)
        ps = hist / len(y)
        ps = ps[ps > 0]
        return -np.sum(ps * np.log2(ps))

    def _information_gain(self, y, y_left, y_right):
        H = self._entropy(y)
        H_left = self._entropy(y_left)
        H_right = self._entropy(y_right)
        p_left = len(y_left) / len(y)
        p_right = len(y_right) / len(y)
        return H - (p_left * H_left + p_right * H_right)

    def _best_split(self, X, y):
        n_samples, n_features = X.shape
        best_gain = -1
        split_idx, split_thr = None, None
        
        for feature_idx in range(n_features):
            thresholds = np.unique(X[:, feature_idx])
            for thr in thresholds:
                left_idx = X[:, feature_idx] <= thr
                right_idx = X[:, feature_idx] > thr
                if np.sum(left_idx) == 0 or np.sum(right_idx) == 0:
                    continue
                y_left, y_right = y[left_idx], y[right_idx]
                gain = self._information_gain(y, y_left, y_right)
                if gain > best_gain:
                    best_gain = gain
                    split_idx = feature_idx
                    split_thr = thr
        return split_idx, split_thr, best_gain

    def _build_tree(self, X, y, depth=0):
        n_samples = X.shape[0]
        n_labels = len(np.unique(y))
        
        if n_labels == 1 or n_samples < self.min_samples_split or (self.max_depth is not None and depth >= self.max_depth):
            leaf_value = np.bincount(y).argmax()
            return TreeNode(value=leaf_value)
        
        feature_idx, threshold, gain = self._best_split(X, y)
        if gain == -1:
            leaf_value = np.bincount(y).argmax()
            return TreeNode(value=leaf_value)
        
        left_idx = X[:, feature_idx] <= threshold
        right_idx = X[:, feature_idx] > threshold
        left = self._build_tree(X[left_idx], y[left_idx], depth + 1)
        right = self._build_tree(X[right_idx], y[right_idx], depth + 1)
        return TreeNode(feature_idx, threshold, left, right)

    def _predict_sample(self, x, node):
        if node.value is not None:
            return node.value
        if x[node.feature_idx] <= node.threshold:
            return self._predict_sample(x, node.left)
        else:
            return self._predict_sample(x, node.right)

    def predict(self, X):
        return np.array([self._predict_sample(x, self.root) for x in X])

    def accuracy(self, X, real):
        pred = self.predict(X)
        accuracy = np.mean(pred == real) * 100
        true_idx = np.where(pred == real)[0]
        wrong_idx = np.where(pred != real)[0]
        return accuracy, pred, real, true_idx, wrong_idx
