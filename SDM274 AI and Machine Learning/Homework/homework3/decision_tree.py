import numpy as np

class DecisionTree:
    class Node:
        def __init__(self) -> None:
            self.value = None
            self.feature_index = None
            self.children = {}

        def __str__(self) -> str:
            if self.children:
                s = f'Internal node <{self.feature_index}>:\n'
                for fv, node in self.children.items():
                    ss = f'[{fv}]-> {node}'
                    s += '\t' + ss.replace('\n', '\n\t') + '\n'
            else:
                s = f'Leaf node ({self.value})'
            return s

    def __init__(self, gain_threshold=1e-2) -> None:
        self.gain_threshold = gain_threshold

    def _entropy(self, y):
        count_y = np.bincount(y)
        prob_y = count_y[np.nonzero(count_y)] / y.size
        entropy_y = -np.sum(prob_y * np.log2(prob_y))
        return entropy_y

    def _conditional_entropy(self, feature, y):
        feature_values = np.unique(feature)
        h = 0.
        for v in feature_values:
            y_sub = y[feature == v]
            prob_y_sub = y_sub.size / y.size
            h += prob_y_sub * self._entropy(y_sub)
        return h

    def _information_gain(self, feature, y):
        ig_feature = self._entropy(y) - self._conditional_entropy(feature, y)
        return ig_feature

    def _select_feature(self, X, y, features_list):
        if features_list:
            gains = np.apply_along_axis(self._information_gain, 0, X[:, features_list], y)
            index = np.argmax(gains)
            if gains[index] > self.gain_threshold:
                return index
        return None

    def _build_tree(self, X, y, features_list):
        node = self.Node()
        labels_count = np.bincount(y)
        node.value = np.argmax(labels_count)

        if np.count_nonzero(labels_count) != 1:
            index = self._select_feature(X, y, features_list)
            if index is not None:
                node.feature_index = features_list.pop(index)
                feature_values = np.unique(X[:, node.feature_index])
                for v in feature_values:
                    idx = X[:, node.feature_index] == v
                    X_sub, y_sub = X[idx], y[idx]
                    node.children[v] = self._build_tree(X_sub, y_sub, features_list.copy())
        return node

    def train(self, X_train, y_train):
        _, n = X_train.shape
        self.tree_ = self._build_tree(X_train, y_train, list(range(n)))

    def _predict_one(self, x):
        node = self.tree_
        while node.children:
            child = node.children.get(x[node.feature_index])
            if not child:
                break
            node = child
        return node.value

    def predict(self, X):
        return np.apply_along_axis(self._predict_one, axis=1, arr=X)

    def __str__(self):
        if hasattr(self, 'tree_'):
            return str(self.tree_)
        return ''

    def calculate_tree_depth(self, node):
        if not node.children:
            return 1
        else:
            max_child_depth = max(self.calculate_tree_depth(child) for child in node.children.values())
            return max_child_depth + 1

    def get_tree_depth(self):
        if hasattr(self, 'tree_'):
            return self.calculate_tree_depth(self.tree_)
        else:
            return 0


if __name__ == '__main__':
    data = np.loadtxt('./lenses/lenses.data', dtype=int)
    X = data[:, 1:-1]
    y = data[:, -1]

    dt01 = DecisionTree()
    dt01.train(X, y)

    depth = dt01.get_tree_depth()