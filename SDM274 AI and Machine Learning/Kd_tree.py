import numpy as np
from collections import Counter
import heapq

class KdNode:
    def __init__(self, point=None, label=None, axis=None, left=None, right=None):
        self.point = point
        self.label = label
        self.axis = axis
        self.left = left
        self.right = right

class KdTree:
    def __init__(self, data, labels):
        if data.shape[0] != len(labels):
            raise ValueError("Data and labels must have the same number of samples.")
        self.data = data
        self.labels = labels
        self.dim = data.shape[1]
        self.root = self._build_tree(list(range(len(data))))

    def _build_tree(self, indices, depth=0):
        if not indices:
            return None

        axis = depth % self.dim

        indices.sort(key=lambda i: self.data[i, axis])
        median_idx = len(indices) // 2
        median_point_idx = indices[median_idx]

        node = KdNode(
            point=self.data[median_point_idx],
            label=self.labels[median_point_idx],
            axis=axis
        )
        node.left = self._build_tree(indices[:median_idx], depth + 1)
        node.right = self._build_tree(indices[median_idx + 1:], depth + 1)

        return node

    def _distance(self, p1, p2):
        return np.sum((p1 - p2)**2)

    def query(self, query_point, k=1):
        if self.root is None:
            return []

        self.k_nearest = []

        def search(node):
            if node is None:
                return

            dist = self._distance(query_point, node.point)
            if len(self.k_nearest) < k:
                heapq.heappush(self.k_nearest, (-dist, node.label))
            elif dist < -self.k_nearest[0][0]:
                heapq.heapreplace(self.k_nearest, (-dist, node.label))
            axis = node.axis
            if query_point[axis] < node.point[axis]:
                closer_subtree = node.left
                further_subtree = node.right
            else:
                closer_subtree = node.right
                further_subtree = node.left
            search(closer_subtree)
            hyperplane_dist_sq = (query_point[axis] - node.point[axis])**2

            if len(self.k_nearest) < k or hyperplane_dist_sq < -self.k_nearest[0][0]:
                search(further_subtree)

        search(self.root)
        return [label for neg_dist, label in sorted(self.k_nearest, key=lambda x: -x[0])]
