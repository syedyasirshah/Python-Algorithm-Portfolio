from collections import Counter
from dataclasses import dataclass
import math
from typing import Optional

import numpy as np
from numpy.typing import NDArray


@dataclass
class TreeNode:
    """ Class for Node """

    feature_index: Optional[int] = None
    threshold: Optional[float] = None
    left: Optional["TreeNode"] = None
    right: Optional["TreeNode"] = None
    value: Optional[object] = None

    @property
    def is_leaf(self) -> bool:
        """ To check for leap node """
        return self.value is not None


class DecisionTreeClassifier:
    """ Class for overall decision tree """

    def __init__(
        self,
        max_depth: int = 5,
        min_samples_split: int = 2,
        
    ) -> None:
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
       
        self.root: Optional[TreeNode] = None
        self.n_features: Optional[int] = None

    @staticmethod
    def _validate_data(
        X: NDArray, y: Optional[NDArray] = None
    ) -> tuple[NDArray, Optional[NDArray]]:
    

        if X.ndim != 2 or X.shape[0] == 0 or X.shape[1] == 0:
            raise ValueError("X must be a non-empty 2D array.")
        if not np.all(np.isfinite(X)):
            raise ValueError("Features cannot contain NaN or infinity.")

        if y is None:
            return X, None

        y = np.asarray(y)
        if y.ndim != 1:
            raise ValueError("y must be a 1D array.")
        if len(X) != len(y):
            raise ValueError("X and y must contain the same number of samples.")


        return X, y

    @staticmethod
    def _entropy(y: NDArray) -> float:
        
        if len(y) == 0:
            return 0.0

        counts = Counter(y.tolist())
        total = len(y)
        entropy = 0.0

        for count in counts.values():
            probability = count / total
            entropy -= probability * math.log2(probability)

        return entropy

    def _best_split(
        self, X: NDArray, y: NDArray
    ) -> Optional[tuple[int, float]]:
        """Find a split with high positive information gain."""
        n_samples, n_features = X.shape
        parent_entropy = self._entropy(y)
        best_gain = 0.0
        best_split = None

        for feature_index in range(n_features):
            feature_values = np.unique(X[:, feature_index])
            thresholds = (
                feature_values[:-1] + feature_values[1:]
            ) / 2.0

            for threshold in thresholds:
                left_mask = X[:, feature_index] <= threshold
                n_left = int(np.sum(left_mask))
                n_right = n_samples - n_left
                
                left_entropy = self._entropy(y[left_mask])
                right_entropy = self._entropy(y[~left_mask])

                # Calculate the weighted_entropy for Information gain


                weighted_entropy = (
                    (n_left / n_samples) * left_entropy
                    + (n_right / n_samples) * right_entropy
                )
                information_gain = parent_entropy - weighted_entropy

                if information_gain > best_gain:
                    best_gain = information_gain
                    best_split = (feature_index, float(threshold))

        return best_split

    def _build_tree(
        self, X: NDArray, y: NDArray, depth: int = 0
    ) -> TreeNode:
        
        class_counts = Counter(y.tolist())

        # check the stopping criteria

       
        majority_class = class_counts.most_common(1)[0][0]
        node = TreeNode(value=majority_class)
        if (depth >= self.max_depth or len(class_counts) == 1 or len(y) < self.min_samples_split):
            return node
        

        # find the best split    

        best_split = self._best_split(X, y)
        if best_split is None:
            return node

        feature_index, threshold = best_split
        left_mask = X[:, feature_index] <= threshold

        node.value = None
        node.feature_index = feature_index
        node.threshold = threshold
        node.left = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        node.right = self._build_tree(X[~left_mask], y[~left_mask], depth + 1)
        return node

    def fit(self, X: NDArray, y: NDArray) -> "DecisionTreeClassifier":
        """ Learn a decision tree from labeled training """
        X, y = self._validate_data(X, y)
        
        self.root = self._build_tree(X, y)
        self.n_features = X.shape[1]
        return self

    def _predict_one(self, x: NDArray, node: TreeNode) -> object:
        """Traverse the tree to classify a single observation."""
        if node.is_leaf:
            return node.value

        if x[node.feature_index] <= node.threshold:
            return self._predict_one(x, node.left)
        return self._predict_one(x, node.right)

        

    def predict(self, X: NDArray) -> NDArray:
        return np.array([self._predict_one(x, self.root) for x in X])



        