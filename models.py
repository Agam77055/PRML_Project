import re
import os
import json
import nltk
import numpy as np
import wn
from collections import Counter
from nltk.corpus import wordnet

# Scikit-learn and SciPy imports
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score
from scipy.sparse import issparse
from joblib import Parallel, delayed

# Ensure NLTK resources are downloaded (if not already)
nltk.download('stopwords')
nltk.download('punkt')

class MultinomialNBFromScratch(BaseEstimator, ClassifierMixin):
    def __init__(self, alpha=1.0):
        self.alpha = alpha  # Smoothing parameter

    def fit(self, X, y):
        X, y = check_X_y(X, y, accept_sparse=True)
        self.classes_ = unique_labels(y)
        n_classes = len(self.classes_)
        n_features = X.shape[1]

        # Convert sparse matrix to dense if needed
        X_array = X.toarray() if issparse(X) else X

        # Ensure all values are non-negative for Multinomial NB
        if np.any(X_array < 0):
            raise ValueError("Input X must be non-negative for MultinomialNB")

        # Initialize parameters
        self.class_count_ = np.zeros(n_classes, dtype=np.float64)
        self.feature_count_ = np.zeros((n_classes, n_features), dtype=np.float64)

        # Count samples in each class
        for i, c in enumerate(self.classes_):
            X_c = X_array[y == c]
            self.class_count_[i] = X_c.shape[0]
            self.feature_count_[i] = np.sum(X_c, axis=0)

        # Calculate log probabilities with smoothing
        smoothed_fc = self.feature_count_ + self.alpha
        smoothed_cc = np.sum(smoothed_fc, axis=1)

        self.feature_log_prob_ = np.log(smoothed_fc) - np.log(smoothed_cc.reshape(-1, 1))
        self.class_log_prior_ = np.log(self.class_count_) - np.log(np.sum(self.class_count_))

        return self

    def predict_log_proba(self, X):
        check_is_fitted(self, ['feature_log_prob_', 'class_log_prior_'])
        X = check_array(X, accept_sparse=True)

        # Convert sparse matrix to dense if needed
        X_array = X.toarray() if issparse(X) else X

        # Calculate log probabilities for each class
        joint_log_likelihood = np.zeros((X_array.shape[0], len(self.classes_)))

        for i, c in enumerate(self.classes_):
            joint_log_likelihood[:, i] = self.class_log_prior_[i]
            joint_log_likelihood[:, i] += np.dot(X_array, self.feature_log_prob_[i])

        # Normalize to get probabilities (using logsumexp trick for numerical stability)
        log_prob_x = np.max(joint_log_likelihood, axis=1)
        log_prob_x_adjusted = log_prob_x + np.log(np.sum(
            np.exp(joint_log_likelihood - log_prob_x.reshape(-1, 1)), axis=1
        ))
        log_probas = joint_log_likelihood - log_prob_x_adjusted.reshape(-1, 1)

        return log_probas

    def predict_proba(self, X):
        return np.exp(self.predict_log_proba(X))

    def predict(self, X):
        log_probas = self.predict_log_proba(X)
        return self.classes_[np.argmax(log_probas, axis=1)]

class DecisionTreeClassifierFromScratch(BaseEstimator, ClassifierMixin):
    def _init_(self, max_depth=None, min_samples_split=2, min_samples_leaf=1,
                 criterion='gini', max_features=None, random_state=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.criterion = criterion
        self.max_features = max_features
        self.random_state = random_state
        self.tree_ = None
        self.feature_importances_ = None

    class Node:
        def _init_(self, feature=None, threshold=None, left=None, right=None, value=None):
            self.feature = feature  # Which feature to split on
            self.threshold = threshold  # Threshold value for the split
            self.left = left  # Left child
            self.right = right  # Right child
            self.value = value  # Class distribution or prediction

    def _gini(self, y):
        """Calculate Gini impurity"""
        m = len(y)
        if m == 0:
            return 0

        # Use bincount with proper handling of class indices
        counts = np.bincount(y, minlength=len(self.classes_))
        probas = counts / m
        return 1 - np.sum(probas**2)

    def _entropy(self, y):
        """Calculate entropy"""
        m = len(y)
        if m == 0:
            return 0

        # Use bincount with proper handling of class indices
        counts = np.bincount(y, minlength=len(self.classes_))
        probas = counts / m
        # Avoid log(0) by filtering zero probabilities
        nonzero_probas = probas[probas > 0]
        return -np.sum(nonzero_probas * np.log2(nonzero_probas))

    def _calculate_impurity(self, y):
        """Calculate impurity based on the criterion"""
        if self.criterion == 'gini':
            return self._gini(y)
        elif self.criterion == 'entropy':
            return self._entropy(y)
        else:
            raise ValueError(f"Unknown criterion: {self.criterion}")

    def _best_split(self, X, y, features_to_consider):
        """Find the best split for the data"""
        m, n = X.shape
        if m <= 1:
            return None, None

        # Get current impurity
        current_impurity = self._calculate_impurity(y)

        best_gain = 0.0
        best_feature, best_threshold = None, None

        # Only consider the randomly selected features
        for feature in features_to_consider:
            # Get unique values for the feature
            feature_values = X[:, feature]

            # Check if all values are identical
            if np.all(feature_values == feature_values[0]):
                continue

            # Use a more efficient approach for finding potential thresholds
            # Instead of using every unique value, sample a reasonable number of thresholds
            # For small datasets, still use all unique values
            unique_values = np.unique(feature_values)
            if len(unique_values) > 10:
                # For large number of unique values, use percentiles instead
                percentiles = np.linspace(5, 95, 10)
                thresholds = np.percentile(feature_values, percentiles)
            else:
                thresholds = unique_values

            # Try each threshold
            for threshold in thresholds:
                # Split the data
                left_indices = X[:, feature] <= threshold
                right_indices = ~left_indices

                # Skip if either side has fewer than min_samples_leaf
                if np.sum(left_indices) < self.min_samples_leaf or np.sum(right_indices) < self.min_samples_leaf:
                    continue

                # Calculate impurity for each side
                left_impurity = self._calculate_impurity(y[left_indices])
                right_impurity = self._calculate_impurity(y[right_indices])

                # Calculate the weighted average impurity
                n_left, n_right = np.sum(left_indices), np.sum(right_indices)
                weighted_impurity = (n_left / m) * left_impurity + (n_right / m) * right_impurity

                # Calculate information gain
                gain = current_impurity - weighted_impurity

                # Update best split if this is better
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold

        return best_feature, best_threshold

    def _select_features(self, n_features, random_state=None):
        """Select features to consider for splitting"""
        if self.max_features is None:
            return np.arange(n_features)

        rng = np.random.RandomState(random_state)

        if isinstance(self.max_features, int):
            n_features_to_consider = min(self.max_features, n_features)
        elif isinstance(self.max_features, float):
            n_features_to_consider = max(1, int(self.max_features * n_features))
        elif self.max_features == 'sqrt':
            n_features_to_consider = max(1, int(np.sqrt(n_features)))
        elif self.max_features == 'log2':
            n_features_to_consider = max(1, int(np.log2(n_features)))
        else:
            n_features_to_consider = n_features

        return rng.choice(n_features, size=n_features_to_consider, replace=False)

    def _build_tree(self, X, y, depth=0, node_idx=0, feature_importances=None):
        """Recursively build the decision tree"""
        n_samples, n_features = X.shape

        # Convert class labels to indices for proper bincount
        y_idx = np.searchsorted(self.classes_, y)

        # Get the class distribution (use the class indices to ensure proper counts)
        value = np.bincount(y_idx, minlength=len(self.classes_))

        # Create a leaf node if stopping criteria are met
        if (self.max_depth is not None and depth >= self.max_depth) or \
           n_samples < self.min_samples_split or \
           n_samples < 2 * self.min_samples_leaf or \
           np.unique(y).size == 1:
            return self.Node(value=value)

        # Select features to consider (for random forest)
        features_to_consider = self._select_features(
            n_features,
            random_state=self.random_state + node_idx if self.random_state is not None else None
        )

        # Find the best split
        best_feature, best_threshold = self._best_split(X, y_idx, features_to_consider)

        # If no good split was found, create a leaf node
        if best_feature is None:
            return self.Node(value=value)

        # Split the data
        left_indices = X[:, best_feature] <= best_threshold
        right_indices = ~left_indices

        # Track feature importance
        if feature_importances is not None:
            # Compute node impurity reduction and weight by node size
            current_impurity = self._calculate_impurity(y_idx)
            left_impurity = self._calculate_impurity(y_idx[left_indices])
            right_impurity = self._calculate_impurity(y_idx[right_indices])

            n_left, n_right = np.sum(left_indices), np.sum(right_indices)
            weighted_impurity = (n_left / n_samples) * left_impurity + (n_right / n_samples) * right_impurity

            # Update feature importance
            impurity_decrease = current_impurity - weighted_impurity
            feature_importances[best_feature] += impurity_decrease

        # Build subtrees
        left_subtree = self._build_tree(
            X[left_indices], y[left_indices],
            depth + 1, node_idx * 2 + 1, feature_importances
        )
        right_subtree = self._build_tree(
            X[right_indices], y[right_indices],
            depth + 1, node_idx * 2 + 2, feature_importances
        )

        return self.Node(
            feature=best_feature,
            threshold=best_threshold,
            left=left_subtree,
            right=right_subtree,
            value=value
        )

    def fit(self, X, y):
        # Check that X and y have correct shape and convert sparse to dense
        X, y = check_X_y(X, y, accept_sparse=True)

        # Convert sparse matrix to dense if needed
        X_array = X.toarray() if issparse(X) else X

        self.classes_ = unique_labels(y)

        # Initialize feature importances array
        self.feature_importances_ = np.zeros(X_array.shape[1])

        # Set random seed if provided
        if self.random_state is not None:
            np.random.seed(self.random_state)

        # Build the tree
        self.tree_ = self.build_tree(X_array, y, feature_importances=self.feature_importances)

        # Normalize feature importances to sum to 1
        if np.sum(self.feature_importances_) > 0:
            self.feature_importances_ /= np.sum(self.feature_importances_)

        return self

    def _predict_one(self, x, node):
        """Predict for a single instance"""
        if node.feature is None:  # Leaf node
            # Handle division by zero safely
            if np.sum(node.value) == 0:
                # Return uniform distribution if value is all zeros
                return np.ones(len(self.classes_)) / len(self.classes_)
            return node.value / np.sum(node.value)  # Return probability distribution

        if x[node.feature] <= node.threshold:
            return self._predict_one(x, node.left)
        else:
            return self._predict_one(x, node.right)

    def predict_proba(self, X):
        check_is_fitted(self, ['tree_', 'classes_'])
        X = check_array(X, accept_sparse=True)

        # Convert sparse matrix to dense if needed
        X_array = X.toarray() if issparse(X) else X

        # Make predictions for each instance
        probas = np.array([self.predict_one(x, self.tree) for x in X_array])

        return probas

    def predict(self, X):
        probas = self.predict_proba(X)
        return self.classes_[np.argmax(probas, axis=1)]