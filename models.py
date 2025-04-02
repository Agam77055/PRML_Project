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