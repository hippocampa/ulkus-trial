import numpy as np
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

class FeatureSelector:
    def __init__(self, n_features=100):
        self.n_features = n_features
        self.scaler = MinMaxScaler()
        self.selector = SelectKBest(score_func=chi2, k=n_features)
        self.feature_scores = None
        self.selected_feature_indices = None
        
    def fit_transform(self, X, y):
        """
        Fit the selector and transform the data
        """
        # Scale features to make them non-negative for chi-square
        X_scaled = self.scaler.fit_transform(X)
        
        # Apply chi-square feature selection
        X_selected = self.selector.fit_transform(X_scaled, y)
        
        # Store feature scores and indices
        self.feature_scores = self.selector.scores_
        self.selected_feature_indices = self.selector.get_support(indices=True)
        
        return X_selected
    
    def transform(self, X):
        """
        Transform new data using the fitted selector
        """
        X_scaled = self.scaler.transform(X)
        return self.selector.transform(X_scaled)
