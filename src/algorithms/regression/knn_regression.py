"""
EcoTox-AI: K-Nearest Neighbors Regressor
KNN with hyperparameter tuning for distance-based predictions.
"""

import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV
import logging

logger = logging.getLogger(__name__)


def train_knn(X_train, y_train, cv=None):
    """
    Train K-Nearest Neighbors Regressor with GridSearchCV.
    
    KNN predicts toxicity based on the average LC50 of the most similar
    compounds in the training set. "Similar" is defined by the distance
    metric in the molecular descriptor space.
    
    Key hyperparameters:
    - n_neighbors: How many similar compounds to consider
    - weights: 'uniform' (equal weight) or 'distance' (closer = more weight)
    - metric: Distance function (euclidean, manhattan, minkowski)
    """
    param_grid = {
        'n_neighbors': [3, 5, 7, 9, 11, 15, 21, 25, 31],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan', 'minkowski'],
        'p': [1, 2, 3],  # Minkowski power parameter
    }
    
    model = KNeighborsRegressor()
    grid_search = GridSearchCV(
        model, param_grid,
        cv=cv if cv is not None else 10,
        scoring='r2',
        n_jobs=-1,
        verbose=0,
        return_train_score=True
    )
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    logger.info(
        f"KNN — Best params: {grid_search.best_params_}, "
        f"CV R²: {grid_search.best_score_:.4f}"
    )
    
    return {
        'model': best_model,
        'name': 'KNN Regressor',
        'best_params': grid_search.best_params_,
        'cv_score': grid_search.best_score_,
        'grid_search': grid_search,
    }
