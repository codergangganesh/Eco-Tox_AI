"""
EcoTox-AI: Deep Neural Network (MLP)
Multi-layer perceptron regressor using scikit-learn.
"""

import numpy as np
import os
import logging

logger = logging.getLogger(__name__)


def build_mlp_model(input_dim: int, 
                    hidden_layers: list = None,
                    dropout_rates: list = None,
                    learning_rate: float = 0.001,
                    epochs: int = 500,
                    patience: int = 30,
                    batch_size: int = 32):
    """
    Build a multi-layer perceptron (MLP) regressor using scikit-learn's MLPRegressor.
    
    Args:
        input_dim: Number of input features (ignored, kept for compatibility)
        hidden_layers: List of hidden layer sizes (default: [128, 64, 32])
        dropout_rates: Dropout rates (ignored by MLPRegressor, L2 regularization is used instead)
        learning_rate: Initial learning rate for Adam optimizer
        epochs: Maximum number of epochs (max_iter)
        patience: Patience for early stopping
        batch_size: Size of minibatches for optimization
        
    Returns:
        Unfitted MLPRegressor model
    """
    from sklearn.neural_network import MLPRegressor
    
    if hidden_layers is None:
        hidden_layers = [128, 64, 32]
        
    hidden_layer_sizes = tuple(hidden_layers)
    
    # We map learning_rate to learning_rate_init, epochs to max_iter
    model = MLPRegressor(
        hidden_layer_sizes=hidden_layer_sizes,
        activation='relu',
        solver='adam',
        alpha=0.001,  # Equivalent to regularizers.l2(0.001)
        batch_size=batch_size,
        learning_rate_init=learning_rate,
        max_iter=epochs,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=patience,
        random_state=42
    )
    
    return model


def train_neural_network(X_train, y_train, X_val=None, y_val=None, cv=None,
                         epochs: int = 500, batch_size: int = 32,
                         patience: int = 30):
    """
    Train the MLPRegressor with early stopping.
    
    Uses multiple architectures and selects the best one via
    validation performance.
    
    Args:
        X_train: Training features (should be scaled)
        y_train: Training targets
        X_val: Validation features (optional, MLPRegressor will split internally if None)
        y_val: Validation targets (optional)
        cv: Cross-validation splitter (unused during architecture search, kept for consistency)
        epochs: Maximum training epochs
        batch_size: Training batch size
        patience: Early stopping patience
        
    Returns:
        Result dictionary with trained model and metrics
    """
    logger.info("=" * 50)
    logger.info("Training Deep Neural Network (MLP via scikit-learn)...")
    logger.info("=" * 50)
    
    input_dim = X_train.shape[1]
    
    # If explicit validation set is provided, we can use it, otherwise MLPRegressor will split internally.
    # To keep identical split behavior to the original, if X_val is None, we split here
    # so we can compute validation score on the same validation set for all architectures.
    if X_val is None or y_val is None:
        from sklearn.model_selection import train_test_split
        X_train_nn, X_val, y_train_nn, y_val = train_test_split(
            X_train, y_train, test_size=0.15, random_state=42
        )
    else:
        X_train_nn = X_train
        y_train_nn = y_train
        
    # Try multiple architectures
    architectures = [
        {'hidden_layers': [128, 64, 32], 'dropout_rates': [0.3, 0.2, 0.1], 'lr': 0.001},
        {'hidden_layers': [256, 128, 64, 32], 'dropout_rates': [0.4, 0.3, 0.2, 0.1], 'lr': 0.0005},
        {'hidden_layers': [64, 32, 16], 'dropout_rates': [0.2, 0.15, 0.1], 'lr': 0.001},
        {'hidden_layers': [128, 128, 64], 'dropout_rates': [0.3, 0.25, 0.2], 'lr': 0.0008},
    ]
    
    best_val_r2 = -float('inf')
    best_model = None
    best_arch = None
    
    for i, arch in enumerate(architectures):
        logger.info(f"  Architecture {i+1}/{len(architectures)}: {arch['hidden_layers']}")
        
        model = build_mlp_model(
            input_dim=input_dim,
            hidden_layers=arch['hidden_layers'],
            dropout_rates=arch['dropout_rates'],
            learning_rate=arch['lr'],
            epochs=epochs,
            patience=patience,
            batch_size=batch_size
        )
        
        # Train MLPRegressor (it will use early stopping internally on X_train_nn)
        model.fit(X_train_nn, y_train_nn)
        
        # Evaluate on the validation set (R2 score)
        val_r2 = model.score(X_val, y_val)
        actual_epochs = model.n_iter_
        
        logger.info(f"    Val R²: {val_r2:.4f}, Epochs: {actual_epochs}")
        
        if val_r2 > best_val_r2:
            best_val_r2 = val_r2
            best_model = model
            best_arch = arch
            
    logger.info(f"  Best architecture: {best_arch['hidden_layers']}, Val R²: {best_val_r2:.4f}")
    
    # Compute R² on validation set for the best model
    val_r2 = best_model.score(X_val, y_val)
    logger.info(f"  Validation R²: {val_r2:.4f}")
    
    return {
        'model': best_model,
        'name': 'Deep MLP (Neural Network)',
        'best_params': best_arch,
        'cv_score': val_r2,
        'history': None,
        'val_loss': 1.0 - val_r2,  # R² mapped to a loss metric for consistency
        'is_keras': False,         # Now scikit-learn based
    }


def save_nn_model(model, filepath: str):
    """Save an MLPRegressor model to disk using joblib."""
    import joblib
    if filepath.endswith('.keras'):
        filepath = filepath.replace('.keras', '.joblib')
    joblib.dump(model, filepath)
    logger.info(f"Neural network saved to {filepath}")


def load_nn_model(filepath: str):
    """Load a saved MLPRegressor model from disk using joblib."""
    import joblib
    if filepath.endswith('.keras'):
        filepath = filepath.replace('.keras', '.joblib')
    if not os.path.exists(filepath) and os.path.exists(filepath.replace('.joblib', '.keras')):
        filepath = filepath.replace('.joblib', '.keras')
    model = joblib.load(filepath)
    logger.info(f"Neural network loaded from {filepath}")
    return model
