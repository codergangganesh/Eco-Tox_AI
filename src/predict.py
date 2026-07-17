"""
EcoTox-AI: Prediction Pipeline
Load saved models and make predictions on new chemical compounds.
Outputs LC50 value, confidence, and toxicity classification.
"""

import numpy as np
import os
import joblib
import json
import logging

from src.data_loader import FEATURE_NAMES, PROJECT_ROOT

logger = logging.getLogger(__name__)

MODELS_DIR = os.path.join(PROJECT_ROOT, 'outputs', 'models')

# EPA GHS Aquatic Acute Toxicity Classification
# LC50 in -LOG(mol/L): higher value = more toxic (lower actual concentration)
# Standard classification uses mg/L but our model predicts -log(mol/L)
TOXICITY_CLASSES = {
    'Very High Toxicity': {'min': 6.0, 'max': float('inf'), 'color': '#FF0000', 'emoji': '☠️'},
    'High Toxicity': {'min': 4.5, 'max': 6.0, 'color': '#FF6600', 'emoji': '⚠️'},
    'Moderate Toxicity': {'min': 3.0, 'max': 4.5, 'color': '#FFAA00', 'emoji': '⚡'},
    'Low Toxicity': {'min': 0.0, 'max': 3.0, 'color': '#00CC00', 'emoji': '✅'},
}


def classify_toxicity(lc50_neg_log: float) -> dict:
    """
    Classify a predicted -LOG(mol/L) LC50 value into toxicity categories.
    
    Higher -log(mol/L) means more toxic (lower actual concentration kills).
    
    Args:
        lc50_neg_log: Predicted LC50 in -LOG(mol/L)
        
    Returns:
        Dictionary with classification details
    """
    for class_name, bounds in TOXICITY_CLASSES.items():
        if bounds['min'] <= lc50_neg_log < bounds['max']:
            return {
                'class': class_name,
                'color': bounds['color'],
                'emoji': bounds['emoji'],
                'lc50_neg_log': float(lc50_neg_log),
            }
    
    # Default to low toxicity for edge cases
    return {
        'class': 'Low Toxicity',
        'color': '#00CC00',
        'emoji': '✅',
        'lc50_neg_log': float(lc50_neg_log),
    }


def load_model(model_name: str = 'best_model'):
    """
    Load a saved model from disk.
    
    Args:
        model_name: Name of the saved model file (without extension)
        
    Returns:
        Loaded model
    """
    path = os.path.join(MODELS_DIR, f'{model_name}.joblib')
    if not os.path.exists(path):
        raise FileNotFoundError(f"No saved model found at {path}")
    
    model = joblib.load(path)
    logger.info(f"Model loaded from {path}")
    return model


def load_scaler():
    """Load the saved feature scaler."""
    path = os.path.join(MODELS_DIR, 'scaler.joblib')
    if not os.path.exists(path):
        raise FileNotFoundError(f"No saved scaler found at {path}")
    return joblib.load(path)


def load_model_info() -> dict:
    """Load saved model metadata (which model was selected, its metrics, etc.)."""
    path = os.path.join(MODELS_DIR, 'model_info.json')
    if os.path.exists(path):
        with open(path, 'r') as f:
            return json.load(f)
    return {}


def format_scientific_notation(val: float) -> str:
    """Format a float into premium scientific notation with superscript exponents."""
    if val >= 0.01:
        return f"{val:.4f} mol/L"
    s = f"{val:.2e}"
    base, exp = s.split('e')
    exp = int(exp)
    superscripts = {
        '-': '⁻', '0': '⁰', '1': '¹', '2': '²', '3': '³', 
        '4': '⁴', '5': '⁵', '6': '⁶', '7': '⁷', '8': '⁸', '9': '⁹'
    }
    exp_str = "".join(superscripts.get(c, c) for c in str(exp))
    return f"{base} × 10{exp_str} mol/L"


def get_intuitive_concentration(val: float) -> str:
    """Convert molar concentration into a human-friendly mM, µM, or nM unit."""
    if val >= 1e-3:
        return f"{val * 1e3:.2f} mM (millimolar)"
    elif val >= 1e-6:
        return f"{val * 1e6:.2f} µM (micromolar)"
    else:
        return f"{val * 1e9:.2f} nM (nanomolar)"


def get_estimated_mg_L_range(val: float) -> str:
    """Estimate mg/L concentration range assuming standard organic contaminant MW of 100-300 g/mol."""
    min_mg_L = val * 100 * 1000  # MW = 100 g/mol, in mg/L
    max_mg_L = val * 300 * 1000  # MW = 300 g/mol, in mg/L
    if min_mg_L >= 100:
        return f"{min_mg_L:.1f} – {max_mg_L:.1f} mg/L (ppm)"
    elif min_mg_L >= 0.1:
        return f"{min_mg_L:.2f} – {max_mg_L:.2f} mg/L (ppm)"
    else:
        return f"{min_mg_L:.4f} – {max_mg_L:.4f} mg/L (ppm)"


ECO_HAZARD_GUIDANCE = {
    'Low Toxicity': "Practically non-toxic to aquatic life. Minimal regulatory restrictions. General environmental stewardship advised.",
    'Moderate Toxicity': "Slightly to moderately toxic. Continuous discharge may pose chronic risk. Environmental monitoring recommended.",
    'High Toxicity': "Highly toxic to aquatic organisms. Strictly regulated under EPA and GHS standards. Prevent release into storm drains or waterways.",
    'Very High Toxicity': "Extremely lethal to fish. Severe biohazard. Subject to stringent international environmental controls. Immediate containment and treatment required."
}


def predict_single(descriptors: dict, model=None, scaler=None,
                   model_name: str = 'best_model') -> dict:
    """
    Predict LC50 for a single compound given its molecular descriptors.
    
    Args:
        descriptors: Dictionary with keys matching FEATURE_NAMES
            Example: {'CIC0': 1.5, 'SM1_Dz_Z': 0.8, 'GATS1i': 0.3,
                      'NdsCH': 1, 'NdssC': 0, 'MLOGP': 2.5}
        model: Pre-loaded model (if None, loads from disk)
        scaler: Pre-loaded scaler (if None, loads from disk)
        model_name: Which saved model to use
        
    Returns:
        Dictionary with prediction results
    """
    # Load model and scaler if not provided
    if model is None:
        model = load_model(model_name)
    if scaler is None:
        scaler = load_scaler()
    
    # Validate input
    missing = [f for f in FEATURE_NAMES if f not in descriptors]
    if missing:
        raise ValueError(f"Missing descriptors: {missing}. Required: {FEATURE_NAMES}")
    
    # Create feature array in correct order
    X = np.array([[descriptors[f] for f in FEATURE_NAMES]])
    
    # Scale features
    X_scaled = scaler.transform(X)
    
    # Predict
    prediction = float(model.predict(X_scaled)[0])
    
    # Classify toxicity
    classification = classify_toxicity(prediction)
    
    # Derived human-understandable concentration metrics
    # pLC50 = -log10(LC50_mol_L) => LC50_mol_L = 10^-pLC50
    molar_lc50 = 10 ** (-prediction)
    molar_formatted = format_scientific_notation(molar_lc50)
    molar_intuitive = get_intuitive_concentration(molar_lc50)
    estimated_mgl_range = get_estimated_mg_L_range(molar_lc50)
    eco_guidance = ECO_HAZARD_GUIDANCE.get(classification['class'], ECO_HAZARD_GUIDANCE['Low Toxicity'])
    
    result = {
        'predicted_lc50_neg_log_mol_L': round(prediction, 4),
        'toxicity_classification': classification['class'],
        'toxicity_emoji': classification['emoji'],
        'toxicity_color': classification['color'],
        'molar_lc50_mol_L': molar_lc50,
        'molar_lc50_formatted': molar_formatted,
        'molar_lc50_intuitive': molar_intuitive,
        'estimated_mgl_range': estimated_mgl_range,
        'eco_hazard_guidance': eco_guidance,
        'input_descriptors': descriptors,
        'feature_order': FEATURE_NAMES,
    }
    
    return result


def predict_batch(descriptors_list: list, model=None, scaler=None,
                  model_name: str = 'best_model') -> list:
    """
    Predict LC50 for multiple compounds.
    
    Args:
        descriptors_list: List of descriptor dictionaries
        model: Pre-loaded model
        scaler: Pre-loaded scaler
        model_name: Which saved model to use
        
    Returns:
        List of prediction result dictionaries
    """
    if model is None:
        model = load_model(model_name)
    if scaler is None:
        scaler = load_scaler()
    
    results = []
    for desc in descriptors_list:
        try:
            result = predict_single(desc, model, scaler, model_name)
            results.append(result)
        except Exception as e:
            results.append({'error': str(e), 'input_descriptors': desc})
    
    return results


def format_prediction_report(result: dict) -> str:
    """Format a prediction result as a human-readable report."""
    if 'error' in result:
        return f"Error: {result['error']}"
    
    report = []
    report.append("=" * 50)
    report.append("  EcotoxAI — Toxicity Prediction")
    report.append("=" * 50)
    report.append(f"  Predicted LC50: {result['predicted_lc50_neg_log_mol_L']:.4f} [-LOG(mol/L)]")
    report.append(f"  Classification: {result['toxicity_emoji']} {result['toxicity_classification']}")
    report.append("-" * 50)
    report.append("  Input Descriptors:")
    for feat, val in result['input_descriptors'].items():
        report.append(f"    {feat}: {val}")
    report.append("=" * 50)
    
    return "\n".join(report)


if __name__ == '__main__':
    # Example prediction
    example_descriptors = {
        'CIC0': 1.732,
        'SM1_Dz_Z': 1.393,
        'GATS1i': 0.849,
        'NdsCH': 1,
        'NdssC': 0,
        'MLOGP': 2.246,
    }
    
    try:
        result = predict_single(example_descriptors)
        print(format_prediction_report(result))
    except FileNotFoundError:
        print("No trained model found. Run train.py first.")
