"""
EcoTox-AI: Flask Web Application
Premium dark-mode web interface for toxicity predictions,
model comparison dashboard, and feature importance visualization.
"""

import os
import sys
import json
import numpy as np
from flask import Flask, render_template, request, jsonify, send_from_directory

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.predict import predict_single, load_model, load_scaler, load_model_info
from src.data_loader import FEATURE_NAMES

app = Flask(__name__)

# Load model and scaler once at startup
_model = None
_scaler = None
_model_info = None


def get_model():
    global _model
    if _model is None:
        try:
            _model = load_model('best_model')
        except FileNotFoundError:
            return None
    return _model


def get_scaler():
    global _scaler
    if _scaler is None:
        try:
            _scaler = load_scaler()
        except FileNotFoundError:
            return None
    return _scaler


def get_model_info():
    global _model_info
    if _model_info is None:
        _model_info = load_model_info()
    return _model_info


def load_comparison_data():
    """Load model comparison data from CSV."""
    import pandas as pd
    csv_path = os.path.join(PROJECT_ROOT, 'outputs', 'results', 'model_comparison.csv')
    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path, index_col=0, encoding='utf-8')
            return df.to_dict('records')
        except Exception as e:
            print(f"Error loading comparison CSV: {e}")
            return []
    return []


@app.route('/')
def index():
    """Main prediction page."""
    model_info = get_model_info()
    comparison_data = load_comparison_data()
    model_loaded = get_model() is not None
    return render_template(
        'index.html',
        feature_names=FEATURE_NAMES,
        model_info=model_info,
        comparison_data=comparison_data,
        model_loaded=model_loaded
    )


@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction request."""
    try:
        model = get_model()
        scaler = get_scaler()
        
        if model is None or scaler is None:
            return jsonify({
                'error': 'Model not loaded. Please run train.py first.'
            }), 500
        
        # Parse input descriptors
        descriptors = {}
        for feat in FEATURE_NAMES:
            value = request.form.get(feat) or request.json.get(feat) if request.is_json else request.form.get(feat)
            if value is None or value == '':
                return jsonify({'error': f'Missing descriptor: {feat}'}), 400
            try:
                descriptors[feat] = float(value)
            except ValueError:
                return jsonify({'error': f'Invalid value for {feat}: {value}'}), 400
        
        # Make prediction
        result = predict_single(descriptors, model, scaler)
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/predict', methods=['POST'])
def api_predict():
    """REST API endpoint for programmatic access."""
    return predict()


@app.route('/models')
def models_dashboard():
    """Model comparison dashboard (returns JSON for AJAX)."""
    comparison_data = load_comparison_data()
    model_info = get_model_info()
    return jsonify({
        'comparison': comparison_data,
        'model_info': model_info
    })


@app.route('/api/plots')
def api_plots():
    """Return list of available plot images as JSON."""
    plots_dir = os.path.join(PROJECT_ROOT, 'outputs', 'plots')
    image_extensions = {'.png', '.jpg', '.jpeg', '.svg', '.webp'}
    plots = []
    if os.path.isdir(plots_dir):
        for f in sorted(os.listdir(plots_dir)):
            ext = os.path.splitext(f)[1].lower()
            if ext in image_extensions:
                try:
                    stat = os.stat(os.path.join(plots_dir, f))
                    title = os.path.splitext(f)[0].replace('_', ' ').title()
                    plots.append({
                        'filename': f,
                        'url': f'/outputs/plots/{f}',
                        'title': title,
                        'modified': stat.st_mtime,
                        'size': stat.st_size,
                    })
                except OSError:
                    continue
    return jsonify({'plots': plots, 'count': len(plots)})


@app.route('/outputs/plots/<path:filename>')
def serve_plot(filename):
    """Serve training plots from the outputs/plots directory."""
    plots_dir = os.path.join(PROJECT_ROOT, 'outputs', 'plots')
    return send_from_directory(plots_dir, filename)


if __name__ == '__main__':
    print("\n🧪 EcoTox-AI Web Interface")
    print("=" * 40)
    print("Open your browser at: http://localhost:5000")
    print("=" * 40 + "\n")
    app.run(debug=True, host='0.0.0.0', port=5000)
