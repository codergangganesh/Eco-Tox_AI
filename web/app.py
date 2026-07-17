"""
EcoTox-AI: Flask Web Application
Premium dark-mode web interface for toxicity predictions,
model comparison dashboard, and feature importance visualization.
"""

import os
import sys
import json
import numpy as np
from importlib import metadata
from flask import Flask, render_template, request, jsonify, send_from_directory

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.predict import predict_single, load_model, load_scaler, load_model_info
from src.data_loader import FEATURE_NAMES, TARGET_NAME

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
            for column in df.columns:
                if column.replace('Â', '') == 'R²' or column in {'R2', 'R^2'}:
                    df['R2'] = df[column]
                    break
            return df.to_dict('records')
        except Exception as e:
            print(f"Error loading comparison CSV: {e}")
            return []
    return []


def _get_metrics_timestamps():
    """Return modification timestamps for the key output files."""
    csv_path = os.path.join(PROJECT_ROOT, 'outputs', 'results', 'model_comparison.csv')
    info_path = os.path.join(PROJECT_ROOT, 'outputs', 'models', 'model_info.json')
    ts = {}
    for label, path in [('csv', csv_path), ('model_info', info_path)]:
        try:
            ts[label] = os.path.getmtime(path)
        except OSError:
            ts[label] = None
    return ts


def _get_package_versions():
    """Return installed versions for the main runtime libraries when available."""
    packages = {
        'Python': f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        'Flask': 'flask',
        'pandas': 'pandas',
        'NumPy': 'numpy',
        'scikit-learn': 'scikit-learn',
        'XGBoost': 'xgboost',
        'SHAP': 'shap',
        'Matplotlib': 'matplotlib',
        'Seaborn': 'seaborn',
        'Joblib': 'joblib',
    }
    versions = {}
    for label, package in packages.items():
        if label == 'Python':
            versions[label] = package
            continue
        try:
            versions[label] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[label] = 'Not installed'
    return versions


def _get_dataset_summary():
    """Build lightweight dataset metadata for the About page."""
    import pandas as pd

    dataset_path = os.path.join(PROJECT_ROOT, 'data', 'qsar_fish_toxicity.csv')
    summary = {
        'source_name': 'UCI QSAR Fish Toxicity',
        'source_url': 'https://archive.ics.uci.edu/dataset/504/qsar+fish+toxicity',
        'local_file': os.path.relpath(dataset_path, PROJECT_ROOT),
        'raw_records': None,
        'feature_count': len(FEATURE_NAMES),
        'target_name': TARGET_NAME,
        'target_unit': '-LOG(mol/L)',
        'organism': 'Fathead minnow (Pimephales promelas)',
        'exposure': '96-hour LC50',
        'features': FEATURE_NAMES,
        'target_stats': {},
        'feature_stats': [],
        'file_modified': None,
    }

    if os.path.exists(dataset_path):
        try:
            df = pd.read_csv(dataset_path, sep=';', header=None, names=FEATURE_NAMES + [TARGET_NAME])
            summary['raw_records'] = len(df)
            summary['file_modified'] = os.path.getmtime(dataset_path)
            summary['target_stats'] = {
                'min': float(df[TARGET_NAME].min()),
                'max': float(df[TARGET_NAME].max()),
                'mean': float(df[TARGET_NAME].mean()),
                'median': float(df[TARGET_NAME].median()),
                'std': float(df[TARGET_NAME].std()),
            }
            for feature in FEATURE_NAMES:
                summary['feature_stats'].append({
                    'name': feature,
                    'min': float(df[feature].min()),
                    'max': float(df[feature].max()),
                    'mean': float(df[feature].mean()),
                })
        except Exception as exc:
            summary['error'] = str(exc)

    links_path = os.path.join(PROJECT_ROOT, 'data', 'dataset_links.json')
    summary['linked_sources'] = []
    if os.path.exists(links_path):
        try:
            with open(links_path, 'r', encoding='utf-8') as f:
                summary['linked_sources'] = json.load(f)
        except Exception:
            summary['linked_sources'] = []

    return summary


def _get_plot_summary():
    plots_dir = os.path.join(PROJECT_ROOT, 'outputs', 'plots')
    image_extensions = {'.png', '.jpg', '.jpeg', '.svg', '.webp'}
    plots = []
    if os.path.isdir(plots_dir):
        for filename in sorted(os.listdir(plots_dir)):
            ext = os.path.splitext(filename)[1].lower()
            if ext in image_extensions:
                plots.append({
                    'filename': filename,
                    'title': os.path.splitext(filename)[0].replace('_', ' ').title(),
                })
    return plots


def build_about_metadata(model_info: dict, comparison_data: list):
    """Collect dynamic facts displayed by the expanded About section."""
    best_row = comparison_data[0] if comparison_data else {}
    metrics_timestamps = _get_metrics_timestamps()
    return {
        'project': {
            'name': 'EcotoxAI',
            'version': '',
            'release_stage': 'Training Pipeline',
            'purpose': 'AI-assisted aquatic toxicity prediction for chemical compounds.',
        },
        'dataset': _get_dataset_summary(),
        'packages': _get_package_versions(),
        'plots': _get_plot_summary(),
        'best_row': best_row,
        'model_info': model_info or {},
        'comparison_count': len(comparison_data or []),
        'metrics_timestamps': metrics_timestamps,
        'total_samples_after_preprocessing': (
            (model_info or {}).get('training_samples', 0) +
            (model_info or {}).get('test_samples', 0)
        ),
    }


@app.route('/')
def index():
    """Main prediction page."""
    model_info = get_model_info()
    comparison_data = load_comparison_data()
    about_metadata = build_about_metadata(model_info, comparison_data)
    model_loaded = get_model() is not None
    return render_template(
        'index.html',
        feature_names=FEATURE_NAMES,
        model_info=model_info,
        comparison_data=comparison_data,
        about=about_metadata,
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


@app.route('/api/metrics')
def api_metrics():
    """Return the latest metrics, model info, and file timestamps.

    Always reads from disk so the dashboard can refresh dynamically
    after a model is retrained without restarting the server.
    """
    # Force fresh read (bypass cached globals)
    comparison_data = load_comparison_data()

    # Read model_info.json directly from disk (not cached)
    info_path = os.path.join(PROJECT_ROOT, 'outputs', 'models', 'model_info.json')
    model_info = {}
    if os.path.exists(info_path):
        try:
            with open(info_path, 'r') as f:
                model_info = json.load(f)
        except Exception:
            model_info = {}

    timestamps = _get_metrics_timestamps()

    return jsonify({
        'comparison': comparison_data,
        'model_info': model_info,
        'timestamps': timestamps,
        'has_data': len(comparison_data) > 0,
    })


@app.route('/api/metrics/check')
def api_metrics_check():
    """Lightweight endpoint — returns only timestamps so the frontend
    can decide whether to fetch full metrics."""
    return jsonify(_get_metrics_timestamps())


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
    print("\n🧪 EcotoxAI Web Interface")
    print("=" * 40)
    print("Open your browser at: http://localhost:5000")
    print("=" * 40 + "\n")
    app.run(debug=True, host='0.0.0.0', port=5000)
