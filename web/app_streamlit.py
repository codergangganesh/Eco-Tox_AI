import os
import sys
import json
import numpy as np
import pandas as pd
import streamlit as st

# Add project root to python path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.predict import (
    predict_single,
    load_model,
    load_scaler,
    load_model_info,
    TOXICITY_CLASSES,
    ECO_HAZARD_GUIDANCE
)
from src.data_loader import FEATURE_NAMES

# Page configuration
st.set_page_config(
    page_title="EcoTox-AI | Aquatic Toxicity Hub",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom premium styling using HTML/CSS
st.markdown("""
<style>
    /* Global Background and Text Color */
    .reportview-container {
        background: #0f172a;
        color: #f1f5f9;
    }
    
    /* Neon glow card container */
    .premium-card {
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(99, 102, 241, 0.15);
        border-radius: 12px;
        padding: 24px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        backdrop-filter: blur(10px);
        margin-bottom: 20px;
        transition: all 0.3s ease;
    }
    
    .premium-card:hover {
        border-color: rgba(99, 102, 241, 0.4);
        box-shadow: 0 8px 32px 0 rgba(99, 102, 241, 0.15);
    }
    
    /* Radial Glow for Toxicity prediction */
    .result-glow {
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        margin-top: 15px;
        border: 2px solid;
        animation: pulse 2s infinite alternate;
    }
    
    @keyframes pulse {
        0% { transform: scale(1.0); }
        100% { transform: scale(1.02); }
    }
    
    .metric-value {
        font-family: 'JetBrains Mono', monospace;
        font-weight: 700;
        font-size: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# Helper function to get timestamps
def get_modified_time(path):
    if os.path.exists(path):
        import datetime
        t = os.path.getmtime(path)
        return datetime.datetime.fromtimestamp(t).strftime('%Y-%m-%d %H:%M:%S')
    return "N/A"

# Title bar
st.markdown("<h1 style='text-align: center; color: #818cf8; margin-bottom: 5px;'>🧪 EcoTox-AI</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #94a3b8; font-size: 1.2rem; margin-bottom: 30px;'>Aquatic Toxicity Predictor & Model Benchmarking Hub</p>", unsafe_allow_html=True)

# Navigation tabs
tab_pred, tab_dash, tab_about = st.tabs(["⚗️ Toxicity Predictor", "📊 Benchmarks & Insights", "ℹ️ Model Info & Dataset"])

# Load model, scaler, and info
@st.cache_resource
def load_assets():
    try:
        model = load_model('best_model')
        scaler = load_scaler()
        info = load_model_info()
        return model, scaler, info
    except Exception as e:
        st.error(f"Error loading model assets: {e}")
        return None, None, {}

model, scaler, model_info = load_assets()

# TAB 1: Predictor
with tab_pred:
    st.header("Predict Aquatic Toxicity (LC50)")
    st.write("Adjust the molecular descriptors below to predict toxicity for fathead minnow (*Pimephales promelas*).")

    if model is None or scaler is None:
        st.warning("⚠️ Best model or scaler is missing. Please run `python train.py` first to train and generate the model artifacts.")
    else:
        # Pre-populate sample descriptors when requested
        example_values = {
            'CIC0': 1.732,
            'SM1_Dz_Z': 1.393,
            'GATS1i': 0.849,
            'NdsCH': 1.0,
            'NdssC': 0.0,
            'MLOGP': 2.246
        }
        
        # Load sample values button
        if "loaded_values" not in st.session_state:
            st.session_state.loaded_values = example_values.copy()
            
        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            if st.button("📋 Load Example Compound (Highly Toxic Phenol Derivative)"):
                st.session_state.loaded_values = example_values.copy()
                st.rerun()
        with col2:
            if st.button("🔄 Reset to Zeroes"):
                st.session_state.loaded_values = {f: 0.0 for f in FEATURE_NAMES}
                st.rerun()
                
        # Split inputs into two columns
        input_col1, input_col2 = st.columns(2)
        
        descriptors = {}
        with input_col1:
            st.markdown("### Topological & Autocorrelation Descriptors")
            descriptors['CIC0'] = st.number_input(
                "CIC0 (Complementary Information Content, index of neighborhood symmetry)",
                min_value=0.0, max_value=10.0,
                value=float(st.session_state.loaded_values['CIC0']),
                step=0.001, format="%.3f"
            )
            descriptors['SM1_Dz_Z'] = st.number_input(
                "SM1_Dz(Z) (Spectral moment of topological distance matrix, weighted by atomic number)",
                min_value=0.0, max_value=10.0,
                value=float(st.session_state.loaded_values['SM1_Dz_Z']),
                step=0.001, format="%.3f"
            )
            descriptors['GATS1i'] = st.number_input(
                "GATS1i (Geary autocorrelation of lag 1, weighted by ionization potential)",
                min_value=0.0, max_value=10.0,
                value=float(st.session_state.loaded_values['GATS1i']),
                step=0.001, format="%.3f"
            )
            
        with input_col2:
            st.markdown("### Structural & Lipophilicity Descriptors")
            descriptors['NdsCH'] = st.number_input(
                "NdsCH (Number of single-bonded, unsubstituted =CH- fragment groups)",
                min_value=0.0, max_value=20.0,
                value=float(st.session_state.loaded_values['NdsCH']),
                step=1.0, format="%.0f"
            )
            descriptors['NdssC'] = st.number_input(
                "NdssC (Number of double-bonded, unsubstituted =C< fragment groups)",
                min_value=0.0, max_value=20.0,
                value=float(st.session_state.loaded_values['NdssC']),
                step=1.0, format="%.0f"
            )
            descriptors['MLOGP'] = st.number_input(
                "MLOGP (Moriguchi Octanol-Water Partition Coefficient - Lipophilicity)",
                min_value=-5.0, max_value=15.0,
                value=float(st.session_state.loaded_values['MLOGP']),
                step=0.001, format="%.3f"
            )
            
        # Prediction triggers
        if st.button("🔬 Predict Toxicity", type="primary", use_container_width=True):
            with st.spinner("Analyzing chemical toxicity profile..."):
                res = predict_single(descriptors, model, scaler)
                
                # Render results card
                color = res['toxicity_color']
                emoji = res['toxicity_emoji']
                tox_class = res['toxicity_classification']
                
                st.markdown(f"""
                <div class="result-glow" style="border-color: {color}; background-color: {color}12;">
                    <h2 style="color: {color}; margin-top: 0;">{emoji} {tox_class}</h2>
                    <p style="font-size: 1.1rem; max-width: 800px; margin: 10px auto; color: #cbd5e1;">{res['eco_hazard_guidance']}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Detailed stats columns
                res_col1, res_col2, res_col3 = st.columns(3)
                
                with res_col1:
                    st.metric(
                        label="Predicted pLC50 (-LOG(mol/L))", 
                        value=f"{res['predicted_lc50_neg_log_mol_L']:.4f}",
                        help="Higher values signify greater toxicity (requires a lower concentration to cause lethality)."
                    )
                with res_col2:
                    st.metric(
                        label="Molar LC50", 
                        value=res['molar_lc50_intuitive'],
                        help="The predicted molar concentration causing 50% mortality."
                    )
                with res_col3:
                    st.metric(
                        label="Estimated Concentration Range",
                        value=res['estimated_mgl_range'],
                        help="Estimated mass-based concentration in mg/L (ppm) assuming standard organic molecular weights (100–300 g/mol)."
                    )
                    
                # Relative Benchmark
                st.markdown("### 📍 Relative Toxicity Reference Scale")
                
                # Rotenone: 6.80, Malathion: 5.50, Phenol: 3.20
                user_val = res['predicted_lc50_neg_log_mol_L']
                
                benchmarks = [
                    {"name": "Rotenone (Insecticide)", "val": 6.80, "cat": "Very High Toxicity ☠️"},
                    {"name": "PREDICTED COMPOUND", "val": user_val, "cat": f"{tox_class} {emoji}", "is_user": True},
                    {"name": "Malathion (Pesticide)", "val": 5.50, "cat": "High Toxicity ⚠️"},
                    {"name": "Phenol (Disinfectant)", "val": 3.20, "cat": "Moderate Toxicity ⚡"},
                    {"name": "Methanol (Industrial Solvent)", "val": 0.50, "cat": "Low Toxicity ✅"}
                ]
                # Sort benchmarks by value descending
                benchmarks = sorted(benchmarks, key=lambda x: x['val'], reverse=True)
                
                for bm in benchmarks:
                    if bm.get('is_user'):
                        st.markdown(f"""
                        <div style="background-color: {color}22; padding: 12px; border-left: 5px solid {color}; border-radius: 4px; margin: 8px 0;">
                            <strong>🔥 PREDICTED COMPOUND: {bm['val']:.4f} pLC50</strong> — <span style="color: {color}; font-weight: bold;">{bm['cat']}</span>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div style="background-color: rgba(255,255,255,0.03); padding: 10px; border-left: 3px solid #94a3b8; border-radius: 4px; margin: 6px 0; color: #94a3b8;">
                            <strong>{bm['name']}</strong>: {bm['val']:.2f} pLC50 ({bm['cat']})
                        </div>
                        """, unsafe_allow_html=True)

# TAB 2: Dashboard/Comparison
with tab_dash:
    st.header("Model Performance & Comparison")
    st.write("Compare results of all trained classifiers and regressors across standard metrics (R², RMSE, MAE).")
    
    # Load comparison CSV
    csv_path = os.path.join(PROJECT_ROOT, 'outputs', 'results', 'model_comparison.csv')
    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path, index_col=0)
            st.markdown("### 🏆 Performance Leaderboard")
            st.dataframe(df, use_container_width=True)
            
            # Simple bar plot of R2 score
            st.markdown("### 📊 R² Metric Comparison")
            # Find R² column dynamically
            r2_col = None
            for col in df.columns:
                if 'r2' in col.lower() or 'r²' in col.lower():
                    r2_col = col
                    break
            
            if r2_col:
                st.bar_chart(df[r2_col])
            else:
                st.info("R² column not found in model comparison results.")
        except Exception as e:
            st.error(f"Error loading model comparison results: {e}")
    else:
        st.info("No model comparison file found. Please make sure the models have been evaluated.")
        
    # Render static training plots if they exist
    plots_dir = os.path.join(PROJECT_ROOT, 'outputs', 'plots')
    if os.path.isdir(plots_dir):
        st.markdown("### 📈 Pipeline Visualizations")
        plot_files = [f for f in sorted(os.listdir(plots_dir)) if f.endswith(('.png', '.webp'))]
        
        if plot_files:
            selected_plot = st.selectbox("Select plot to inspect:", plot_files)
            if selected_plot:
                plot_path = os.path.join(plots_dir, selected_plot)
                st.image(plot_path, caption=selected_plot.replace('_', ' ').replace('.png', '').title(), use_container_width=True)
        else:
            st.info("No pipeline plots found in outputs/plots.")

# TAB 3: About & Dataset Metadata
with tab_about:
    st.header("Dataset & Pipeline Information")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 📂 UCI Dataset Characteristics")
        st.write("""
        The model uses the **QSAR fish toxicity dataset** sourced from the UCI Machine Learning Repository.
        
        - **Total chemical compounds:** 908
        - **Exposure time:** 96-hour LC50 (Concentration causing mortality in 50% of the test organism)
        - **Organism:** Fathead minnow (*Pimephales promelas*)
        - **Target Unit:** -LOG(mol/L) (higher is more toxic)
        """)
        
        dataset_path = os.path.join(PROJECT_ROOT, 'data', 'qsar_fish_toxicity.csv')
        if os.path.exists(dataset_path):
            st.success(f"✅ Local dataset found: `{os.path.basename(dataset_path)}`")
            st.info(f"Last modified: {get_modified_time(dataset_path)}")
        else:
            st.warning("⚠️ Local dataset CSV not found. It will be downloaded automatically during model training.")

    with col2:
        st.markdown("### ⚙️ Production Model Metadata")
        if model_info:
            st.json(model_info)
        else:
            st.info("No model metadata details found in model_info.json.")
            
    st.markdown("---")
    st.markdown("##### EcoTox-AI Ecosystem • Powered by Streamlit & Scikit-Learn")
