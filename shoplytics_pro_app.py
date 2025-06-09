# shoplytics_pro_app.py
# Version 2.1 - Correction des erreurs de type et de nom de colonne.

# --- 1. Imports ---
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
import lightgbm as lgb
import plotly.express as px
import plotly.graph_objects as go
import io
import base64
import tempfile
import zipfile
import os
import json
import datetime
from fpdf import FPDF
from PIL import Image

st.set_page_config(
    page_title="Shoplytics Pro 2.0 | Plateforme d'Analyse IA",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 3. Dictionnaires de Langue ---
LANGUAGES = {
    "Français": {
        "page_title": "🛍️ Shoplytics Pro 2.0",
        "page_subtitle": "Plateforme d'analyse prédictive du comportement client, optimisée par l'IA.",
        "data_import_header": "📂 Chargement des Données",
        "uploader_label": "Importer un ou plusieurs fichiers CSV",
        "use_sample_label": "Utiliser le jeu de données d'exemple",
        "run_analysis_button": "🚀 Lancer l'Analyse",
        "no_data_warning": "Veuillez importer un fichier ou utiliser l'exemple pour commencer.",
        "analysis_intro": "### Bienvenue sur Shoplytics Pro 2.0\n\nChargez vos données et cliquez sur **'Lancer l'Analyse'** pour démarrer la magie prédictive.",
        "model_config_header": "⚙️ Configuration du Modèle",
        "model_choice_label": "Choisir le modèle de prédiction",
        "tab_overview": "📊 Aperçu des Données",
        "tab_eda": "📈 Analyse Exploratoire (EDA)",
        "tab_modeling": "🧠 Modélisation Prédictive",
        "tab_simulation": "🧪 Simulation Client",
        "tab_export": "📤 Export & Rapports",
        "overview_title": "Aperçu des Données Chargées",
        "overview_shape": "Dimensions du jeu de données :",
        "overview_stats": "Statistiques descriptives :",
        "eda_title": "Analyse Exploratoire Interactive",
        "eda_dist_title": "Distribution de la variable cible",
        "eda_corr_title": "Matrice de Corrélation des variables numériques",
        "modeling_title": "Résultats du Modèle Prédictif",
        "modeling_metrics_title": "Métriques de Performance du Modèle",
        "modeling_rmse": "RMSE (Erreur Quadratique Moyenne)",
        "modeling_r2": "Score R² (Coefficient de Détermination)",
        "modeling_preds_vs_actual": "Prédictions vs Valeurs Réelles",
        "modeling_shap_title": "Interprétabilité du Modèle (SHAP)",
        "modeling_shap_global": "Importance Globale des Variables",
        "modeling_shap_local": "Analyse d'une Prédiction Individuelle",
        "modeling_shap_select_instance": "Choisissez une observation du jeu de test pour l'analyser :",
        "simulation_title": "🧪 Simuler le montant d'achat pour un nouveau client",
        "simulation_expander": "🔧 Remplir le profil pour une prédiction",
        "simulation_result": "💰 Le montant d’achat estimé est de :",
        "export_title": "Exporter les Résultats",
        "export_excel_button": "📥 Télécharger les prédictions (Excel)",
        "export_pdf_button": "📄 Générer le Rapport PDF",
        "export_zip_button": "📦 Générer l'archive complète (.zip)",
        "export_g_sheets": "📤 Envoyer vers Google Sheets",
        "export_email": "📧 Envoyer le rapport par Email",
    },
    "English": {
        "page_title": "🛍️ Shoplytics Pro 2.0",
        "page_subtitle": "AI-Powered Predictive Customer Behavior Analytics Platform.",
        "data_import_header": "📂 Data Loading",
        "uploader_label": "Upload one or more CSV files",
        "use_sample_label": "Use sample dataset",
        "run_analysis_button": "🚀 Run Analysis",
        "no_data_warning": "Please upload a file or use the sample dataset to start.",
        "analysis_intro": "### Welcome to Shoplytics Pro 2.0\n\nLoad your data and click **'Run Analysis'** to start the predictive magic.",
        "model_config_header": "⚙️ Model Configuration",
        "model_choice_label": "Choose prediction model",
        "tab_overview": "📊 Data Overview",
        "tab_eda": "📈 Exploratory Data Analysis (EDA)",
        "tab_modeling": "🧠 Predictive Modeling",
        "tab_simulation": "🧪 Customer Simulation",
        "tab_export": "📤 Export & Reports",
        "overview_title": "Loaded Data Overview",
        "overview_shape": "Dataset Dimensions:",
        "overview_stats": "Descriptive Statistics:",
        "eda_title": "Interactive Exploratory Analysis",
        "eda_dist_title": "Distribution of Target Variable",
        "eda_corr_title": "Correlation Matrix of Numerical Variables",
        "modeling_title": "Predictive Model Results",
        "modeling_metrics_title": "Model Performance Metrics",
        "modeling_rmse": "RMSE (Root Mean Squared Error)",
        "modeling_r2": "R² Score (Coefficient of Determination)",
        "modeling_preds_vs_actual": "Predictions vs. Actual Values",
        "modeling_shap_title": "Model Interpretability (SHAP)",
        "modeling_shap_global": "Global Feature Importance",
        "modeling_shap_local": "Individual Prediction Analysis",
        "modeling_shap_select_instance": "Choose an observation from the test set to analyze:",
        "simulation_title": "🧪 Simulate Purchase Amount for a New Customer",
        "simulation_expander": "🔧 Fill in the profile for a prediction",
        "simulation_result": "💰 The estimated purchase amount is:",
        "export_title": "Export Results",
        "export_excel_button": "📥 Download Predictions (Excel)",
        "export_pdf_button": "📄 Generate PDF Report",
        "export_zip_button": "📦 Generate Complete Archive (.zip)",
        "export_g_sheets": "📤 Upload to Google Sheets",
        "export_email": "📧 Send Report by Email",
    }
}

# --- 4. Fonctions Utilitaires et de Modélisation (avec cache) ---

class PDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 15)
        self.cell(80)
        self.cell(30, 10, 'Shoplytics Pro - Rapport d\'Analyse', 0, 0, 'C')
        self.ln(20)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

    def chapter_title(self, title):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, title, 0, 1, 'L')
        self.ln(5)

    def chapter_body(self, content):
        self.set_font('Arial', '', 12)
        self.multi_cell(0, 10, content)
        self.ln()
        
    def add_plot(self, plot_path, width=190):
        self.image(plot_path, x=None, y=None, w=width)
        self.ln(5)

@st.cache_data
def load_data(uploaded_files, use_sample):
    if uploaded_files:
        try:
            dfs = [pd.read_csv(file) for file in uploaded_files]
            df = pd.concat(dfs, ignore_index=True)
            return df, "Fichiers importés"
        except Exception as e:
            st.error(f"Erreur lors de la lecture des fichiers CSV : {e}")
            return None, ""
    elif use_sample:
        try:
            df = pd.read_csv("shopping_trends_final.csv")
            return df, "Jeu de données d'exemple"
        except FileNotFoundError:
            st.error("Fichier d'exemple 'shopping_trends_final.csv' non trouvé. Assurez-vous qu'il est dans le bon dossier.")
            return None, ""
    return None, ""

def sanitize_columns(df):
    cols = df.columns
    new_cols = []
    for col in cols:
        new_col = col.strip().lower().replace(' ', '_').replace('(', '').replace(')', '')
        new_cols.append(new_col)
    df.columns = new_cols
    return df

@st.cache_data
def preprocess_data(df, target_col):
    df_processed = df.dropna().copy()
    
    y = df_processed[target_col]
    X = df_processed.drop(columns=[target_col, 'customer_id', 'unnamed:_0'], errors='ignore')

    categorical_features = X.select_dtypes(include=['object', 'category']).columns
    
    X_encoded = pd.get_dummies(X, columns=categorical_features, drop_first=True)
    
    # On force la conversion de toutes les colonnes (y compris les booléens créés par get_dummies) en entiers (0 ou 1)
    # Cela garantit que le DataFrame est 100% numérique pour les modèles et SHAP.
    X_encoded = X_encoded.astype(int)
    
    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)
    
    train_cols = X_train.columns
    test_cols = X_test.columns
    
    missing_in_test = set(train_cols) - set(test_cols)
    for c in missing_in_test:
        X_test[c] = 0
        
    extra_in_test = set(test_cols) - set(train_cols)
    X_test = X_test.drop(columns=list(extra_in_test))
    
    X_test = X_test[train_cols]
    
    return X_train, X_test, y_train, y_test, X, y


@st.cache_resource
def train_model(_X_train, _y_train, model_name, params):
    if model_name == "RandomForest":
        model = RandomForestRegressor(n_estimators=params['n_estimators'], max_depth=params['max_depth'], random_state=42, n_jobs=-1)
    elif model_name == "XGBoost":
        model = xgb.XGBRegressor(n_estimators=params['n_estimators'], max_depth=params['max_depth'], learning_rate=0.1, random_state=42, n_jobs=-1)
    elif model_name == "LightGBM":
        model = lgb.LGBMRegressor(n_estimators=params['n_estimators'], max_depth=params['max_depth'], learning_rate=0.1, random_state=42, n_jobs=-1)
    
    model.fit(_X_train, _y_train)
    return model

@st.cache_data
def get_predictions_and_metrics(_model, _X_test, _y_test):
    y_pred = _model.predict(_X_test)
    rmse = np.sqrt(mean_squared_error(_y_test, y_pred))
    r2 = r2_score(_y_test, y_pred)
    return y_pred, {"rmse": rmse, "r2": r2}

@st.cache_resource
def get_shap_values(_model, _X_train):
    explainer = shap.Explainer(_model, _X_train)
    shap_values = explainer(_X_train)
    return explainer, shap_values

# --- 5. Initialisation de l'État de Session ---
if 'analysis_run' not in st.session_state:
    st.session_state.analysis_run = False
if 'lang' not in st.session_state:
    st.session_state.lang = "Français"
if 'df' not in st.session_state:
    st.session_state.df = None


# --- 6. Barre Latérale (Sidebar) ---
with st.sidebar:
    st.title("Shoplytics Pro 2.0")

    selected_lang = st.selectbox("🌐 Langue / Language", list(LANGUAGES.keys()), index=0 if st.session_state.lang == "Français" else 1)
    st.session_state.lang = selected_lang
    L = LANGUAGES[st.session_state.lang]
    
    st.header(L["data_import_header"])
    use_sample = st.checkbox(L["use_sample_label"], value=True)
    uploaded_files = st.file_uploader(L["uploader_label"], type=["csv"], accept_multiple_files=True, disabled=use_sample)
    
    if st.button(L["run_analysis_button"], type="primary", use_container_width=True):
        df, source = load_data(uploaded_files, use_sample)
        if df is not None:
            # On nettoie les noms de colonnes du DataFrame principal UNE SEULE FOIS
            df = sanitize_columns(df) 
            st.session_state.df = df
            st.session_state.source = source
            st.session_state.analysis_run = True
            st.rerun() # Force a rerun to update the main page
        else:
            st.warning(L["no_data_warning"])
            st.session_state.analysis_run = False

    if st.session_state.analysis_run:
        st.header(L["model_config_header"])
        model_choice = st.selectbox(L["model_choice_label"], ["RandomForest", "XGBoost", "LightGBM"])
        
        n_estimators = st.slider("Nombre d'estimateurs", 50, 500, 150, 10)
        max_depth = st.slider("Profondeur maximale", 3, 20, 10, 1)

        st.session_state.model_params = {'n_estimators': n_estimators, 'max_depth': max_depth}
        st.session_state.model_choice = model_choice


# --- 7. Page Principale ---
L = LANGUAGES[st.session_state.lang]
st.title(L["page_title"])
st.markdown(L["page_subtitle"])

if not st.session_state.analysis_run or st.session_state.df is None:
    st.info(L["analysis_intro"])
    st.stop()

# --- Préparation des données et Modélisation ---
with st.spinner("Préparation des données, entraînement du modèle et calcul des analyses... Veuillez patienter."):
    df_clean = st.session_state.df.copy()
    target_col_name = "purchase_amount_usd"
    X_train, X_test, y_train, y_test, X_original, y_original = preprocess_data(df_clean, target_col_name)

    model = train_model(X_train, y_train, st.session_state.model_choice, st.session_state.model_params)
    y_pred, metrics = get_predictions_and_metrics(model, X_test, y_test)
    explainer, shap_values = get_shap_values(model, X_train)
    
    st.session_state.metrics = metrics
    st.session_state.model = model
    st.session_state.X_train = X_train
    st.session_state.X_test = X_test
    st.session_state.y_test = y_test
    st.session_state.y_pred = y_pred
    st.session_state.explainer = explainer
    st.session_state.shap_values = shap_values
    st.session_state.X_original = X_original
    
# --- Affichage des onglets ---
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    L["tab_overview"], 
    L["tab_eda"], 
    L["tab_modeling"], 
    L["tab_simulation"], 
    L["tab_export"]
])

with tab1:
    st.header(L["overview_title"])
    st.markdown(f"**Source :** {st.session_state.source}")
    st.markdown(f"**{L['overview_shape']}** `{st.session_state.df.shape[0]}` lignes, `{st.session_state.df.shape[1]}` colonnes.")
    st.dataframe(st.session_state.df.head(), use_container_width=True)
    
    with st.expander(L["overview_stats"]):
        st.dataframe(st.session_state.df.describe(include='all').T, use_container_width=True)

with tab2:
    st.header(L["eda_title"])
    
    st.subheader(L["eda_dist_title"])
    fig_dist = px.histogram(st.session_state.df, x=target_col_name, nbins=50, title="Distribution du Montant d'Achat", marginal="box")
    st.plotly_chart(fig_dist, use_container_width=True)

    st.subheader(L["eda_corr_title"])
    numeric_df = st.session_state.df.select_dtypes(include=np.number)
    if len(numeric_df.columns) > 1:
        corr_matrix = numeric_df.corr()
        fig_corr = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu_r',
            zmin=-1, zmax=1
        ))
        fig_corr.update_layout(title="Matrice de Corrélation")
        st.plotly_chart(fig_corr, use_container_width=True)
    
    st.subheader("Analyse bivariée")
    col1, col2 = st.columns(2)
    cat_cols = st.session_state.X_original.select_dtypes(include=['object', 'category']).columns
    
    with col1:
        selected_cat = st.selectbox("Choisir une variable catégorielle", cat_cols)
    
    fig_box = px.box(st.session_state.df, x=selected_cat, y=target_col_name, title=f"Montant d'achat par {selected_cat}", color=selected_cat)
    st.plotly_chart(fig_box, use_container_width=True)

with tab3:
    st.header(L["modeling_title"])
    st.subheader(L["modeling_metrics_title"])
    
    kpi1, kpi2 = st.columns(2)
    kpi1.metric(L["modeling_rmse"], f"{st.session_state.metrics['rmse']:.2f} USD")
    kpi2.metric(L["modeling_r2"], f"{st.session_state.metrics['r2']:.2%}")

    st.subheader(L["modeling_preds_vs_actual"])
    pred_df = pd.DataFrame({'Réel': st.session_state.y_test, 'Prédit': st.session_state.y_pred})
    fig_pred = px.scatter(pred_df, x='Réel', y='Prédit', 
                          title="Prédictions vs Valeurs Réelles",
                          labels={'Réel': 'Valeur Réelle (USD)', 'Prédit': 'Valeur Prédite (USD)'},
                          trendline='ols', trendline_color_override='red')
    st.plotly_chart(fig_pred, use_container_width=True)

    st.subheader(L["modeling_shap_title"])
    
    with st.expander(L["modeling_shap_global"]):
        fig_shap_bar, ax_shap_bar = plt.subplots()
        shap.summary_plot(st.session_state.shap_values, st.session_state.X_train, plot_type="bar", show=False)
        st.pyplot(fig_shap_bar)
        
    with st.expander(L["modeling_shap_local"]):
        instance_index = st.slider(L["modeling_shap_select_instance"], 0, len(st.session_state.X_test)-1, 0)
        
        fig_waterfall, ax_waterfall = plt.subplots()
        shap_explainer_instance = st.session_state.explainer(st.session_state.X_test.iloc[[instance_index]])
        shap.plots.waterfall(shap_explainer_instance[0], show=False)
        st.pyplot(fig_waterfall)

with tab4:
    st.header(L["simulation_title"])
    
    with st.expander(L["simulation_expander"], expanded=True):
        sample_input = {}
        X_orig_sim = st.session_state.X_original
        
        cols = st.columns(3)
        col_idx = 0
        
        for col in X_orig_sim.columns:
            with cols[col_idx]:
                if X_orig_sim[col].dtype == "object":
                    options = sorted(X_orig_sim[col].dropna().unique().tolist())
                    sample_input[col] = st.selectbox(f"{col.replace('_', ' ').title()}", options, key=f"sim_{col}")
                elif X_orig_sim[col].dtype in [np.int64, np.int32] and len(X_orig_sim[col].unique()) < 20:
                    options = sorted(X_orig_sim[col].dropna().unique().tolist())
                    sample_input[col] = st.selectbox(f"{col.replace('_', ' ').title()}", options, key=f"sim_{col}")
                elif X_orig_sim[col].dtype in [np.int64, np.int32]:
                    min_val, max_val = int(X_orig_sim[col].min()), int(X_orig_sim[col].max())
                    mean_val = int(X_orig_sim[col].mean())
                    sample_input[col] = st.slider(f"{col.replace('_', ' ').title()}", min_val, max_val, mean_val, key=f"sim_{col}")
                elif X_orig_sim[col].dtype in [np.float64, np.float32]:
                    min_val, max_val = float(X_orig_sim[col].min()), float(X_orig_sim[col].max())
                    mean_val = float(X_orig_sim[col].mean())
                    sample_input[col] = st.slider(f"{col.replace('_', ' ').title()}", min_val, max_val, mean_val, key=f"sim_{col}")
            col_idx = (col_idx + 1) % 3
            
        if st.button("Prédire", type="primary"):
            input_df = pd.DataFrame([sample_input])
            input_encoded = pd.get_dummies(input_df)
            
            model_cols = st.session_state.X_train.columns
            input_reindexed = input_encoded.reindex(columns=model_cols, fill_value=0)
            
            prediction = st.session_state.model.predict(input_reindexed)[0]
            st.success(f"**{L['simulation_result']} {prediction:.2f} USD**")

with tab5:
    st.header(L["export_title"])

    export_df = st.session_state.X_test.copy()
    export_df['actual_purchase_amount'] = st.session_state.y_test
    export_df['predicted_purchase_amount'] = st.session_state.y_pred
    
    excel_buffer = io.BytesIO()
    export_df.to_excel(excel_buffer, index=True, engine='openpyxl')
    excel_bytes = excel_buffer.getvalue()
    
    st.download_button(
        label=L["export_excel_button"],
        data=excel_bytes,
        file_name="shoplytics_pro_predictions.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

    if st.button(L["export_pdf_button"]):
        with st.spinner("Génération du rapport PDF en cours..."):
            with tempfile.TemporaryDirectory() as temp_dir:
                pdf = PDF()
                pdf.add_page()
                
                pdf.chapter_title("1. Performance du Modèle")
                pdf.chapter_body(f"RMSE: {st.session_state.metrics['rmse']:.2f} USD\nScore R²: {st.session_state.metrics['r2']:.2%}")
                
                fig_pred_path = os.path.join(temp_dir, "pred_vs_actual.png")
                pred_df = pd.DataFrame({'Réel': st.session_state.y_test, 'Prédit': st.session_state.y_pred})
                fig_pred = px.scatter(pred_df, x='Réel', y='Prédit', trendline='ols', trendline_color_override='red')
                fig_pred.write_image(fig_pred_path)
                pdf.chapter_title("2. Graphique: Prédictions vs Valeurs Réelles")
                pdf.add_plot(fig_pred_path)
                
                fig_shap_path = os.path.join(temp_dir, "shap_summary.png")
                fig, ax = plt.subplots()
                shap.summary_plot(st.session_state.shap_values, st.session_state.X_train, plot_type="bar", show=False)
                fig.savefig(fig_shap_path, bbox_inches='tight')
                plt.close(fig)
                pdf.chapter_title("3. Importance des Variables (SHAP)")
                pdf.add_plot(fig_shap_path)
                
                pdf_output_path = os.path.join(temp_dir, "Shoplytics_Pro_Report.pdf")
                pdf.output(pdf_output_path)
                
                with open(pdf_output_path, "rb") as f:
                    st.download_button(
                        label="📥 Télécharger le Rapport PDF",
                        data=f.read(),
                        file_name="Shoplytics_Pro_Report.pdf",
                        mime="application/pdf"
                    )

    if st.button(L["export_zip_button"]):
         with st.spinner("Création de l'archive ZIP..."):
            with tempfile.TemporaryDirectory() as temp_dir:
                preds_path = os.path.join(temp_dir, "predictions.csv")
                export_df.to_csv(preds_path, index=False)
                
                csv_path = os.path.join(temp_dir, "original_data.csv")
                st.session_state.df.to_csv(csv_path, index=False)

                zip_path = os.path.join(temp_dir, "Shoplytics_Report_Complete.zip")
                with zipfile.ZipFile(zip_path, 'w') as zipf:
                    zipf.write(preds_path, arcname="predictions.csv")
                    zipf.write(csv_path, arcname="original_data.csv")
                
                with open(zip_path, "rb") as f:
                    st.download_button(
                        label="⬇️ Télécharger l'archive ZIP",
                        data=f.read(),
                        file_name="Shoplytics_Report_Complete.zip"
                    )
                    
    with st.expander(L["export_g_sheets"]):
        st.info("Fonctionnalité avancée : Connectez-vous à Google Sheets pour un suivi en temps réel.")

    with st.expander(L["export_email"]):
        st.warning("⚠️ **Sécurité** : Utilisez un **mot de passe d'application** généré par Google, et non votre mot de passe personnel. Ne partagez jamais vos identifiants.")