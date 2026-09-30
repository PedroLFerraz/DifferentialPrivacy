from flask import Flask, request, render_template, send_file, session, url_for, flash, redirect
import pandas as pd
import os
import numpy as np
import opendp.prelude as dp
import math
import json
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.spatial.distance import cdist
from pyproj import CRS, Transformer

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64

dp.enable_features("contrib")

app = Flask(__name__)
app.secret_key = 'your-secret-key'
UPLOAD_FOLDER = "data/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
processed_dfs = {}

app.jinja_env.filters['log'] = lambda x, base=math.e: math.log(x, base) if x > 0 else -3

# Texts for bilingual support
TEXTS = {
    'pt': {
        'h1': "Enviar Dataset para Anonimização",
        'upload': "Enviar",
        'preview_orig': "Pré-visualização do Dataset Original",
        'stats_orig': "Resumo Estatístico",
        'preview_anon': "Pré-visualização do Dataset Anonimizado",
        'stats_anon': "Resumo Estatístico",
        'select_column': "Escolha a coluna a ser anonimizada:",
        'select_method': "Escolha o método de anonimização:",
        'random_aggregation': "Agregação Aleatória",
        'distance_aggregation': "Agregação por Distância",
        'differential_privacy': "Privacidade Diferencial",
        'differential_privacy_random': "Privacidade Diferencial com Agregação Aleatória",
        'distance_slider': "Distância máxima para agregação:",
        'distance_val': "Valor atual",
        'distance_min_max': "(Mín: {min} m, Máx: {max} m)",
        'group_size': "Tamanho do grupo para agregação aleatória:",
        'epsilon_val': "Valor de Epsilon:",
        'use_clamping': "Usar Clamping:",
        'clamp_boxplot': "Limites do Boxplot",
        'clamp_manual': "Limites personalizados",
        'clamp_min': "Min:",
        'clamp_max': "Max:",
        'anonimize': "Anonimizar",
        'download': "Baixar arquivo anoninizado",
        'epsilon_sum': "Epsilon acumulado nesta sessão:",
        'epsilon_sum_tip': "Quanto maior o epsilon acumulado, menor a privacidade. Cada requisição Differential Privacy (DP) soma seu epsilon a esse valor.",
        'plot_original': "Original",
        'plot_anon': "Anonimizado",
        'plot_both': "Original vs Anonimizado",
        'plot_noise': "Ruído",
        'rmse_label': "RMSE entre dataset original e anonimizado:",
        'mae_label': "MAE:",
        'cv_label': "Coeficiente de variação (CV):",
        'method_explanation_title': "Explicação do método",
        'warning_latlon': "Agregação por distância requer colunas de latitude e longitude no dataset.",
        'lang_switch': "English Version",
        'lang_param': "en"
    },
    'en': {
        'h1': "Upload Dataset for Anonymization",
        'upload': "Upload",
        'preview_orig': "Original Dataset Preview",
        'stats_orig': "Statistical Summary",
        'preview_anon': "Anonymized Dataset Preview",
        'stats_anon': "Statistical Summary",
        'select_column': "Select the column to anonymize:",
        'select_method': "Select anonymization method:",
        'random_aggregation': "Random Aggregation",
        'distance_aggregation': "Distance Aggregation",
        'differential_privacy': "Differential Privacy",
        'differential_privacy_random': "Differential Privacy + Random Aggregation",
        'distance_slider': "Maximum distance for aggregation:",
        'distance_val': "Current value",
        'distance_min_max': "(Min: {min} m, Max: {max} m)",
        'group_size': "Group size for random aggregation:",
        'epsilon_val': "Epsilon value:",
        'use_clamping': "Use Clamping:",
        'clamp_boxplot': "Boxplot limits",
        'clamp_manual': "Custom limits",
        'clamp_min': "Min:",
        'clamp_max': "Max:",
        'anonimize': "Anonymize",
        'download': "Download anonymized file",
        'epsilon_sum': "Accumulated epsilon this session:",
        'epsilon_sum_tip': "The higher the accumulated epsilon, the lower the privacy. Each Differential Privacy (DP) request adds its epsilon to this value.",
        'plot_original': "Original",
        'plot_anon': "Anonymized",
        'plot_both': "Original vs Anonymized",
        'plot_noise': "Noise",
        'rmse_label': "RMSE between original and anonymized dataset:",
        'mae_label': "MAE:",
        'cv_label': "Coefficient of variation (CV):",
        'method_explanation_title': "Method explanation",
        'warning_latlon': "Distance aggregation requires latitude and longitude columns in your dataset.",
        'lang_switch': "Versão em Português",
        'lang_param': "pt"
    },
}

METHOD_EXPLANATIONS = {
    "random_aggregation": {
        "pt": "A Agregação Aleatória agrupa pontos de dados aleatoriamente e substitui cada grupo por sua média. Isso aumenta a privacidade ao diluir contribuições individuais.",
        "en": "Random Aggregation groups data points randomly and replaces each group by its mean. This increases privacy by diluting individual contributions."
    },
    "distance_aggregation": {
        "pt": "A Agregação por Distância agrupa pontos de dados com base em sua proximidade física (usando latitude e longitude em graus). Para cada ponto, todos os pontos dentro da distância selecionada (em metros) formam um grupo. Grupos com exatamente o mesmo conjunto de pontos são descartados como duplicatas.",
        "en": "Distance Aggregation groups data points based on their physical proximity (using latitude and longitude in degrees). For each point, all points within the selected distance (in meters) form a group. Groups with exactly the same set of points are dropped as duplicates."
    },
    "differential_privacy": {
        "pt": "Privacidade Diferencial adiciona ruído estatístico (usando o mecanismo Laplaciano) aos seus dados, fornecendo fortes garantias de privacidade enquanto preserva tendências gerais dos dados.",
        "en": "Differential Privacy adds statistical noise (using the Laplacian mechanism) to your data, providing strong privacy guarantees while preserving overall data trends."
    },
    "differential_privacy_random": {
        "pt": "Privacidade Diferencial com Agregação Aleatória primeiro agrupa dados aleatoriamente, calcula a média de cada grupo e então adiciona ruído Laplaciano para maior privacidade.",
        "en": "Differential Privacy with Random Aggregation first groups data randomly, computes each group mean, and then adds Laplacian noise to each mean for enhanced privacy."
    }
}

def get_lang():
    lang = request.args.get('lang') or session.get('lang') or 'pt'
    if lang not in ('pt', 'en'):
        lang = 'pt'
    session['lang'] = lang
    return lang

def get_texts():
    lang = get_lang()
    return TEXTS[lang]

def get_method_explanations():
    lang = get_lang()
    return {k: v[lang] for k, v in METHOD_EXPLANATIONS.items()}

def get_switch_url():
    lang = get_lang()
    switch_lang = 'en' if lang == 'pt' else 'pt'
    params = dict(request.view_args) if request.view_args else {}
    params.update(request.args.to_dict())
    params['lang'] = switch_lang
    return url_for(request.endpoint, **params)

def gen_laplacian_noise(epsilon, sensitivity, size=1):
    b = sensitivity / epsilon
    noise = np.random.laplace(loc=0, scale=b, size=size)
    return noise

def latlon_to_utm(df, lat_col, lon_col):
    lat0, lon0 = df[lat_col].mean(), df[lon_col].mean()
    zone = int((lon0 + 180) / 6) + 1
    south = " +south" if lat0 < 0 else ""
    utm_crs = CRS.from_proj4(f"+proj=utm +zone={zone} +datum=WGS84 +units=m{south}")
    transformer = Transformer.from_crs("EPSG:4326", utm_crs, always_xy=True)
    xs, ys = transformer.transform(df[lon_col].values, df[lat_col].values)
    return pd.DataFrame({'x': xs, 'y': ys}, index=df.index)

def plot_distribution_variants(original_series, anonymized_series, column_name, noise=None, plot_title=''):
    plt.figure(figsize=(8, 4))
    sns.histplot(original_series.dropna(), color='blue', kde=True, stat='density', label='Original', linewidth=0, alpha=0.5)
    plt.xlabel(column_name)
    plt.ylabel('Density')
    plt.title('Original' + (f" {plot_title}" if plot_title else ""))
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    plot_orig = base64.b64encode(buf.read()).decode('utf-8')

    plt.figure(figsize=(8, 4))
    sns.histplot(anonymized_series.dropna(), color='orange', kde=True, stat='density', label='Anonymized', linewidth=0, alpha=0.5)
    plt.xlabel(column_name)
    plt.ylabel('Density')
    plt.title('Anonymized' + (f" {plot_title}" if plot_title else ""))
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    plot_anon = base64.b64encode(buf.read()).decode('utf-8')

    plt.figure(figsize=(8, 4))
    sns.histplot(original_series.dropna(), color='blue', kde=True, stat='density', label='Original', linewidth=0, alpha=0.35)
    sns.histplot(anonymized_series.dropna(), color='orange', kde=True, stat='density', label='Anonymized', linewidth=0, alpha=0.35)
    plt.legend()
    plt.xlabel(column_name)
    plt.ylabel('Density')
    plt.title('Original vs Anonymized' + (f" {plot_title}" if plot_title else ""))
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    plot_both = base64.b64encode(buf.read()).decode('utf-8')

    plot_noise = None
    if noise is not None:
        plt.figure(figsize=(8, 4))
        sns.histplot(noise, color='red', stat='density', label='Noise', linewidth=0, alpha=0.6)
        plt.xlabel('Noise')
        plt.ylabel('Density')
        plt.title('Noise Distribution')
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close()
        buf.seek(0)
        plot_noise = base64.b64encode(buf.read()).decode('utf-8')

    return plot_orig, plot_anon, plot_both, plot_noise

def clamp_series(series, min_val=None, max_val=None):
    arr = np.array(series)
    if min_val is not None:
        arr = np.maximum(arr, min_val)
    if max_val is not None:
        arr = np.minimum(arr, max_val)
    return pd.Series(arr, index=series.index)

def distance_aggregation(df, column_name, lat_col, lon_col, threshold):
    utm_coords = latlon_to_utm(df, lat_col, lon_col)
    coords = utm_coords[['x', 'y']].values
    values = df[column_name].values
    n = len(df)
    groups = {}
    for i in range(n):
        dists = np.linalg.norm(coords - coords[i], axis=1)
        group_idx = tuple(sorted(np.where(dists <= threshold)[0]))
        if group_idx not in groups:
            groups[group_idx] = np.mean(values[list(group_idx)])
    return pd.DataFrame({column_name: list(groups.values())})

@app.route('/')
def home():
    session['epsilon_sum'] = 0.0
    lang = get_lang()
    switch_url = get_switch_url()
    return render_template('index.html',
        lang=lang, switch_url=switch_url,
        texts=get_texts(),
        method_explanations_json=json.dumps(get_method_explanations()),
        columns=None,
        file_name=None,
        lat_lon_columns=None,
        dataset_length=None,
        input_table=None,
        input_describe=None,
        output_table=None,
        output_describe=None,
        selected_column=None,
        selected_method="random_aggregation",
        group_size_random=2,
        group_size_dprand=2,
        epsilon_dp=0.001,
        epsilon_dprand=0.001,
        clamping_dp='',
        clamping_dprand='',
        clamping_mode="boxplot",
        clamping_min=None,
        clamping_max=None,
        distance_threshold=1,
        distance_min=None,
        distance_max=None,
        scroll_to_output=False,
        plot_orig=None,
        plot_anon=None,
        plot_both=None,
        plot_noise=None,
        epsilon_sum=0.0,
        rmse=None,
        mae=None,
        cv=None,
        download_link=None,
    )

@app.route('/upload', methods=['POST'])
def upload_file():
    session['epsilon_sum'] = 0.0
    lang = get_lang()
    switch_url = get_switch_url()
    if 'file' not in request.files:
        return "Nenhum arquivo enviado."
    file = request.files['file']
    if file.filename == '':
        return "Nenhum arquivo selecionado."
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)
    df = pd.read_csv(file_path) if file_path.endswith('.csv') else pd.read_excel(file_path)
    lat_lon_columns = [col for col in df.columns if 'lat' in col.lower() or 'lon' in col.lower()]
    numeric_columns = [col for col in df.select_dtypes(include=['int64', 'float64']).columns if col not in lat_lon_columns]
    dataset_length = len(df)
    distance_min, distance_max = None, None
    if len(lat_lon_columns) >= 2:
        utm_coords = latlon_to_utm(df, lat_lon_columns[0], lat_lon_columns[1])
        coords = utm_coords[['x', 'y']].values
        dist_matrix = cdist(coords, coords)
        np.fill_diagonal(dist_matrix, np.nan)
        distance_min = float(np.nanmin(dist_matrix))
        distance_max = float(np.nanmax(dist_matrix))
    preview_df = df.drop(columns=lat_lon_columns, errors='ignore')
    input_table = preview_df.head(8).to_html(classes="data", index=False, float_format="%.2f")
    input_describe = preview_df.describe().to_html(classes="data", float_format="%.2f")
    session['file_name'] = file.filename
    return render_template(
        'index.html',
        lang=lang, switch_url=switch_url, texts=get_texts(),
        method_explanations_json=json.dumps(get_method_explanations()),
        columns=numeric_columns,
        file_name=file.filename,
        lat_lon_columns=lat_lon_columns,
        dataset_length=dataset_length,
        input_table=input_table,
        input_describe=input_describe,
        output_table=None,
        output_describe=None,
        selected_column=None,
        selected_method="random_aggregation",
        group_size_random=2,
        group_size_dprand=2,
        epsilon_dp=0.001,
        epsilon_dprand=0.001,
        clamping_dp='',
        clamping_dprand='',
        clamping_mode="boxplot",
        clamping_min=None,
        clamping_max=None,
        distance_threshold=distance_min if distance_min is not None else 1,
        distance_min=distance_min,
        distance_max=distance_max,
        scroll_to_output=False,
        plot_orig=None,
        plot_anon=None,
        plot_both=None,
        plot_noise=None,
        epsilon_sum=0.0,
        rmse=None,
        mae=None,
        cv=None,
        download_link=url_for('download'),
    )

@app.route('/setlang/<lang>')
def setlang(lang):
    if lang not in ('pt', 'en'):
        lang = 'pt'
    session['lang'] = lang
    return redirect(url_for('home', lang=lang))

@app.route('/process', methods=['POST'])
def process_file():
    lang = get_lang()
    column_name = request.form['column']
    method = request.form['method']
    file_name = request.form['file_name']
    file_path = os.path.join(UPLOAD_FOLDER, file_name)
    group_size_random = int(request.form.get('group_size_random', 2))
    group_size_dprand = int(request.form.get('group_size_dprand', 2))
    epsilon_dp = float(request.form.get('epsilon_dp', 0.001))
    epsilon_dprand = float(request.form.get('epsilon_dprand', 0.001))
    distance_threshold = float(request.form.get('distance_threshold', 1))
    clamping_dp = 'checked' if request.form.get('clamping_dp') == 'on' else ''
    clamping_dprand = 'checked' if request.form.get('clamping_dprand') == 'on' else ''
    clamping_mode = request.form.get('clamping_mode_dp', request.form.get('clamping_mode_dprand', 'boxplot'))
    clamping_min = request.form.get('clamping_min', None)
    clamping_max = request.form.get('clamping_max', None)
    clamping_min = float(clamping_min) if clamping_min not in (None, "", "None") else None
    clamping_max = float(clamping_max) if clamping_max not in (None, "", "None") else None

    input_df = pd.read_csv(file_path) if file_path.endswith('.csv') else pd.read_excel(file_path)
    lat_lon_columns = [col for col in input_df.columns if 'lat' in col.lower() or 'lon' in col.lower()]
    numeric_columns = [col for col in input_df.select_dtypes(include=['int64', 'float64']).columns if col not in lat_lon_columns]

    distance_min, distance_max = None, None
    if len(lat_lon_columns) >= 2:
        utm_coords = latlon_to_utm(input_df, lat_lon_columns[0], lat_lon_columns[1])
        coords = utm_coords[['x', 'y']].values
        dist_matrix = cdist(coords, coords)
        np.fill_diagonal(dist_matrix, np.nan)
        distance_min = float(np.nanmin(dist_matrix))
        distance_max = float(np.nanmax(dist_matrix))

    if method == "distance_aggregation":
        if len(lat_lon_columns) < 2:
            flash(TEXTS[lang]['warning_latlon'], "error")
            preview_df = input_df.drop(columns=lat_lon_columns, errors='ignore')
            input_table = preview_df.head(8).to_html(classes="data", index=False, float_format="%.2f")
            input_describe = preview_df.describe().to_html(classes="data", float_format="%.2f")
            dataset_length = len(input_df)
            return render_template(
                'index.html',
                lang=lang,
                texts=get_texts(),
                method_explanations_json=json.dumps(get_method_explanations()),
                columns=numeric_columns,
                file_name=file_name,
                lat_lon_columns=lat_lon_columns,
                dataset_length=dataset_length,
                input_table=input_table,
                input_describe=input_describe,
                output_table=None,
                output_describe=None,
                selected_column=column_name,
                selected_method=method,
                group_size_random=group_size_random,
                group_size_dprand=group_size_dprand,
                epsilon_dp=epsilon_dp,
                epsilon_dprand=epsilon_dprand,
                clamping_dp=clamping_dp,
                clamping_dprand=clamping_dprand,
                clamping_mode=clamping_mode,
                clamping_min=clamping_min,
                clamping_max=clamping_max,
                distance_threshold=distance_threshold,
                distance_min=distance_min,
                distance_max=distance_max,
                scroll_to_output=False,
                plot_orig=None,
                plot_anon=None,
                plot_both=None,
                plot_noise=None,
                epsilon_sum=session.get('epsilon_sum', 0.0),
                rmse=None,
                mae=None,
                cv=None,
                download_link=url_for('download'),
            )

    if method == "random_aggregation":
        group_size = group_size_random
        epsilon = None
        use_clamping = False
    elif method == "differential_privacy":
        group_size = None
        epsilon = epsilon_dp
        use_clamping = request.form.get('clamping_dp') == 'on'
    elif method == "differential_privacy_random":
        group_size = group_size_dprand
        epsilon = epsilon_dprand
        use_clamping = request.form.get('clamping_dprand') == 'on'
    elif method == "distance_aggregation":
        group_size = None
        epsilon = None
        use_clamping = False
    else:
        group_size = None
        epsilon = None
        use_clamping = False

    preview_df = input_df.drop(columns=lat_lon_columns, errors='ignore')
    input_table = preview_df.head(8).to_html(classes="data", index=False, float_format="%.2f")
    input_describe = preview_df.describe().to_html(classes="data", float_format="%.2f")
    processed_df, noise_array = process_dataset(
        file_path, column_name, method, group_size, distance_threshold, epsilon, use_clamping, return_df=True,
        clamping_mode=clamping_mode, clamping_min=clamping_min, clamping_max=clamping_max, input_df=input_df, return_noise=True,
        lat_lon_columns=lat_lon_columns
    )
    output_preview_df = processed_df.drop(columns=[col for col in processed_df.columns if col in lat_lon_columns], errors='ignore')
    output_table = output_preview_df.head(8).to_html(classes="data", index=False, float_format="%.2f")
    output_describe = output_preview_df.describe().to_html(classes="data", float_format="%.2f")

    # Cálculo do RMSE/MAE/CV (para DP e DP+random)
    rmse = None
    mae = None
    cv = None
    if method == "differential_privacy":
        anon_series = processed_df[column_name]
        orig_series = input_df[column_name]
        minlen = min(len(anon_series), len(orig_series))
        rmse = math.sqrt(mean_squared_error(orig_series.iloc[:minlen], anon_series.iloc[:minlen]))
        mae = mean_absolute_error(orig_series.iloc[:minlen], anon_series.iloc[:minlen])
        mean_orig = orig_series.iloc[:minlen].mean()
        cv = rmse / abs(mean_orig) if mean_orig != 0 else None
    elif method == "differential_privacy_random":
        anon_series = processed_df['Noisy Group Mean']
        orig_groups = []
        num_groups = len(anon_series)
        group_size = group_size_dprand
        df = input_df
        df = df.sample(n=num_groups * group_size, random_state=42).reset_index(drop=True)
        for i in range(0, len(df), group_size):
            orig_groups.append(df[column_name].iloc[i:i + group_size].mean())
        minlen = min(len(orig_groups), len(anon_series))
        rmse = math.sqrt(mean_squared_error(orig_groups[:minlen], anon_series.iloc[:minlen]))
        mae = mean_absolute_error(orig_groups[:minlen], anon_series.iloc[:minlen])
        mean_orig = np.mean(orig_groups[:minlen]) if minlen > 0 else None
        cv = rmse / mean_orig if mean_orig and mean_orig != 0 else None

    if rmse is not None:
        rmse = round(rmse, 2)
    if mae is not None:
        mae = round(mae, 2)
    if cv is not None:
        cv = round(cv, 2)

    if method == "differential_privacy_random":
        orig_series = input_df[column_name]
        anon_series = processed_df['Noisy Group Mean']
    elif method == "random_aggregation":
        orig_series = input_df[column_name]
        anon_series = processed_df[column_name]
    elif method == "differential_privacy":
        orig_series = input_df[column_name]
        anon_series = processed_df[column_name]
    else:
        orig_series = input_df[column_name]
        anon_series = processed_df[column_name]

    plot_orig, plot_anon, plot_both, plot_noise = plot_distribution_variants(orig_series, anon_series, column_name, noise=noise_array)

    dataset_length = len(input_df)
    session['processed_key'] = 'last'
    processed_dfs['last'] = processed_df
    return render_template(
        'index.html',
        lang=lang,
        texts=get_texts(),
        method_explanations_json=json.dumps(get_method_explanations()),
        columns=numeric_columns,
        file_name=file_name,
        lat_lon_columns=lat_lon_columns,
        dataset_length=dataset_length,
        input_table=input_table,
        input_describe=input_describe,
        output_table=output_table,
        output_describe=output_describe,
        download_link=url_for('download'),
        selected_column=column_name,
        selected_method=method,
        group_size_random=group_size_random,
        group_size_dprand=group_size_dprand,
        epsilon_dp=epsilon_dp,
        epsilon_dprand=epsilon_dprand,
        clamping_dp=clamping_dp,
        clamping_dprand=clamping_dprand,
        clamping_mode=clamping_mode,
        clamping_min=clamping_min,
        clamping_max=clamping_max,
        distance_threshold=distance_threshold,
        distance_min=distance_min,
        distance_max=distance_max,
        scroll_to_output=True,
        plot_orig=plot_orig,
        plot_anon=plot_anon,
        plot_both=plot_both,
        plot_noise=plot_noise,
        epsilon_sum=session.get('epsilon_sum', 0.0),
        rmse=rmse,
        mae=mae,
        cv=cv,
    )

@app.route('/download')
def download():
    key = session.get('processed_key', 'last')
    df = processed_dfs.get(key)
    if df is None:
        return "No processed dataset to download. Please anonymize a dataset first.", 404
    buf = io.BytesIO()
    df.to_csv(buf, index=False, float_format="%.2f")
    buf.seek(0)
    return send_file(buf, mimetype='text/csv', as_attachment=True, download_name='anonimized_output.csv')

def process_dataset(file_path, column_name, method, group_size, distance_threshold, epsilon, use_clamping, return_df=False,
                   clamping_mode="boxplot", clamping_min=None, clamping_max=None, input_df=None, return_noise=False,
                   lat_lon_columns=None):
    df = pd.read_csv(file_path) if file_path.endswith('.csv') else pd.read_excel(file_path)
    noise_array = None
    if column_name in df.columns:
        if df[column_name].isnull().any():
            df = df.dropna(subset=[column_name])
        if method == "differential_privacy":
            sensitivity = df[column_name].max() - df[column_name].min()
            laplacian_noise = gen_laplacian_noise(epsilon, sensitivity, size=len(df))
            noisy_results = df[column_name].values + laplacian_noise
            noise_array = laplacian_noise
            df_result = pd.DataFrame({column_name: noisy_results})
        elif method == "random_aggregation":
            num_groups = len(df) // group_size
            df = df.sample(n=num_groups * group_size, random_state=42)
            groups = [df[column_name].iloc[i:i + group_size].mean() for i in range(0, len(df), group_size)]
            df_result = pd.DataFrame({column_name: groups})
        elif method == "differential_privacy_random":
            num_groups = len(df) // group_size
            df = df.sample(n=num_groups * group_size, random_state=42).reset_index(drop=True)
            group_means = [
                df[column_name].iloc[i:i + group_size].mean()
                for i in range(0, len(df), group_size)
            ]
            sensitivity = df[column_name].max() - df[column_name].min()
            lap_noise = gen_laplacian_noise(epsilon, sensitivity / group_size, size=num_groups)
            noisy_means = [mean + noise for mean, noise in zip(group_means, lap_noise)]
            noise_array = np.array(noisy_means) - np.array(group_means)
            df_result = pd.DataFrame({
                'Group Mean': group_means,
                'Noisy Group Mean': noisy_means
            })
        elif method == "distance_aggregation" and lat_lon_columns and len(lat_lon_columns) >= 2:
            lat_col, lon_col = lat_lon_columns[0], lat_lon_columns[1]
            df_result = distance_aggregation(df, column_name, lat_col, lon_col, distance_threshold)
        else:
            df_result = df[[column_name]].copy()

        if use_clamping and column_name in df_result.columns:
            if clamping_mode == "boxplot" and input_df is not None:
                q1 = input_df[column_name].quantile(0.25)
                q3 = input_df[column_name].quantile(0.75)
                iqr = q3 - q1
                clamp_min = q1 - 1.5 * iqr
                clamp_max = q3 + 1.5 * iqr
            elif clamping_mode == "manual":
                clamp_min = clamping_min
                clamp_max = clamping_max
            else:
                clamp_min = None
                clamp_max = None
            df_result[column_name] = clamp_series(df_result[column_name], min_val=clamp_min, max_val=clamp_max)

        if return_df:
            if return_noise:
                return df_result, noise_array
            else:
                return df_result
    if return_noise:
        return pd.DataFrame(), None
    return pd.DataFrame() if return_df else "Erro: coluna não encontrada."

if __name__ == '__main__':
    app.run(debug=True)