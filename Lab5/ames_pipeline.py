"""
Ames Housing Dataset — Pipeline completo
Limpeza de dados + Feature Engineering + Regressão Linear

Uso:
    df_train_student = pd.read_csv("train.csv")
    df_test_student  = pd.read_csv("test.csv")
    X_train, X_test, y_train = build_pipeline(df_train_student, df_test_student)
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.linear_model import Ridge, Lasso, LinearRegression
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error
from statsmodels.stats.outliers_influence import variance_inflation_factor
import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv("train_student.csv")
# ─────────────────────────────────────────────
# FASE 1 — VALORES AUSENTES
# ─────────────────────────────────────────────

def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Trata valores ausentes do dataset Ames Housing.
    Distingue NaN semântico (sem feature) de dado realmente faltante.
    """
    df = df.copy()

    # --- 1a. NaN semântico: ausência = "não tem" → preenche com "None" ---
    cat_none = [
        "PoolQC", "MiscFeature", "Alley", "Fence",
        "FireplaceQu",
        "GarageType", "GarageFinish", "GarageQual", "GarageCond",
        "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2",
        "MasVnrType",
    ]
    for col in cat_none:
        if col in df.columns:
            df[col] = df[col].fillna("None")

    # --- 1b. NaN numérico: ausência = 0 (sem garagem, sem porão, etc.) ---
    num_zero = [
        "GarageArea", "GarageCars",
        "BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF", "TotalBsmtSF",
        "BsmtFullBath", "BsmtHalfBath",
        "MasVnrArea",
    ]
    for col in num_zero:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    # --- 1c. LotFrontage: mediana por Neighborhood (mais preciso) ---
    if "LotFrontage" in df.columns:
        df["LotFrontage"] = df.groupby("Neighborhood")["LotFrontage"].transform(
            lambda x: x.fillna(x.median())
        )
        # fallback global caso algum bairro só tenha NaN
        df["LotFrontage"] = df["LotFrontage"].fillna(df["LotFrontage"].median())

    # --- 1d. GarageYrBlt: substituir por YearBuilt ---
    if "GarageYrBlt" in df.columns:
        df["GarageYrBlt"] = df["GarageYrBlt"].fillna(df["YearBuilt"])

    # --- 1e. Variáveis com poucos NaN: moda ---
    mode_cols = ["Electrical", "MSZoning", "Utilities", "Functional",
                 "KitchenQual", "Exterior1st", "Exterior2nd", "SaleType"]
    for col in mode_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].mode()[0])

    return df


# ─────────────────────────────────────────────
# FASE 2 — ENCODING DE VARIÁVEIS CATEGÓRICAS
# ─────────────────────────────────────────────

# Mapas ordinais explícitos (preservam a ordem natural)
QUALITY_MAP = {"None": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5}

ORDINAL_MAPS = {
    "ExterQual":    QUALITY_MAP,
    "ExterCond":    QUALITY_MAP,
    "BsmtQual":     QUALITY_MAP,
    "BsmtCond":     QUALITY_MAP,
    "HeatingQC":    QUALITY_MAP,
    "KitchenQual":  QUALITY_MAP,
    "GarageQual":   QUALITY_MAP,
    "GarageCond":   QUALITY_MAP,
    "FireplaceQu":  QUALITY_MAP,
    "PoolQC":       QUALITY_MAP,
    "BsmtExposure": {"None": 0, "No": 0, "Mn": 1, "Av": 2, "Gd": 3},
    "BsmtFinType1": {"None": 0, "Unf": 0, "LwQ": 1, "Rec": 2, "BLQ": 3, "ALQ": 4, "GLQ": 5},
    "BsmtFinType2": {"None": 0, "Unf": 0, "LwQ": 1, "Rec": 2, "BLQ": 3, "ALQ": 4, "GLQ": 5},
    "GarageFinish": {"None": 0, "Unf": 0, "RFn": 1, "Fin": 2},
    "LotShape":     {"IR3": 0, "IR2": 1, "IR1": 2, "Reg": 3},
    "LandSlope":    {"Sev": 0, "Mod": 1, "Gtl": 2},
    "PavedDrive":   {"N": 0, "P": 1, "Y": 2},
    "Functional":   {"Sal": 0, "Sev": 1, "Maj2": 2, "Maj1": 3,
                     "Mod": 4, "Min2": 5, "Min1": 6, "Typ": 7},
}

# Variáveis binárias simples
BINARY_MAPS = {
    "CentralAir": {"Y": 1, "N": 0},
    "Street":     {"Pave": 1, "Grvl": 0},
}

# Nominais para one-hot (excluindo Neighborhood que usa target encoding)
NOMINAL_COLS = [
    "MSZoning", "LotConfig", "LandContour", "Neighborhood",
    "Condition1", "Condition2", "BldgType", "HouseStyle",
    "RoofStyle", "RoofMatl",
    "Exterior1st", "Exterior2nd", "MasVnrType",
    "Foundation", "Heating",
    "GarageType", "MiscFeature",
    "SaleType", "SaleCondition",
    "Electrical", "Alley", "Fence",
]


def encode_ordinals(df: pd.DataFrame) -> pd.DataFrame:
    """Mapeia variáveis ordinais para inteiros."""
    df = df.copy()
    for col, mapping in ORDINAL_MAPS.items():
        if col in df.columns:
            df[col] = df[col].map(mapping).fillna(0).astype(int)
    return df


def encode_binary(df: pd.DataFrame) -> pd.DataFrame:
    """Converte variáveis binárias para 0/1."""
    df = df.copy()
    for col, mapping in BINARY_MAPS.items():
        if col in df.columns:
            df[col] = df[col].map(mapping).fillna(0).astype(int)
    return df


def target_encode_neighborhood(
    df_train_student: pd.DataFrame,
    df_test_student: pd.DataFrame,
    target: pd.Series,
    n_splits: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    
    df_train_student = df_train_student.copy()
    df_test_student  = df_test_student.copy()
    global_mean = target.mean()

    # 1. Calcular médias para o TESTE (antes de mexer no treino)
    # Usamos os dados originais passados para a função
    full_mean = (
        df_train_student.assign(target=target.values)
        .groupby("Neighborhood")["target"]
        .mean()
    )

    # 2. Treino: calcular OOF (out-of-fold) para evitar data leakage
    oof_encoded = np.zeros(len(df_train_student))
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    for train_idx, val_idx in kf.split(df_train_student):
        fold_mean = (
            df_train_student.iloc[train_idx]
            .assign(target=target.iloc[train_idx])
            .groupby("Neighborhood")["target"]
            .mean()
        )
        oof_encoded[val_idx] = (
            df_train_student.iloc[val_idx]["Neighborhood"]
            .map(fold_mean)
            .fillna(global_mean)
            .values
        )

    # 3. Aplicar os resultados
    df_train_student["Neighborhood_enc"] = oof_encoded
    df_test_student["Neighborhood_enc"] = (
        df_test_student["Neighborhood"].map(full_mean).fillna(global_mean)
    )

    # 4. AGORA SIM: Deletar a coluna original de ambos
    df_train_student = df_train_student.drop(columns="Neighborhood")
    df_test_student = df_test_student.drop(columns="Neighborhood")

    return df_train_student, df_test_student

def one_hot_encode(df_train_student, df_test_student):
    nominal = [c for c in NOMINAL_COLS if c in df_train_student.columns and c != "Neighborhood"]
    
    # Adicionamos o dtype=int para evitar colunas True/False
    df_train_student_enc = pd.get_dummies(df_train_student, columns=nominal, drop_first=True, dtype=int)
    df_test_student_enc  = pd.get_dummies(df_test_student,  columns=nominal, drop_first=True, dtype=int)

    df_train_student_enc, df_test_student_enc = df_train_student_enc.align(
        df_test_student_enc, join="left", axis=1, fill_value=0
    )
    return df_train_student_enc, df_test_student_enc


# ─────────────────────────────────────────────
# FASE 3 — OUTLIERS
# ─────────────────────────────────────────────

def remove_outliers(df: pd.DataFrame, target: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    """
    Remove outliers documentados do conjunto de treino.
    Não aplicar no conjunto de teste!
    """
    # Outliers clássicos: casas com área enorme mas preço anormalmente baixo
    mask = ~((df["GrLivArea"] > 4000) & (target < 300000))
    df     = df[mask].reset_index(drop=True)
    target = target[mask].reset_index(drop=True)

    # Opcional: manter apenas vendas normais (reduz ruído)
    # mask_normal = df["SaleCondition"] == "Normal"
    # df     = df[mask_normal].reset_index(drop=True)
    # target = target[mask_normal].reset_index(drop=True)

    return df, target


def log_transform_target(target: pd.Series) -> pd.Series:
    """Aplica log1p no target para normalizar a distribuição assimétrica."""
    return np.log1p(target)


def inverse_transform_target(predictions: np.ndarray) -> np.ndarray:
    """Reverte o log1p para obter preços reais."""
    return np.expm1(predictions)


# ─────────────────────────────────────────────
# FASE 4 — FEATURE ENGINEERING
# ─────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cria features derivadas a partir das colunas originais.
    Deve ser aplicado ANTES do encoding e escala.
    """
    df = df.copy()

    # --- 4a. Áreas combinadas ---
    df["TotalSF"] = (
        df.get("TotalBsmtSF", 0) +
        df.get("1stFlrSF", 0) +
        df.get("2ndFlrSF", 0)
    )
    df["TotalPorchSF"] = (
        df.get("OpenPorchSF", 0) +
        df.get("EnclosedPorch", 0) +
        df.get("3SsnPorch", 0) +
        df.get("ScreenPorch", 0)
    )
    df["TotalBathrooms"] = (
        df.get("FullBath", 0) +
        0.5 * df.get("HalfBath", 0) +
        df.get("BsmtFullBath", 0) +
        0.5 * df.get("BsmtHalfBath", 0)
    )

    # --- 4b. Features temporais ---
    yr_sold = df.get("YrSold", 2010)
    df["HouseAge"]   = yr_sold - df["YearBuilt"]
    df["RemodAge"]   = yr_sold - df["YearRemodAdd"]
    df["GarageAge"]  = yr_sold - df.get("GarageYrBlt", df["YearBuilt"])
    df["IsRemodeled"] = (df["YearRemodAdd"] != df["YearBuilt"]).astype(int)

    # Garante que ages não sejam negativas (dados inconsistentes)
    for col in ["HouseAge", "RemodAge", "GarageAge"]:
        df[col] = df[col].clip(lower=0)

    # --- 4c. Flags binárias de existência ---
    df["HasPool"]      = (df.get("PoolArea", 0) > 0).astype(int)
    df["HasGarage"]    = (df.get("GarageArea", 0) > 0).astype(int)
    df["HasBsmt"]      = (df.get("TotalBsmtSF", 0) > 0).astype(int)
    df["HasFireplace"] = (df.get("Fireplaces", 0) > 0).astype(int)
    df["Has2ndFloor"]  = (df.get("2ndFlrSF", 0) > 0).astype(int)
    df["HasAlley"]     = (df.get("Alley", "None") != "None").astype(int)

    # --- 4d. Interações (muito preditivas em regressão linear) ---
    df["QualxArea"]    = df["OverallQual"] * df["GrLivArea"]
    df["QualxAge"]     = df["OverallQual"] * df["HouseAge"]
    df["OverallScore"] = df["OverallQual"] * df["OverallCond"]
    df["QualxTotalSF"] = df["OverallQual"] * df["TotalSF"]

    # --- 4e. Colunas redundantes após engenharia ---
    cols_to_drop = [
        "YearBuilt", "YearRemodAdd", "GarageYrBlt",   # substituídas pelas ages
        "1stFlrSF", "2ndFlrSF", "TotalBsmtSF",         # substituídas por TotalSF
        "OpenPorchSF", "EnclosedPorch", "3SsnPorch",    # substituídas por TotalPorchSF
        "ScreenPorch",
        "FullBath", "HalfBath", "BsmtFullBath",         # substituídas por TotalBathrooms
        "BsmtHalfBath",
        "Utilities",   # quase sem variância (quase todos "AllPub")
        "MoSold",      # baixa importância preditiva
    ]
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

    return df


# ─────────────────────────────────────────────
# FASE 5 — TRANSFORMAÇÃO E ESCALA
# ─────────────────────────────────────────────

def log_skewed_features(df: pd.DataFrame, threshold: float = 0.75) -> pd.DataFrame:
    """
    Aplica log1p em features numéricas com assimetria acima do threshold.
    Só aplica em colunas com valores não-negativos.
    """
    df = df.copy()
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    skewness = df[num_cols].apply(lambda x: stats.skew(x.dropna()))
    skewed   = skewness[abs(skewness) > threshold].index.tolist()

    for col in skewed:
        if df[col].min() >= 0:  # log1p só faz sentido para valores >= 0
            df[col] = np.log1p(df[col])

    print(f"  Log1p aplicado em {len(skewed)} features: {skewed[:8]}{'...' if len(skewed) > 8 else ''}")
    return df


def scale_features(
    df_train_student: pd.DataFrame,
    df_test_student: pd.DataFrame,
    method: str = "robust",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Escala as features numéricas.
    - 'robust'   → RobustScaler (resistente a outliers restantes) — recomendado
    - 'standard' → StandardScaler (z-score clássico)
    Fit apenas no treino, transform em treino e teste.
    """
    num_cols = df_train_student.select_dtypes(include=[np.number]).columns.tolist()

    scaler = RobustScaler() if method == "robust" else StandardScaler()
    df_train_student[num_cols] = scaler.fit_transform(df_train_student[num_cols])
    df_test_student[num_cols]  = scaler.transform(df_test_student[num_cols])

    return df_train_student, df_test_student


def check_vif(df: pd.DataFrame, threshold: float = 10.0) -> pd.DataFrame:
    """
    Calcula o Variance Inflation Factor (VIF) para detectar multicolinearidade.
    Retorna um DataFrame ordenado. Features com VIF > threshold são problemáticas.
    """
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    X = df[num_cols].dropna()

    vif_data = pd.DataFrame({
        "feature": num_cols,
        "VIF": [variance_inflation_factor(X.values, i) for i in range(X.shape[1])],
    }).sort_values("VIF", ascending=False)

    high_vif = vif_data[vif_data["VIF"] > threshold]
    if not high_vif.empty:
        print(f"\n  Atenção — {len(high_vif)} features com VIF > {threshold}:")
        print(high_vif.to_string(index=False))

    return vif_data


# ─────────────────────────────────────────────
# PIPELINE PRINCIPAL
# ─────────────────────────────────────────────

def build_pipeline(
    df_train_student_raw: pd.DataFrame,
    df_test_student_raw: pd.DataFrame,
    target_col: str = "SalePrice",
    check_multicollinearity: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """
    Executa o pipeline completo de limpeza + feature engineering.

    Parâmetros
    ----------
    df_train_student_raw : DataFrame com os dados de treino (inclui SalePrice)
    df_test_student_raw  : DataFrame com os dados de teste (sem SalePrice)
    target_col   : nome da coluna target
    check_multicollinearity : se True, calcula VIF (pode ser lento)

    Retorna
    -------
    X_train : features de treino prontas para modelagem
    X_test  : features de teste prontas para predição
    y_train : target transformado com log1p
    """
    print("=" * 55)
    print("  PIPELINE AMES HOUSING")
    print("=" * 55)

    # Separar target
    y_raw = df_train_student_raw[target_col].copy()
    df_train_student = df_train_student_raw.drop(columns=[target_col, "Id"], errors="ignore").copy()
    df_test_student  = df_test_student_raw.drop(columns=["Id"], errors="ignore").copy()

    # --- Fase 1: Valores ausentes ---
    print("\n[1/6] Tratando valores ausentes...")
    df_train_student = handle_missing_values(df_train_student)
    df_test_student  = handle_missing_values(df_test_student)

    # --- Fase 3a: Remover outliers (só no treino) ---
    print("[2/6] Removendo outliers do treino...")
    df_train_student, y_raw = remove_outliers(df_train_student, y_raw)
    print(f"  Registros de treino após limpeza: {len(df_train_student)}")

    # --- Fase 4: Feature engineering ---
    print("[3/6] Criando features derivadas...")
    df_train_student = engineer_features(df_train_student)
    df_test_student  = engineer_features(df_test_student)

    # --- Fase 2a: Encoding ordinal e binário ---
    print("[4/6] Encoding de variáveis categóricas...")
    df_train_student = encode_ordinals(df_train_student)
    df_test_student  = encode_ordinals(df_test_student)
    df_train_student = encode_binary(df_train_student)
    df_test_student  = encode_binary(df_test_student)

    # --- Fase 2b: Target encoding do Neighborhood ---
    y_log = log_transform_target(y_raw)
    df_train_student, df_test_student = target_encode_neighborhood(df_train_student, df_test_student, y_log)

    # --- Fase 2c: One-hot encoding das nominais ---
    df_train_student, df_test_student = one_hot_encode(df_train_student, df_test_student)

    # --- Fase 5a: Log em features assimétricas ---
    print("[5/6] Transformando features assimétricas...")
    df_train_student = log_skewed_features(df_train_student)
    df_test_student  = log_skewed_features(df_test_student)

    # --- Fase 5b: Escala ---
    print("[6/6] Escalando features...")
    df_train_student, df_test_student = scale_features(df_train_student, df_test_student, method="robust")

    # --- Opcional: verificar multicolinearidade ---
    if check_multicollinearity:
        print("\n[Extra] Calculando VIF...")
        check_vif(df_train_student)

    print(f"\n  Shape final — Treino: {df_train_student.shape} | Teste: {df_test_student.shape}")
    print("=" * 55)

    return df_train_student, df_test_student, y_log


# ─────────────────────────────────────────────
# MODELAGEM — REGRESSÃO LINEAR COM REGULARIZAÇÃO
# ─────────────────────────────────────────────

def evaluate_models(X_train: pd.DataFrame, y_train: pd.Series, cv: int = 5):
    """
    Avalia Ridge e Lasso com cross-validation e reporta RMSLE.
    """
    kf = KFold(n_splits=cv, shuffle=True, random_state=42)

    models = {
        "Ridge (alpha=10)":   Ridge(alpha=10),
        "Ridge (alpha=1)":    Ridge(alpha=1),
        "Lasso (alpha=0.001)": Lasso(alpha=0.001, max_iter=10000),
        "OLS":                 LinearRegression(),
    }

    print("\n" + "=" * 55)
    print("  AVALIAÇÃO DE MODELOS (cross-validation)")
    print(f"  Métrica: RMSE no espaço log (RMSLE)")
    print("=" * 55)

    results = {}
    for name, model in models.items():
        scores = cross_val_score(
            model, X_train, y_train,
            scoring="neg_root_mean_squared_error",
            cv=kf,
        )
        rmse_mean = -scores.mean()
        rmse_std  = scores.std()
        results[name] = rmse_mean
        print(f"  {name:<25} RMSE: {rmse_mean:.5f} ± {rmse_std:.5f}")

    best = min(results, key=results.get)
    print(f"\n  Melhor modelo: {best} (RMSE: {results[best]:.5f})")
    return results


def tune_ridge(X_train: pd.DataFrame, y_train: pd.Series, cv: int = 5) -> Ridge:
    """
    Busca o melhor alpha para Ridge por cross-validation.
    """
    from sklearn.linear_model import RidgeCV

    alphas = np.logspace(-3, 4, 50)  # de 0.001 a 10000
    kf = KFold(n_splits=cv, shuffle=True, random_state=42)

    ridge_cv = RidgeCV(alphas=alphas, cv=kf, scoring="neg_root_mean_squared_error")
    ridge_cv.fit(X_train, y_train)

    print(f"\n  Melhor alpha para Ridge: {ridge_cv.alpha_:.4f}")
    return Ridge(alpha=ridge_cv.alpha_)


def tune_lasso(X_train: pd.DataFrame, y_train: pd.Series, cv: int = 5) -> Lasso:
    """
    Busca o melhor alpha para Lasso por cross-validation.
    """
    from sklearn.linear_model import LassoCV

    kf = KFold(n_splits=cv, shuffle=True, random_state=42)
    lasso_cv = LassoCV(alphas=None, cv=kf, max_iter=10000, random_state=42)
    lasso_cv.fit(X_train, y_train)

    print(f"  Melhor alpha para Lasso: {lasso_cv.alpha_:.6f}")
    n_selected = np.sum(lasso_cv.coef_ != 0)
    print(f"  Features selecionadas pelo Lasso: {n_selected}/{X_train.shape[1]}")
    return Lasso(alpha=lasso_cv.alpha_, max_iter=10000)


def predict_and_submit(
    model,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    test_ids: pd.Series,
    output_path: str = "submission.csv",
):
    """
    Treina o modelo final e gera o arquivo de submissão.
    """
    model.fit(X_train, y_train)
    predictions_log = model.predict(X_test)
    predictions     = inverse_transform_target(predictions_log)

    submission = pd.DataFrame({
        "Id":        test_ids.values,
        "SalePrice": predictions,
    })
    submission.to_csv(output_path, index=False)
    print(f"\n  Submissão salva em: {output_path}")
    print(f"  Previsões: min=${predictions.min():,.0f} | "
          f"max=${predictions.max():,.0f} | "
          f"média=${predictions.mean():,.0f}")
    return submission


# ─────────────────────────────────────────────
# EXEMPLO DE USO COMPLETO
# ─────────────────────────────────────────────

if __name__ == "__main__":
    # 1. Carregar dados
    df_train_student = pd.read_csv("train_student.csv")
    df_test_student  = pd.read_csv("test_student.csv")
    test_ids = df_test_student["Id"]

    # 2. Executar pipeline
    X_train, X_test, y_train = build_pipeline(
        df_train_student,
        df_test_student,
        check_multicollinearity=False,  # True para análise detalhada de VIF
    )

    # 3. Comparar modelos base
    evaluate_models(X_train, y_train)

    # 4. Tunar Ridge e Lasso automaticamente
    ridge = tune_ridge(X_train, y_train)
    lasso = tune_lasso(X_train, y_train)

    # 5. Escolher o melhor e gerar submissão
    # (ajuste conforme o resultado do evaluate_models)
    best_model = ridge
    submission = predict_and_submit(
        best_model, X_train, y_train, X_test,
        test_ids=test_ids,
        output_path="submission.csv",
    )

    print("\nPronto! ✓")
