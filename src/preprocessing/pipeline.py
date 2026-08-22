"""
Pipeline de pré-processamento integrado ao modelo (requisito do enunciado).

Todas as transformações (imputação, escala, encoding) vivem dentro de um
ColumnTransformer que entra num Pipeline do sklearn junto com o estimador.
Consequência: o `fit` das transformações acontece SEMPRE e SOMENTE no conjunto
de treino de cada split; o sklearn garante isso por construção, eliminando o
vazamento clássico de "ajustar o scaler/imputer na base inteira".
"""
from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ---------------------------------------------------------------------------
# Contrato de colunas da base gold_base_ml_alunos
# ---------------------------------------------------------------------------
TARGET = "alfabetizado"
WEIGHT = "peso_aluno"
# Chaves de grupo/split: nunca entram como feature (alta cardinalidade = decorar território)
KEYS = ["ano", "id_municipio", "id_escola"]

CAT_FEATURES = ["rede", "regiao", "sigla_uf"]

NUM_FEATURES = [
    # aluno / escola
    "alunos_avaliados_escola",
    # Censo Escolar (municipal)
    "n_escolas_anos_iniciais", "pct_escolas_urbanas",
    "pct_biblioteca_ou_sala_leitura", "pct_internet", "pct_internet_alunos",
    "pct_lab_informatica", "pct_agua_potavel", "pct_esgoto_rede_publica",
    "pct_energia_rede_publica", "pct_quadra_esportes", "pct_alimentacao",
    "media_matriculas_ai_por_escola", "pct_equipamento_computador",
    # IBGE
    "populacao", "pib_per_capita_ano_anterior",
    # Atlas ADH 2010
    "idhm", "idhm_e", "idhm_l", "idhm_r", "indice_gini", "renda_pc",
    "prop_pobreza", "prop_pobreza_criancas", "taxa_analfabetismo_15_mais",
    "taxa_criancas_fora_escola_6_14", "expectativa_anos_estudo",
    "taxa_agua_encanada",
    # IDEB (edição anterior)
    "ideb_ai_publica_anterior", "taxa_aprovacao_ai_anterior",
    "nota_saeb_lp_ai_anterior",
]

ALL_FEATURES = CAT_FEATURES + NUM_FEATURES


def load_dataset(path: str = "data/gold_base_ml_alunos.parquet") -> pd.DataFrame:
    """Carrega a base de modelagem com dtypes econômicos (máquina de 8 GB)."""
    df = pd.read_parquet(path)
    for c in NUM_FEATURES:
        df[c] = df[c].astype("float32")
    df[TARGET] = df[TARGET].astype("int8")
    return df


def build_preprocessor(scale: bool = False) -> ColumnTransformer:
    """
    Pré-processamento:
      - numéricas: imputação pela MEDIANA (robusta a caudas longas como população
        e PIB per capita) + padronização opcional (necessária só para o modelo linear);
      - categóricas: One-Hot Encoding com `handle_unknown="ignore"` (categoria nova
        em produção não derruba o modelo).
    """
    num_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale:
        num_steps.append(("scaler", StandardScaler()))
    return ColumnTransformer(
        [
            ("num", Pipeline(num_steps), NUM_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=True), CAT_FEATURES),
        ],
        verbose_feature_names_out=False,
    )


def make_pipeline(estimator, scale: bool = False) -> Pipeline:
    """Pipeline completo: pré-processamento + estimador (leakage-safe por construção)."""
    return Pipeline([("prep", build_preprocessor(scale=scale)), ("model", estimator)])
