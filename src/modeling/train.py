"""
Treinamento e validação dos modelos de predição de alfabetização.

Protocolo (justificado na EDA e no README):
  1. SPLIT TEMPORAL como avaliação principal: treina em 2023, testa em 2024.
     É o cenário real de uso (prever a próxima edição) e a prova mais honesta
     de generalização.
  2. Comparação de candidatos por validação cruzada AGRUPADA por município
     (StratifiedGroupKFold) dentro de 2023: alunos do mesmo município nunca
     ficam em treino e validação ao mesmo tempo; sem isso o modelo "decora"
     o território e a métrica infla.
  3. A comparação e a busca de hiperparâmetros rodam numa SUBAMOSTRA agrupada
     (limite de memória da máquina, 8 GB); o campeão é re-treinado na base
     completa de 2023 e avaliado em 2024.
  4. Replicabilidade: SEED fixa em tudo; ambiente versionado em requirements.txt.

Uso:
    uv run python -m src.modeling.train        # a partir da raiz do repo
"""
from __future__ import annotations

import json
import pathlib
import sys
import time

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    RandomizedSearchCV, StratifiedGroupKFold, cross_val_score,
)

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from src.evaluation.metrics import metricas  # noqa: E402
from src.preprocessing.pipeline import (  # noqa: E402
    ALL_FEATURES, TARGET, WEIGHT, load_dataset, make_pipeline,
)

SEED = 42
SUBSAMPLE = 400_000          # linhas para comparação/tuning (memória)
REPORTS = pathlib.Path("reports")
DATA = pathlib.Path("data")


def sample_grouped(df: pd.DataFrame, n: int, seed: int = SEED) -> pd.DataFrame:
    """Subamostra por MUNICÍPIO inteiro (preserva a estrutura de grupos)."""
    rng = np.random.default_rng(seed)
    grupos = rng.permutation(np.asarray(df["id_municipio"].unique(), dtype=object))
    tamanhos = df.groupby("id_municipio").size()
    escolhidos, acum = [], 0
    for g in grupos:
        escolhidos.append(g)
        acum += tamanhos[g]
        if acum >= n:
            break
    return df[df["id_municipio"].isin(escolhidos)]


def candidatos() -> dict:
    """Modelos comparados, todos como Pipeline (pré-processamento integrado)."""
    return {
        "baseline_dummy": make_pipeline(
            DummyClassifier(strategy="most_frequent")),
        "regressao_logistica": make_pipeline(
            LogisticRegression(max_iter=2000, random_state=SEED), scale=True),
        "random_forest": make_pipeline(
            RandomForestClassifier(
                n_estimators=200, max_depth=16, min_samples_leaf=20,
                max_samples=0.5, n_jobs=-1, random_state=SEED)),
        "lightgbm": make_pipeline(
            LGBMClassifier(
                n_estimators=400, learning_rate=0.08, num_leaves=63,
                random_state=SEED, n_jobs=-1, verbosity=-1)),
    }


def espaco_busca() -> dict:
    return {
        "model__n_estimators": [200, 400, 600, 800],
        "model__learning_rate": [0.03, 0.05, 0.08, 0.12],
        "model__num_leaves": [31, 63, 127],
        "model__min_child_samples": [20, 50, 100, 200],
        "model__subsample": [0.7, 0.85, 1.0],
        "model__colsample_bytree": [0.7, 0.85, 1.0],
        "model__reg_lambda": [0.0, 1.0, 5.0],
    }


def main() -> None:
    t0 = time.time()
    REPORTS.mkdir(exist_ok=True)
    df = load_dataset()
    treino = df[df["ano"] == 2023]
    teste = df[df["ano"] == 2024]
    print(f"treino 2023: {len(treino):,} | teste 2024: {len(teste):,}")

    # ------------------------------------------------------------------
    # 1) Comparação de candidatos: CV agrupada por município (subamostra)
    # ------------------------------------------------------------------
    amostra = sample_grouped(treino, SUBSAMPLE)
    Xa, ya, ga = amostra[ALL_FEATURES], amostra[TARGET], amostra["id_municipio"]
    cv = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=SEED)
    comparacao = {}
    for nome, pipe in candidatos().items():
        t = time.time()
        scores = cross_val_score(pipe, Xa, ya, groups=ga, cv=cv,
                                 scoring="roc_auc", n_jobs=1)
        comparacao[nome] = {"roc_auc_cv": float(scores.mean()),
                            "desvio": float(scores.std())}
        print(f"[cv] {nome}: AUC {scores.mean():.4f} ± {scores.std():.4f} "
              f"({time.time()-t:.0f}s)")

    # ------------------------------------------------------------------
    # 2) Otimização do campeão (LightGBM) na mesma subamostra
    # ------------------------------------------------------------------
    busca = RandomizedSearchCV(
        candidatos()["lightgbm"], espaco_busca(), n_iter=12, cv=cv,
        scoring="roc_auc", random_state=SEED, n_jobs=1, refit=False, verbose=1)
    busca.fit(Xa, ya, groups=ga)
    print(f"[tuning] melhor AUC cv: {busca.best_score_:.4f}")
    print(f"[tuning] params: {busca.best_params_}")

    # ------------------------------------------------------------------
    # 3) Re-treino do campeão na base completa de 2023 + avaliação em 2024
    # ------------------------------------------------------------------
    final = candidatos()["lightgbm"]
    final.set_params(**busca.best_params_)
    final.fit(treino[ALL_FEATURES], treino[TARGET])

    resultado = {"comparacao_cv": comparacao,
                 "tuning": {"best_score_cv": float(busca.best_score_),
                            "best_params": busca.best_params_},
                 "avaliacao_temporal_2024": {}}
    for nome, base in [("treino_2023", treino), ("teste_2024", teste)]:
        prob = final.predict_proba(base[ALL_FEATURES])[:, 1]
        pred = (prob >= 0.5).astype(int)
        w = base[WEIGHT].fillna(0)
        resultado["avaliacao_temporal_2024"][nome] = {
            "simples": metricas(base[TARGET], pred, prob),
            "ponderada_peso_amostral": metricas(base[TARGET], pred, prob, sample_weight=w),
        }
        print(f"[{nome}] AUC {resultado['avaliacao_temporal_2024'][nome]['simples']['roc_auc']:.4f}")

    # probabilidades por aluno no teste (insumo da aplicação estratégica)
    teste_out = teste[["ano", "id_municipio", "id_escola", WEIGHT, TARGET]].copy()
    teste_out["prob_alfabetizado"] = final.predict_proba(teste[ALL_FEATURES])[:, 1]
    teste_out.to_parquet(DATA / "predicoes_teste_2024.parquet", index=False)

    joblib.dump(final, DATA / "modelo_final.joblib")
    with open(REPORTS / "metricas_modelagem.json", "w") as f:
        json.dump(resultado, f, indent=2, ensure_ascii=False)
    print(f"OK em {time.time()-t0:.0f}s; modelo em data/modelo_final.joblib, "
          f"métricas em reports/metricas_modelagem.json")


if __name__ == "__main__":
    main()
