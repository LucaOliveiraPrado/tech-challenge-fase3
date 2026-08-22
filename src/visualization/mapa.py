"""
Mapa coroplético municipal do Brasil.

A malha vem do geobr (IBGE, 2020, simplificada) e fica cacheada em
data/geo_municipios.parquet para os notebooks não dependerem de rede.
"""
from __future__ import annotations

import pathlib

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd

from .style import CMAP_RISCO

GEO_PATH = pathlib.Path(__file__).resolve().parents[2] / "data" / "geo_municipios.parquet"


def carregar_malha() -> gpd.GeoDataFrame:
    """Carrega a malha municipal (baixa uma única vez via geobr se preciso)."""
    if not GEO_PATH.exists():
        import geobr
        geobr.read_municipality(year=2020, simplified=True).to_parquet(GEO_PATH)
    geo = gpd.read_parquet(GEO_PATH)
    geo["id_municipio"] = geo["code_muni"].astype(int).astype(str)
    return geo[["id_municipio", "geometry"]]


def mapa_municipal(valores: pd.DataFrame, coluna: str, titulo: str,
                   legenda: str, caminho_png: str | pathlib.Path,
                   cmap: str = CMAP_RISCO) -> None:
    """
    Pinta um indicador municipal no mapa do Brasil.

    `valores` precisa de `id_municipio` (código IBGE, 7 dígitos) e da `coluna`.
    Municípios sem dado ficam em cinza-claro (ex.: Roraima, sem avaliação).
    """
    geo = carregar_malha().merge(valores[["id_municipio", coluna]],
                                 on="id_municipio", how="left")
    fig, ax = plt.subplots(figsize=(9, 9))
    geo.plot(column=coluna, cmap=cmap, linewidth=0, ax=ax,
             legend=True, legend_kwds={"shrink": 0.6, "label": legenda},
             missing_kwds={"color": "#e9ecef", "label": "sem dado"})
    ax.set_axis_off()
    ax.set_title(titulo, fontsize=13)
    plt.tight_layout()
    plt.savefig(caminho_png, bbox_inches="tight", dpi=150)
    plt.show()
