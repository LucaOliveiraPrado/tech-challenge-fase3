"""
Identidade visual dos gráficos do projeto.

Centraliza paleta e tema para que notebooks e scripts produzam figuras
consistentes entre si (mesmos tons nos relatórios, README e vídeo).
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import seaborn as sns

# Paleta do projeto
AZUL_ESCURO = "#264653"   # séries principais / neutro
VERDE = "#2a9d8f"         # resultados positivos
LARANJA = "#e76f51"       # risco / alerta
CINZA = "#8d99ae"         # referências e linhas de apoio

# Escala contínua para mapas de risco (verde -> amarelo -> laranja -> vermelho)
CMAP_RISCO = "RdYlGn_r"


def aplicar_tema(dpi: int = 110) -> None:
    """Tema padrão dos gráficos (chamar uma vez no início de cada notebook)."""
    sns.set_theme(style="whitegrid", palette="deep")
    plt.rcParams["figure.dpi"] = dpi
