# Base Analítica de Modelagem — `gold_base_ml_alunos`

> Gerada por `src/preprocessing/build_dataset.py`, materializada na camada **Gold**
> do BigQuery (projeto da Fase 2) e exportada para `data/gold_base_ml_alunos.parquet`.

## Grão e alvo

- **1 linha = 1 aluno avaliado** (Avaliação da Alfabetização / Alfabetiza Brasil, INEP).
- **3.355.846 linhas**: 1.503.058 de 2023 e 1.852.788 de 2024, apenas alunos **presentes**.
- **Alvo:** `alfabetizado` (booleano) — proficiência ≥ 743 na escala Saeb.
  Balanceamento: 59,1% alfabetizados × 40,9% não alfabetizados.

## Por que só presentes?

Na origem, alunos ausentes chegam com `alfabetizado = 0` preenchido — mas isso é
artefato de preenchimento, não medição (ninguém avaliou a criança). Manter esses
registros ensinaria o modelo com rótulos falsos. Ausência vira análise de contexto
na EDA, não linha de treino.

## Decisões anti-data-leakage

| # | Decisão | Motivo |
|---|---|---|
| 1 | `proficiencia` excluída | O alvo é literalmente `proficiencia >= 743`; usar a nota é prever o alvo com o próprio alvo |
| 2 | Nenhuma taxa de alfabetização do mesmo ano | A taxa municipal de 2024 foi calculada com os mesmos alunos que o modelo tenta prever |
| 3 | IDEB da edição **anterior** (2021→alunos 2023; 2023→alunos 2024) | Contexto educacional sem contaminação temporal |
| 4 | PIB per capita do **ano anterior** | Idem |
| 5 | `id_municipio`/`id_escola` são chaves de grupo, não features | Alta cardinalidade = decorar território; usadas no GroupKFold e nas agregações estratégicas |
| 6 | Pré-processamento (imputação/encoding/escala) dentro do Pipeline sklearn | `fit` apenas no treino, sempre |

## Dicionário de colunas (39)

### Chaves e pesos (não entram no modelo)
| Coluna | Descrição |
|---|---|
| `ano` | Ano da avaliação (2023/2024) — usado no split temporal |
| `id_municipio` | Código IBGE (7 dígitos) — chave de grupo e de agregação |
| `id_escola` | Código **anonimizado** pelo INEP — chave de grupo |
| `peso_aluno` | Peso amostral do aluno (métricas ponderadas) |

### Alvo
| Coluna | Descrição |
|---|---|
| `alfabetizado` | Aluno atingiu 743+ na escala Saeb (booleano) |

### Features — aluno e escola
| Coluna | Fonte | Descrição |
|---|---|---|
| `rede` | INEP | Federal / Estadual / Municipal / Privada |
| `alunos_avaliados_escola` | derivada | Porte da escola (nº de alunos avaliados no ano) |

### Features — território
| Coluna | Descrição |
|---|---|
| `regiao` | Região (1º dígito do código IBGE) |
| `sigla_uf` | UF (2 primeiros dígitos do código IBGE) |

### Features — Censo Escolar (municipal, ano corrente; escolas em atividade com anos iniciais)
`n_escolas_anos_iniciais`, `pct_escolas_urbanas`, `pct_biblioteca_ou_sala_leitura`,
`pct_internet`, `pct_internet_alunos`, `pct_lab_informatica`, `pct_agua_potavel`,
`pct_esgoto_rede_publica`, `pct_energia_rede_publica`, `pct_quadra_esportes`,
`pct_alimentacao`, `media_matriculas_ai_por_escola`, `pct_equipamento_computador`

> Infraestrutura é atributo conhecido no início do ano letivo — não é desfecho
> da avaliação. `quantidade_computador_aluno` foi descartada: 100% nula no Censo
> 2022–2024 (campo descontinuado); o flag `equipamento_computador` é o que existe.

### Features — IBGE
| Coluna | Descrição |
|---|---|
| `populacao` | População municipal no ano da avaliação |
| `pib_per_capita_ano_anterior` | PIB municipal ÷ população, ambos do ano anterior |

### Features — Atlas do Desenvolvimento Humano (Censo 2010, estrutural)
`idhm`, `idhm_e`, `idhm_l`, `idhm_r`, `indice_gini`, `renda_pc`, `prop_pobreza`,
`prop_pobreza_criancas`, `taxa_analfabetismo_15_mais`,
`taxa_criancas_fora_escola_6_14`, `expectativa_anos_estudo`, `taxa_agua_encanada`

### Features — IDEB (anos iniciais, rede pública, edição anterior)
`ideb_ai_publica_anterior`, `taxa_aprovacao_ai_anterior`, `nota_saeb_lp_ai_anterior`

## Cobertura dos joins (% de nulos)

Excelente: pior caso é o IDEB com 1,55% de nulos (municípios sem edição anterior
publicada); Atlas ≈ 0,04%; demais fontes ≈ 0%. A imputação desses casos acontece
dentro do Pipeline sklearn (mediana), com o `fit` restrito ao treino.

## Limitação documentada: cruzamento por escola é impossível

O `id_escola` dos microdados é **anonimizado pelo INEP** (sequencial 60000001+,
42.811 escolas): foi testado o join contra o Censo Escolar e o casamento é **zero**.
Por isso o enriquecimento escolar é agregado no nível municipal. Fica como
limitação conhecida do projeto (a alternativa exigiria acesso restrito ao INEP).

## Tabelas Gold auxiliares exportadas (aplicação estratégica)

- `gold_indicador_municipio.parquet` — taxa oficial por município/ano/rede;
- `gold_metas_x_resultados.parquet` — taxa observada × meta 2030 (gap);
- `gold_evolucao_temporal.parquet` — variação 2023 → 2024;
- `gold_taxa_municipio_microdados.parquet` — taxa recalculada dos microdados.
