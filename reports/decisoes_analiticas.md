# Registro de Decisões Analíticas

> Cada decisão relevante do projeto, com o problema, as alternativas
> consideradas e o porquê da escolha. Complementa o README (visão geral) e o
> `reports/base_analitica.md` (dicionário da base).

---

## D1 — Grão do modelo: aluno (não município)

**Problema:** o enunciado pede prever se "um aluno" será alfabetizado; a base
tem tanto microdados por aluno quanto agregados municipais.
**Alternativas:** (a) classificar alunos; (b) regredir a taxa municipal.
**Decisão:** (a). É o que o enunciado pede, usa 3,35 mi de observações em vez
de ~11 mil, e a visão municipal continua disponível agregando as probabilidades
individuais (é como o ranking de risco é construído).

## D2 — População de treino: apenas alunos presentes

**Problema:** os ausentes chegam da fonte com `alfabetizado = 0` preenchido.
**Decisão:** filtrar `presenca = 'Presente'`. O zero do ausente é artefato de
preenchimento, não medição — treinaria o modelo com rótulos falsos (512 mil
casos). A ausência vira tema de contexto, não linha de treino.
**Rastro:** herda o apontamento da revisão da Fase 2 sobre nulos silenciosos no
rótulo dos microdados.

## D3 — `proficiencia` fora das features

**Problema:** a nota na escala Saeb está na base e é altamente disponível.
**Decisão:** excluir sempre. O alvo é definido como `proficiencia >= 743`;
usá-la é prever o alvo com o próprio alvo (leakage direto e total).

## D4 — Nenhum agregado de alfabetização do mesmo ano

**Problema:** a taxa municipal do ano corrente é o preditor mais tentador.
**Decisão:** proibir. A taxa de 2024 foi calculada com os mesmos alunos que o
modelo tenta prever (leakage de agregação). Contexto educacional entra apenas
de edições anteriores: IDEB/Saeb ≤ ano−1, PIB do ano anterior, Atlas 2010.

## D5 — Modelo principal sem o histórico do próprio indicador

**Problema:** para alunos de 2024 existe a taxa municipal de 2023; para os de
2023 (primeira edição) não existe equivalente.
**Alternativas:** (a) incluir a feature e imputar para 2023; (b) excluir do
modelo principal e medir o ganho num experimento controlado.
**Decisão:** (b). Imputar uma feature ausente para **toda** uma coorte
contaminaria a comparação temporal. O experimento (notebook 05) quantifica o
ganho em 2024 com GroupKFold pareado — e define o roadmap: a partir da próxima
edição, a versão em produção incorpora o histórico.

## D6 — Validação principal: split temporal 2023 → 2024

**Alternativas:** split aleatório estratificado; split por grupo; split temporal.
**Decisão:** temporal como avaliação principal (é o uso real: prever a próxima
edição), com validação cruzada **agrupada por município** dentro de 2023 para
comparação de modelos e tuning. Split aleatório puro infla a métrica porque
alunos do mesmo município caem em treino e teste.

## D7 — Chaves de grupo fora das features

**Decisão:** `id_municipio` e `id_escola` nunca entram como features (alta
cardinalidade = decorar território); são chaves de agrupamento da validação e
das agregações estratégicas. O território entra por representações de baixa
cardinalidade (região, UF) e pelos atributos do município.

## D8 — Pré-processamento dentro do Pipeline

**Decisão:** `ColumnTransformer` (imputação mediana + `StandardScaler` quando o
estimador precisa + One-Hot com `handle_unknown="ignore"`) **dentro** do
`Pipeline`. O `fit` das transformações acontece somente no treino de cada
split, por construção — em todos os folds da CV e da busca de hiperparâmetros.

## D9 — Comparação e tuning em subamostra agrupada; campeão na base completa

**Problema:** 7 GB de RAM na máquina de treino; RF/CV na base completa não cabem.
**Decisão:** subamostra de ~400 mil alunos por **municípios inteiros**
(preserva a estrutura de grupos) para comparar candidatos e otimizar; o campeão
re-treina na base completa de 2023. Registrado como limitação de protocolo.

## D10 — Escolha do LightGBM

**Dados:** CV agrupada: LogReg 0,641 ± 0,018 · RF 0,649 ± 0,020 · LGBM 0,642 ±
0,013 · LGBM otimizado **0,654**.
**Decisão:** empate técnico antes do tuning; o LightGBM otimizado lidera e
treina em minutos na base completa com memória modesta (o RF equivalente não).
Hiperparâmetros vencedores são conservadores (regularização contra overfitting).

## D11 — Métricas ponderadas pelo peso amostral

**Decisão:** reportar métricas simples **e** ponderadas por `peso_aluno` (o
desenho amostral do INEP dá pesos diferentes aos alunos; a visão ponderada é a
representativa da população). As duas convergem, o que dá robustez à leitura.

## D12 — Enriquecimento escolar no nível municipal

**Problema:** o `id_escola` dos microdados é anonimizado pelo INEP.
**Evidência:** join testado contra o Censo Escolar 2023: **zero** casamento em
42.811 escolas.
**Decisão:** agregar a infraestrutura do Censo Escolar por município (13
indicadores, escolas em atividade com anos iniciais). Limitação documentada.

## D13 — Metas municipais por ano (não só 2030)

**Problema:** a Silver da Fase 2 preservou apenas `meta_alfabetizacao_2030`
(apontado na revisão da fase anterior).
**Decisão:** exportar da Bronze a trajetória completa 2024–2030 e comparar a
performance prevista com a **meta do ano seguinte** (2025), que é a comparação
que importa para gestão.

## D14 — Interpretação honesta da importância das variáveis

**Dados:** permutation importance: UF 0,030 · nota Saeb LP anterior 0,024 ·
IDEB anterior 0,010 · socioeconômicas ≈ 0,001 cada.
**Decisão:** reportar a ordem que o dado impôs (UF dominante), com a ressalva
de colinearidade: UF e IDEB já carregam boa parte da informação socioeconômica,
então a importância marginal baixa de IDHM/pobreza **não** significa que não
importam — significa que são redundantes para o preditor, não para o
diagnóstico. Correlação municipal clara está na EDA.

## D15 — Threshold como decisão do gestor, não do algoritmo

**Decisão:** além do corte padrão 0,5, publicar a tabela de sensibilidade
(cortes 0,3–0,7) com recall e precision da classe "não alfabetizado". Para
triagem de risco, deixar de sinalizar uma criança custa mais que sinalizar
demais; a escolha do ponto é de política, e o material dá o cardápio.
