"""
Constrói a base analítica de modelagem (grão ALUNO) para a Fase 3.

Origem: camada Gold/Silver da Fase 2 (projeto `tech-challenge-alfabetizacao`
no BigQuery) + enriquecimento externo via Base dos Dados:

  - Censo Escolar (INEP): infraestrutura escolar agregada por município;
  - IBGE: população municipal e PIB municipal (per capita, ano anterior);
  - Atlas do Desenvolvimento Humano (2010): IDHM, renda, pobreza, saneamento;
  - IDEB (INEP): anos iniciais, rede pública, última edição ANTERIOR ao ano avaliado.

O resultado é materializado como `gold.gold_base_ml_alunos` (a base de ML nasce
na própria camada Gold, como pede o enunciado) e exportado em Parquet para data/.

Decisões anti-data-leakage (documentadas também no README):
  1. `proficiencia` NUNCA entra: o alvo `alfabetizado` é definido como
     proficiencia >= 743 — usar a nota seria prever o alvo com o próprio alvo.
  2. Apenas alunos PRESENTES: os ausentes chegam com alfabetizado = 0 de fábrica,
     mas isso é artefato de preenchimento, não medição (rótulo falso).
  3. Nenhum agregado de alfabetização do MESMO ano entra como feature: a taxa
     municipal de 2024 foi calculada com os mesmos alunos que estamos prevendo.
     Contexto educacional vem de fontes anteriores (IDEB <= ano-1, PIB ano-1,
     Atlas 2010) ou estruturais (Censo Escolar do ano corrente = infraestrutura
     conhecida no início do ano letivo, não desfecho).
  4. `id_municipio` e `id_escola` saem no arquivo como CHAVES DE GRUPO para a
     validação (GroupKFold) e para agregações estratégicas — não são features.

FinOps: toda query passa por dry-run antes de executar e o custo em MB é logado.

Uso:
    uv run python src/preprocessing/build_dataset.py
"""
from __future__ import annotations

import logging
import pathlib

from google.cloud import bigquery

PROJECT_ID = "tech-challenge-alfabetizacao"
LOCATION = "US"
DATA_DIR = pathlib.Path("data")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("build_dataset")

# ---------------------------------------------------------------------------
# SQL — base de modelagem no grão aluno, materializada na camada Gold
# ---------------------------------------------------------------------------
SQL_BASE_ML = f"""
create or replace table `{PROJECT_ID}.gold.gold_base_ml_alunos` as
with
-- Grão aluno: apenas presentes (ausente não tem medição real; o 0 é artefato).
-- proficiencia fica de fora por leakage direto (define o alvo no corte 743).
alunos as (
    select ano, id_municipio, codigo_uf, id_escola, rede, peso_aluno, alfabetizado
    from `{PROJECT_ID}.silver.silver_alunos`
    where presenca = 'Presente'
),

-- Porte da escola no ano (estrutural: tamanho da rede avaliada na escola).
porte_escola as (
    select ano, id_escola, count(*) as alunos_avaliados_escola
    from alunos
    group by 1, 2
),

-- Censo Escolar: infraestrutura das escolas EM ATIVIDADE ('1') com anos
-- iniciais do fundamental, agregada por município e ano. Infraestrutura é
-- atributo conhecido no início do ano letivo — não é desfecho da avaliação.
censo as (
    select
        ano,
        id_municipio,
        count(*)                                            as n_escolas_anos_iniciais,
        avg(cast(tipo_localizacao = '1' as int64))          as pct_escolas_urbanas,
        avg(cast(greatest(coalesce(biblioteca, 0),
                          coalesce(sala_leitura, 0)) as int64)) as pct_biblioteca_ou_sala_leitura,
        avg(internet)                                       as pct_internet,
        avg(internet_alunos)                                as pct_internet_alunos,
        avg(laboratorio_informatica)                        as pct_lab_informatica,
        avg(agua_potavel)                                   as pct_agua_potavel,
        avg(esgoto_rede_publica)                            as pct_esgoto_rede_publica,
        avg(energia_rede_publica)                           as pct_energia_rede_publica,
        avg(quadra_esportes)                                as pct_quadra_esportes,
        avg(alimentacao)                                    as pct_alimentacao,
        avg(quantidade_matricula_fundamental_anos_iniciais) as media_matriculas_ai_por_escola,
        -- quantidade_computador_aluno foi descontinuada no Censo recente (100% nula
        -- em 2022-2024); o flag equipamento_computador é o que existe de fato.
        avg(equipamento_computador)                         as pct_equipamento_computador
    from `basedosdados.br_inep_censo_escolar.escola`
    where ano in (2023, 2024)
      and tipo_situacao_funcionamento = '1'
      and etapa_ensino_fundamental_anos_iniciais = 1
    group by 1, 2
),

-- População municipal (ano corrente e ano anterior, para o PIB per capita).
pop as (
    select ano, id_municipio, populacao
    from `basedosdados.br_ibge_populacao.municipio`
    where ano between 2022 and 2024
),

-- PIB municipal do ANO ANTERIOR ao avaliado (2022 -> alunos 2023; 2023 -> 2024).
pib as (
    select ano, id_municipio, pib
    from `basedosdados.br_ibge_pib.municipio`
    where ano in (2022, 2023)
),

-- Atlas do Desenvolvimento Humano (Censo 2010): contexto socioeconômico
-- estrutural do município. Estático e anterior a qualquer avaliação.
adh as (
    select
        id_municipio,
        idhm, idhm_e, idhm_l, idhm_r,
        indice_gini,
        renda_pc,
        prop_pobreza,
        prop_pobreza_criancas,
        taxa_analfabetismo_15_mais,
        taxa_criancas_fora_escola_6_14,
        expectativa_anos_estudo,
        taxa_agua_encanada
    from `basedosdados.mundo_onu_adh.municipio`
    where ano = 2010
),

-- IDEB anos iniciais, rede pública, última edição ANTERIOR ao ano avaliado
-- (2021 -> alunos 2023; 2023 -> alunos 2024). Evita contaminação temporal.
ideb as (
    select ano, id_municipio, ideb, taxa_aprovacao, nota_saeb_lingua_portuguesa
    from `basedosdados.br_inep_ideb.municipio`
    where ano in (2021, 2023)
      and ensino = 'fundamental'
      and anos_escolares = 'iniciais (1-5)'
      and rede = 'publica'
)

select
    -- chaves e pesos (NÃO são features de modelo)
    a.ano,
    a.id_municipio,
    a.id_escola,
    a.peso_aluno,
    -- alvo
    a.alfabetizado,
    -- features do aluno / escola
    a.rede,
    pe.alunos_avaliados_escola,
    -- território
    case substr(a.id_municipio, 1, 1)
        when '1' then 'Norte'
        when '2' then 'Nordeste'
        when '3' then 'Sudeste'
        when '4' then 'Sul'
        when '5' then 'Centro-Oeste'
    end as regiao,
    case a.codigo_uf
        when '11' then 'RO' when '12' then 'AC' when '13' then 'AM'
        when '14' then 'RR' when '15' then 'PA' when '16' then 'AP'
        when '17' then 'TO' when '21' then 'MA' when '22' then 'PI'
        when '23' then 'CE' when '24' then 'RN' when '25' then 'PB'
        when '26' then 'PE' when '27' then 'AL' when '28' then 'SE'
        when '29' then 'BA' when '31' then 'MG' when '32' then 'ES'
        when '33' then 'RJ' when '35' then 'SP' when '41' then 'PR'
        when '42' then 'SC' when '43' then 'RS' when '50' then 'MS'
        when '51' then 'MT' when '52' then 'GO' when '53' then 'DF'
    end as sigla_uf,
    -- Censo Escolar (municipal, ano corrente)
    c.n_escolas_anos_iniciais,
    c.pct_escolas_urbanas,
    c.pct_biblioteca_ou_sala_leitura,
    c.pct_internet,
    c.pct_internet_alunos,
    c.pct_lab_informatica,
    c.pct_agua_potavel,
    c.pct_esgoto_rede_publica,
    c.pct_energia_rede_publica,
    c.pct_quadra_esportes,
    c.pct_alimentacao,
    c.media_matriculas_ai_por_escola,
    c.pct_equipamento_computador,
    -- IBGE
    pop_atual.populacao,
    safe_divide(pib.pib, pop_prev.populacao) as pib_per_capita_ano_anterior,
    -- Atlas 2010
    adh.idhm, adh.idhm_e, adh.idhm_l, adh.idhm_r,
    adh.indice_gini, adh.renda_pc,
    adh.prop_pobreza, adh.prop_pobreza_criancas,
    adh.taxa_analfabetismo_15_mais,
    adh.taxa_criancas_fora_escola_6_14,
    adh.expectativa_anos_estudo,
    adh.taxa_agua_encanada,
    -- IDEB (edição anterior)
    ideb.ideb                          as ideb_ai_publica_anterior,
    ideb.taxa_aprovacao                as taxa_aprovacao_ai_anterior,
    ideb.nota_saeb_lingua_portuguesa   as nota_saeb_lp_ai_anterior
from alunos a
left join porte_escola pe using (ano, id_escola)
left join censo c using (ano, id_municipio)
left join pop pop_atual
       on pop_atual.id_municipio = a.id_municipio and pop_atual.ano = a.ano
left join pop pop_prev
       on pop_prev.id_municipio = a.id_municipio and pop_prev.ano = a.ano - 1
left join pib
       on pib.id_municipio = a.id_municipio and pib.ano = a.ano - 1
left join adh
       on adh.id_municipio = a.id_municipio
left join ideb
       on ideb.id_municipio = a.id_municipio
      and ideb.ano = if(a.ano = 2023, 2021, 2023)
"""

# Tabelas Gold auxiliares exportadas para a aplicação estratégica (etapa 6).
GOLD_EXPORTS = [
    "gold_base_ml_alunos",
    "gold_indicador_municipio",
    "gold_metas_x_resultados",
    "gold_evolucao_temporal",
    "gold_taxa_municipio_microdados",
]

# Exports auxiliares por query (aplicação estratégica).
QUERY_EXPORTS = {
    # Nome oficial dos municípios (diretório da Base dos Dados) — para os
    # rankings de risco falarem "Município (UF)", não código IBGE.
    "dim_municipios": """
        select id_municipio, nome, sigla_uf
        from `basedosdados.br_bd_diretorios_brasil.municipio`
    """,
    # Metas municipais POR ANO (2024–2030). A Silver da Fase 2 manteve apenas a
    # meta de 2030; para comparar previsão × meta do ano seguinte precisamos da
    # trajetória anual completa, direto da Bronze.
    "metas_municipio_por_ano": f"""
        select
            id_municipio,
            rede,
            taxa_alfabetizacao      as taxa_base,
            meta_alfabetizacao_2024, meta_alfabetizacao_2025,
            meta_alfabetizacao_2026, meta_alfabetizacao_2027,
            meta_alfabetizacao_2028, meta_alfabetizacao_2029,
            meta_alfabetizacao_2030
        from `{PROJECT_ID}.bronze.meta_alfabetizacao_municipio`
        where ano = (select max(ano) from `{PROJECT_ID}.bronze.meta_alfabetizacao_municipio`)
    """,
}


def estimate_mb(client: bigquery.Client, sql: str, label: str) -> float:
    """Dry-run: estima os bytes processados antes de executar (FinOps)."""
    cfg = bigquery.QueryJobConfig(dry_run=True, use_query_cache=False)
    job = client.query(sql, job_config=cfg)
    mb = job.total_bytes_processed / 1e6
    log.info("[dry-run] %s: %.1f MB a processar", label, mb)
    return mb


def main() -> None:
    client = bigquery.Client(project=PROJECT_ID, location=LOCATION)
    DATA_DIR.mkdir(exist_ok=True)

    estimate_mb(client, SQL_BASE_ML, "gold_base_ml_alunos")
    client.query(SQL_BASE_ML).result()
    n = client.get_table(f"{PROJECT_ID}.gold.gold_base_ml_alunos").num_rows
    log.info("gold.gold_base_ml_alunos materializada: %d linhas", n)

    exports = {t: f"select * from `{PROJECT_ID}.gold.{t}`" for t in GOLD_EXPORTS}
    exports.update(QUERY_EXPORTS)
    for nome, sql in exports.items():
        estimate_mb(client, sql, f"export {nome}")
        df = client.query(sql).to_dataframe()
        out = DATA_DIR / f"{nome}.parquet"
        df.to_parquet(out, index=False)
        log.info("exportado %s: %d linhas -> %s", nome, len(df), out)

    log.info("Base analítica pronta em %s/", DATA_DIR)


if __name__ == "__main__":
    main()
