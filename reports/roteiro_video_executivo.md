# Roteiro — Vídeo Executivo (até 5 minutos)

> Cenário simulado: reunião com gestores públicos de educação (secretaria
> estadual/municipal, MEC). Tom: executivo, direto, sem jargão técnico.
> Apoio visual sugerido entre parênteses em cada bloco (gráficos já estão em
> `images/`).

---

## Bloco 1 — O problema (0:00–0:45)

**Fala:**

"Bom dia. O Compromisso Nacional Criança Alfabetizada estabelece que, até 2030,
toda criança brasileira esteja alfabetizada ao fim do 2º ano. Hoje, 4 em cada 10
alunos avaliados ainda não chegam lá. E o dado oficial tem um problema prático:
ele chega **depois** do ano letivo, quando a janela de ação daquela turma já
fechou.

A pergunta que este projeto responde: dá para saber **antes** da avaliação onde
o risco vai se concentrar, e agir a tempo?"

*(Visual: `eda_uf.png` — o mapa da desigualdade por estado.)*

## Bloco 2 — O que construímos (0:45–1:45)

**Fala:**

"Construímos um modelo preditivo sobre os microdados oficiais do INEP: 3,3
milhões de alunos avaliados em 2023 e 2024, cruzados com o que já se sabe de
cada território antes da prova: o histórico educacional da rede, infraestrutura
das escolas, condições socioeconômicas do município.

O teste foi honesto: o modelo aprendeu só com 2023 e previu 2024, um ano que ele
nunca viu. E tomamos cuidado técnico rigoroso para o modelo não 'colar': nenhuma
informação do próprio resultado entra como pista.

O resultado que interessa à gestão: o ranking de municípios que o modelo previu
como críticos acompanhou fortemente o resultado real de 2024. Em números: quem
o modelo apontou como risco, de fato teve os piores resultados."

*(Visual: `estrategia_validacao_ranking.png` — previsto × real.)*

## Bloco 3 — Os três achados principais (1:45–3:15)

**Fala:**

"Três achados para decisão.

**Primeiro: o fator que mais pesa não é renda — é o estado.** A UF da criança
importa mais que o nível socioeconômico do município. Isso é uma boa notícia:
significa que política estadual estruturada de alfabetização muda o jogo, e que
replicar os arranjos dos estados que performam acima do esperado é a alavanca
mais promissora que os dados mostram.

**Segundo: o histórico recente da rede prediz o presente da criança.** Município
com IDEB e Saeb fracos na edição anterior tende a repetir o resultado, salvo
intervenção. Rede com histórico fraco precisa de apoio estrutural, não pontual.

**Terceiro: as metas de 2025 já nascem em risco em 59% dos municípios que
conseguimos avaliar.** 1.699 municípios têm desempenho previsto abaixo da meta
pactuada para 2025. Temos a lista nominal, por estado, com o tamanho do gap de
cada um."

*(Visuais: `interp_permutation.png`, `estrategia_metas_uf.png`.)*

## Bloco 4 — Como usar (3:15–4:15)

**Fala:**

"Este modelo não substitui a avaliação oficial — ele antecipa a prioridade.
Três usos imediatos:

Um: **focalização**. O ranking municipal de risco orienta onde concentrar
apoio técnico, formação de professores alfabetizadores e recomposição de
infraestrutura, ainda durante o ano letivo.

Dois: **alerta de metas**. A lista dos municípios abaixo da trajetória de 2025
permite renegociar apoio antes do resultado ruim, não depois.

Três: **desenho de política**. Os dados agrupam os municípios em três famílias
com perfis muito diferentes; a família crítica, concentrada no Norte e
Nordeste, pede ação socioeducacional combinada, não apenas educacional."

*(Visual: `estrategia_clusters.png` e `estrategia_risco_municipal.png`.)*

## Bloco 5 — Fecho (4:15–4:50)

**Fala:**

"Em resumo: com dado público e método aberto, é possível transformar a
avaliação de alfabetização de um retrato tardio em um instrumento de gestão
antecipada. O código, as listas por município e toda a documentação estão no
repositório do projeto, prontos para uso e auditoria.

A meta de 2030 é de todos. Saber onde ela está em risco, hoje, é o primeiro
passo para cumpri-la. Obrigado."

---

## Checklist de gravação

- [ ] Até 5 minutos (roteiro calibrado para ~4:50 em ritmo de fala normal);
- [ ] Tela: alternar entre apresentador e os gráficos indicados (ou slides com
      os PNGs de `images/`);
- [ ] Sem jargão: não dizer "AUC", "SHAP", "LightGBM" — dizer "acerto do
      ranking", "fatores que mais pesam", "modelo";
- [ ] Subir o vídeo (YouTube não listado ou drive da pós) e adicionar o link no
      README.
