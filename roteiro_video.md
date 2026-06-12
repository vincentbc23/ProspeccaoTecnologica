# Roteiro de Apresentação em Vídeo
## XGBoost vs Random Forest: Uma Análise Comparativa em Problemas de Classificação

**Duração total alvo:** 12-14 minutos (dentro da faixa de 10-15 min exigida)
**Estrutura baseada no Anexo III** (Sugestão de Roteiro de Apresentação)

---

## Visão Geral dos Blocos

| Bloco | Tema | Tempo |
|-------|------|-------|
| 1 | O Gancho | 1,5 min |
| 2 | O "Core" Técnico | 4 min |
| 3 | Show me the Code! | 5 min |
| 4 | Veredito e Mercado | 2,5 min |
| 5 | Perguntas e Respostas | 1-2 min |

---

## Materiais Necessários (sem slides)

A ideia é gravar a tela usando **apenas os arquivos já existentes do projeto** — sem criar apresentação separada:

- `relatorio_XGBoost_vs_RandomForest.pdf` (17 páginas) — aberto em um leitor de PDF, navegando até a página/seção indicada em cada bloco
- `colunaVertebral/conlunaVertebral.ipynb` — dataset simples
- `Pokemon/pokemon.ipynb` — dataset médio
- `digitos/digitos.ipynb` — dataset complexo

Dica: deixe os 3 notebooks já **executados** (com os outputs visíveis) abertos em abas separadas do Jupyter antes de começar a gravação, e o PDF aberto em outra janela/aba — assim a transição entre "teoria" (PDF) e "código" (notebooks) é só alt-tab.

---

## BLOCO 1 — O Gancho (1,5 min)

### Abertura

**Visual:** Página 1 do PDF — capa do relatório, com o título e os dados do trabalho.

**Fala sugerida:**

> "Olá, pessoal! Hoje eu vou falar sobre dois dos algoritmos de machine learning mais usados no mundo para problemas de classificação: o **Random Forest** e o **XGBoost**.
>
> Esses dois algoritmos têm uma coisa em comum: os dois são *ensembles* — combinam várias árvores de decisão para fazer uma predição melhor. Mas a forma como cada um faz isso é **completamente oposta**, e essa diferença é o que define quando um vence o outro."

### O Problema

**Visual:** Página 1 do PDF, seção 1.3 "Problema que Resolve" — onde está a pergunta em destaque: *"Devo usar Random Forest ou XGBoost para este problema?"*

**Fala sugerida:**

> "Imaginem que vocês estão construindo um sistema de diagnóstico médico, ou um detector de fraude bancária. Escolher o algoritmo errado pode significar um modelo que erra justamente nos casos mais críticos — não detectar uma doença, ou deixar passar uma transação fraudulenta.
>
> A pergunta que guia este trabalho não é só 'qual dos dois é melhor?' — é **'em que condições, e *por quê*, um supera o outro?'**. E para responder isso de verdade, eu não fiz um, mas **três experimentos com complexidade crescente**: um problema simples, um médio e um complexo. Spoiler: o resultado não foi o que eu esperava."

---

## BLOCO 2 — O "Core" Técnico (4 min)

> Objetivo: explicar os 3 pilares da arquitetura sem se perder em detalhes. Os diagramas ASCII já estão prontos na página 3 do PDF (seção 2.3) — basta deixar a tela parada nessa página enquanto fala.

### Pilar 1 — Random Forest = Bagging (votação em paralelo)

**Visual:** Página 3 do PDF, seção 2.3 — diagrama do Random Forest:

```
Dados de Treino
      │
      ├─ Bootstrap Sample 1 → Árvore 1
      ├─ Bootstrap Sample 2 → Árvore 2
      ├─ ...
      └─ Bootstrap Sample N → Árvore N
              │
              ▼
     Votação Majoritária → Predição Final
```

**Fala sugerida:**

> "O Random Forest, proposto por Leo Breiman em 2001, funciona em **paralelo**. Ele pega o conjunto de treino, cria várias amostras aleatórias com reposição — chamadas *bootstrap samples* — e treina uma árvore de decisão independente em cada uma.
>
> Além disso, em cada divisão de nó, o algoritmo só considera um subconjunto aleatório das *features*. Isso força diversidade entre as árvores. No final, todas as árvores 'votam' em uma classe, e a maioria vence.
>
> A ideia central: **muitas árvores erradas de formas diferentes, em média, acertam mais que uma árvore só.**"

### Pilar 2 — XGBoost = Boosting (correção sequencial de erros)

**Visual:** Página 3-4 do PDF, seção 2.3 — diagrama do XGBoost (role um pouco a página):

```
Dados de Treino
      │
      ▼
Árvore 1 → Resíduos (Erros)
      │
      ▼
Árvore 2 (corrige erros da 1) → Novos Resíduos
      │
      ▼
      ...
      │
      ▼
Árvore N (corrige erros anteriores)
      │
      ▼
Soma Ponderada (learning_rate) → Predição Final
```

**Fala sugerida:**

> "Já o XGBoost, proposto por Chen e Guestrin em 2016 — completando 10 anos agora —, funciona de forma **sequencial**. A primeira árvore faz uma predição inicial e gera erros, chamados *resíduos*. A segunda árvore não tenta resolver o problema do zero: ela é treinada **especificamente para corrigir os erros da primeira**. E assim por diante.
>
> É a diferença entre um aluno que sempre estuda conteúdo novo sem revisar onde errou — bagging — e um aluno que refaz exatamente os exercícios em que errou — boosting."

### Pilar 3 — O diferencial: `learning_rate` e regularização nativa

**Visual:** Página 7 do PDF, seção 3.3 "Comparação de Hiperparâmetros" — tabela com os hiperparâmetros exclusivos do XGBoost.

**Fala sugerida:**

> "O grande diferencial do XGBoost é o **`learning_rate`** — a taxa de aprendizado. Ele controla o quanto cada nova árvore contribui para a predição final. Valores baixos, como 0.05, fazem o modelo aprender devagar e generalizar melhor — só que com mais árvores.
>
> O Random Forest **não tem esse conceito**: todas as árvores têm peso igual na votação. Em compensação, o XGBoost traz outras 'alavancas' que o RF não tem: `gamma` para podar árvores, `reg_alpha` e `reg_lambda` para regularização L1/L2, e `scale_pos_weight` para lidar com classes desbalanceadas.
>
> Resumindo o Bloco 2: **Random Forest = paralelo, robusto, poucas tunagens. XGBoost = sequencial, mais poderoso, mas mais sensível a ajustes.**"

---

## BLOCO 3 — Show me the Code! (5 min)

> Objetivo: demonstrar a PoC rodando nos 3 notebooks, destacar o código mais importante e ser honesto sobre as dificuldades. **Tenha os 3 notebooks já executados e abertos em abas** (Lei de Murphy — não confie em re-executar ao vivo).

### 3.1 A PoC — três datasets, complexidade crescente (1,5 min)

**Visual:** Página 9 do PDF, seções 4.1/4.2 (Ambiente de Desenvolvimento + Dataset 1), depois alternar rapidamente para as 3 abas dos notebooks abertos (`conlunaVertebral.ipynb`, `pokemon.ipynb`, `digitos.ipynb`) só para mostrar que os três existem e já têm output.

**Fala sugerida:**

> "Para testar essas teorias na prática, montei três notebooks com datasets de complexidade crescente:
>
> 1. **Coluna Vertebral** — simples: 310 amostras, 6 features numéricas, classificação binária.
> 2. **Pokémon Lendário** — médio: 800 amostras, classes severamente desbalanceadas — só 8% são lendários.
> 3. **Dígitos manuscritos** (`sklearn.load_digits`) — complexo: 1.797 amostras, 64 features (pixels), 10 classes.
>
> Em todos os três, segui o mesmo pipeline: `GridSearchCV` para tunar hiperparâmetros de ambos os modelos, treino, avaliação por F1 Score e Acurácia, e uma análise de overfitting comparando treino vs teste."

### 3.2 Destaque de código — GridSearchCV e a célula de overfitting (2 min)

**Visual:** Aba `digitos.ipynb` — célula 7 (a do `GridSearchCV` do XGBoost, logo após a do Random Forest). Role devagar mostrando as duas células de GridSearchCV uma em seguida da outra.

**Fala sugerida:**

> "Aqui está o trecho mais importante: a configuração do `GridSearchCV` para o XGBoost. Note os hiperparâmetros exclusivos que mencionei — `learning_rate`, `subsample`, `colsample_bytree`. No dataset de dígitos, por exemplo, o XGBoost precisou de configuração extra: `objective='multi:softprob'` e `num_class=10`, porque o problema tem 10 classes, não apenas duas."

```python
xgb_base = XGBClassifier(
    tree_method='hist',
    num_class=10,
    objective='multi:softprob',  # multiclasse nativo
    random_state=42,
    n_jobs=-1
)
```

> "E aqui está a célula que considero mais reveladora do trabalho todo: a **análise de overfitting**. Para cada modelo, eu comparo o F1 Score no treino versus no teste. Se a diferença for maior que 10%, é sinal de que o modelo está 'decorando' os dados de treino em vez de generalizar."

**Visual:** Trocar para a aba `pokemon.ipynb`, ir até a última célula (comentário `# --- Análise de Overfitting: Treino vs Teste ---`) e mostrar o output já executado — é o resultado mais dramático dos três notebooks:

```
====================================================
                         TREINO      TESTE
====================================================
      RF  — F1 Score     98,21%     83,33%
      XGB — F1 Score    100,00%     86,96%
====================================================
Diferença RF  (treino - teste) F1:  +14,88%
Diferença XGB (treino - teste) F1:  +13,04%
```

> "No Pokémon, o XGBoost chegou a **100% de F1 no treino** — memorizou perfeitamente os Pokémon lendários que viu. Isso é overfitting clássico em dataset pequeno e desbalanceado."

### 3.3 Resultados — e a virada inesperada (1 min)

**Visual:** Voltar ao PDF, página 16, seção 6.1 — tabela final com os 3 datasets e vencedores.

**Fala sugerida:**

> "Os resultados finais foram:
>
> - **Coluna Vertebral:** XGBoost venceu, 88,89% vs 84,78% — vantagem de +4,11 pontos.
> - **Pokémon:** XGBoost venceu de novo, 86,96% vs 83,33% — +3,63 pontos.
> - **Dígitos:** e aqui a surpresa — o **Random Forest venceu**, 96,92% vs 96,63%. Por uma margem pequena, mas venceu.
>
> Por que o RF ganhou justamente no problema mais complexo? Porque com 64 features (pixels) e 10 classes, o mecanismo de seleção aleatória de features do Random Forest se torna uma vantagem: ele dilui o peso de pixels pouco informativos, enquanto o XGBoost — trabalhando com gradientes globais — pode dar peso demais a pixels que parecem importantes só naquela iteração específica."

### 3.4 Desafios — sendo honesto (0,5 min)

**Fala sugerida:**

> "Sobre dificuldades: o maior desafio não foi rodar o código — foi **entender e justificar *por que*** um algoritmo vence o outro em cada cenário, em vez de só reportar números. Usei o Claude Code como assistente para estruturar os grids de hiperparâmetros e padronizar as métricas entre os dois modelos, mas toda a execução do `GridSearchCV` — que levou bastante tempo de processamento — e a interpretação dos resultados foram feitas e validadas por mim."

---

## BLOCO 4 — Veredito e Mercado (2,5 min)

### Quem usa

**Visual:** PDF, página 15, seção 5.2 "Adoção na Indústria" — lista de empresas.

**Fala sugerida:**

> "No mercado, o XGBoost é usado pela **Airbnb** para detecção de fraude em reservas, pela **Uber** para estimar tempo de chegada, por bancos como **JP Morgan** e **Goldman Sachs** para score de crédito, e pela **Netflix** e **Spotify** para recomendação e previsão de churn.
>
> Já o Random Forest continua forte em sistemas onde **interpretabilidade é crítica** — como diagnósticos médicos — e em pipelines de MLOps onde a simplicidade e a estabilidade pesam mais que um ganho marginal de performance."

### Vale a pena?

**Visual:** Voltar para a página 16 (seção 6.1) — mesma tabela-resumo da etapa 3.3, pode deixar na tela enquanto fala.

**Fala sugerida:**

> "Então, vale a pena usar XGBoost? **Sim, mas não é bala de prata.** Ele é o estado da arte para dados tabulares estruturados, especialmente com classes desbalanceadas — e por isso domina competições do Kaggle desde 2016.
>
> Mas o meu experimento com os dígitos prova, na prática, que **Random Forest continua sendo uma 'aposta segura'** em cenários de alta dimensionalidade, dados ruidosos, ou quando não há tempo para tunagem extensa.
>
> Minha conclusão: a pergunta certa não é 'qual algoritmo é melhor', e sim **'qual é o contexto do meu problema'** — natureza dos dados, balanceamento de classes, dimensionalidade e tempo disponível para tunagem definem a resposta."

---

## BLOCO 5 — Perguntas e Respostas (1-2 min)

**Visual:** Última página do PDF (seção 6.2 — Referências) ou o repositório GitHub aberto no navegador.

**Fala sugerida (encerramento):**

> "Esse foi meu trabalho comparando Random Forest e XGBoost em três cenários de complexidade crescente. O código completo dos três notebooks e o relatório estão no GitHub, no link na descrição. Fico à disposição para perguntas!"

### Perguntas que podem surgir (preparar respostas curtas)

| Pergunta provável | Resposta-base |
|---|---|
| "Por que o XGBoost domina tanto o Kaggle se o RF venceu no seu teste de dígitos?" | Kaggle é majoritariamente dados tabulares estruturados — o cenário ideal do XGBoost. Alta dimensionalidade tipo imagem é exceção, não regra, em dados tabulares. |
| "O Random Forest está obsoleto?" | Não — ele continua sendo o *baseline* mais confiável: poucas tunagens, robusto a outliers, paralelizável e mais interpretável. |
| "Qual métrica vocês usaram e por quê?" | F1 Score em todos os experimentos — para os dois datasets binários, `average='binary'` (foco na classe positiva); para dígitos (10 classes), `average='weighted'`, que é a forma correta de calcular F1 em problemas multiclasse. |
| "O que vocês fariam diferente com mais tempo?" | No Pokémon, aplicaria regularização mais forte (`gamma`, `reg_lambda`) ou oversampling (SMOTE) para reduzir o overfitting de 100% no treino do XGBoost. |

---

## Checklist Final (Dicas de Ouro do Anexo III)

- [ ] Executar os 3 notebooks com antecedência e deixar abertos com os outputs visíveis (não depender de live demo)
- [ ] Deixar o PDF aberto na página correta antes de cada bloco (evitar ficar rolando ao vivo)
- [ ] Cronometrar cada bloco — total entre 10 e 15 minutos
- [ ] Falar como se explicasse para um colega que nunca ouviu falar de Random Forest/XGBoost
- [ ] Focar em **decisões de projeto e trade-offs**, não em "telas bonitas"
- [ ] Ter o link do PDF e do GitHub prontos para a tela final
