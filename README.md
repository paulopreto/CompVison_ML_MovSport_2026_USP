# Visão Computacional e Deep Learning na Análise do Movimento de Atletas

**Perspectivas para a análise de desempenho** — Apresentação oficial para o IV Simpósio de Tecnologia Aplicada à Análise de Desempenho Esportivo (27 e 28 de Fevereiro de 2026).

Este repositório contém o código-fonte em LaTeX da apresentação, os arquivos de estilo customizados, as imagens utilizadas e um diretório com scripts práticos (`src/`) desenvolvidos para demonstrar o "motor" por trás da inteligência artificial.

---

## A Filosofia (Provocação Metodológica)

O mercado esportivo brasileiro é historicamente dependente de tecnologias importadas (EUA, Europa, Austrália). Consumimos hardwares caríssimos e softwares fechados (*Black Boxes*).

A provocação central desta palestra é: **Como vamos discutir a complexidade da análise de desempenho se não sabemos como funciona o motor do algoritmo?** Este projeto convida pesquisadores e profissionais do esporte a deixarem de ser apenas "apertadores de botões" (usuários *premium*) para se tornarem **desenvolvedores** e validadores de suas próprias soluções metodológicas, entendendo a ciência desde o nível do pixel.

---

## Estrutura da Apresentação

O arquivo `main.tex` está dividido em quatro blocos narrativos:

1. **O Contexto e a Dor do Esporte:** A soberania tecnológica, a ilusão do domínio das "caixas pretas" e o rigor de encontrar o verdadeiro sinal no meio do ruído (O Pálido Ponto Azul).
2. **O Arsenal Tecnológico:** Como as Redes Neurais Profundas viabilizaram a *Markerless Motion Capture*. O uso de arquiteturas como YOLO (detecção) e MediaPipe (estimativa de pose). Inclui também slides que explicam a rede do repositório (RastreadorDePontoCNN) com o diagrama gerado pelo script `visualizar_rede.py`.
3. **Mão na Massa — Validações do LaBioCoM:** Resultados científicos práticos provando que a tecnologia aberta funciona no cenário real (*in the wild*), incluindo futebol de campo, desportos de combate, movimentos complexos de força e a apresentação do **vailá Multimodal Toolbox**.
4. **A Fronteira do Conhecimento e Perspectivas:** A provocação do VAR Semiautomático via smartphones, a fusão de Visão Computacional com dispositivos IoT, simulações para identificação de talentos (*safe deselection*) e a mensagem final.

---

## Como Compilar os Slides (LaTeX)

A apresentação foi construída utilizando a classe `beamer` com o tema visual *Warsaw* e um pacote de estilo customizado (`preto.sty`).

### Pré-requisitos

- Uma distribuição LaTeX instalada (TeX Live, MiKTeX, MacTeX) **ou** uma conta no [Overleaf](https://www.overleaf.com/).

### Compilação local

Para compilar o PDF com as referências bibliográficas corretas, execute a seguinte sequência no seu terminal:

```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

### Arquivos importantes

- **`main.tex`** — O arquivo principal com o conteúdo dos slides.
- **`preto.sty`** — O pacote de estilos com as configurações de rodapé, formatação de código Python e cores personalizadas do LaBioCoM.
- **`references.bib`** — O banco de referências (BibTeX) com os artigos citados durante a palestra.
- **`images/`** — Diretório contendo figuras, GIFs e logomarcas. Algumas figuras (como o diagrama da rede) vêm de **`src/`** (ex.: `src/diagrama_rastreador_rede.png`), geradas pelos scripts Python.
- **`utils/`** — Contém versões já compiladas do PDF da apresentação.

A apresentação cita e utiliza os scripts de `src/` (treino, inferência e visualização da rede); os slides da CNN mostram o diagrama produzido por `visualizar_rede.py`.

---

## Entendendo o Motor na Prática (`src/`)

Para provar que é possível construir essa tecnologia do zero, este repositório conta com a pasta `src/`. Lá você encontrará scripts em Python (usando PyTorch) que:

- Geram um dataset sintético de imagens.
- Constroem e treinam uma Rede Neural Convolucional (CNN) simples (RastreadorDePontoCNN).
- Fazem a inferência para rastrear as coordenadas de um objeto na imagem.
- **Visualizam a rede de forma didática:** o script `visualizar_rede.py` exibe a arquitetura em texto (camadas, pesos) e gera um diagrama PNG do fluxo das camadas; esse diagrama é usado nos slides da apresentação para explicar a CNN em aula.

**Leia o [README da pasta `src/`](src/README.md) para entender o código passo a passo e os comandos de cada script.**

---

## Sobre o Autor

**Prof. Dr. Paulo Roberto Pereira Santiago** — Professor e Pesquisador, Especialista na Escola de Educação Física e Esporte de Ribeirão Preto (EEFERP) — Universidade de São Paulo (USP).

Coordenador do Laboratório de Biomecânica e Controle Motor (LaBioCoM).

- [GitHub pessoal](https://github.com/paulopreto)
- [Repositório desta apresentação](https://github.com/paulopreto/CompVison_ML_MovSport_2026_USP)

---

> "A inteligência artificial e a visão computacional não substituem o treinador; elas fornecem a lente de precisão para a sua intuição e experiência."
