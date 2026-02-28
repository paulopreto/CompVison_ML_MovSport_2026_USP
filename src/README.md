# O Motor da Visão Computacional: Do Pixel à Rede Neural

Este diretório (`src`) contém uma série de *scripts* em Python criados para desmistificar a Inteligência Artificial e a Visão Computacional. O objetivo prático é demonstrar, do zero, como um computador enxerga uma imagem (como uma matriz numérica) e como treinamos uma Rede Neural Convolucional (CNN) para rastrear um objeto — neste caso, um ponto preto em uma matriz 7x7.

Ao invés de usar "caixas pretas" prontas do mercado, aqui nós construímos o algoritmo, geramos os dados, treinamos o modelo e fazemos a inferência.

---

## Dependências

Na **raiz do repositório** existem `pyproject.toml` e `requirements.txt`. Instale as dependências de uma das formas:

```bash
# Opção 1: requirements.txt (a partir da raiz do projeto)
pip install -r requirements.txt

# Opção 2: com uv (a partir da raiz)
uv pip install -r requirements.txt

# Opção 3: ambiente virtual recomendado
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
# .venv\Scripts\activate   # Windows
pip install -r requirements.txt
```

Pacotes necessários: `numpy`, `pillow`, `torch`, `matplotlib`.

---

## Scripts e como rodar

Todos os scripts aceitam `-h` ou `--help` para exibir a ajuda na linha de comando.

| Script | Descrição |
|--------|-----------|
| `img7x7.py` | Cria imagem 7x7 com pixel central preto; mostra matriz no terminal e opcionalmente gráfico. |
| `makedataset.py` | Gera dataset sintético (N imagens 7x7 + CSV de labels). |
| `trainblack7x7.py` | Treina a CNN para regredir (x, y) do ponto e salva o modelo `.pth`. |
| `predictblackdot.py` | Carrega o modelo e uma imagem e prediz as coordenadas do ponto. |
| `visualizar_rede.py` | Visualização didática da rede: resumo em texto (camadas, pesos) e diagrama PNG para uso em aula. |
| `png2raw.py` | Converte uma imagem PNG para arquivo RAW (bytes brutos). |

### 1. Demonstração didática (matriz e bits)

```bash
cd src
python img7x7.py
python img7x7.py --help
python img7x7.py -o minha_imagem.png --no-show
```

### 2. Gerar dataset sintético

```bash
cd src
python makedataset.py
python makedataset.py -n 200 -s dataset_custom
python makedataset.py --help
```

### 3. Treinar a CNN

```bash
cd src
python trainblack7x7.py
python trainblack7x7.py -e 20 -o modelo_custom.pth
python trainblack7x7.py --help
```

### 4. Inferência (prever o ponto em uma imagem)

É necessário ter um modelo treinado (por exemplo `modelo_rastreador.pth`) e uma imagem PNG 7x7 em escala de cinza:

```bash
cd src
python predictblackdot.py -i dataset_pontos/img_000.png -n modelo_rastreador.pth
python predictblackdot.py --help
```

### 5. Visualizar a rede (uso em aula)

Para entender a arquitetura e os pesos do modelo `.pth` de forma legível (texto + diagrama):

```bash
cd src
python visualizar_rede.py -m modelo_rastreador.pth
python visualizar_rede.py -m modelo_rastreador.pth -o diagrama_rede.png
python visualizar_rede.py --sem-diagrama
python visualizar_rede.py --export-onnx modelo.onnx
python visualizar_rede.py --help
```

- **Resumo em texto**: estrutura da rede (camadas e fluxo), nomes e formatos dos tensores no `state_dict`, total de parâmetros.
- **Diagrama PNG**: fluxo visual das camadas (Entrada → Conv → ReLU → … → Saída) para projetar em aula.
- **Export ONNX**: opcional; use `--export-onnx modelo.onnx` e abra o arquivo em [netron.app](https://netron.app) para uma visão interativa.

### 6. Converter PNG para RAW

```bash
cd src
python png2raw.py -i pixel_central_7x7.png
python png2raw.py --help
```

---

## Fluxo completo (do zero à inferência)

```bash
cd src
pip install -r ../requirements.txt   # se ainda não instalou

# 1) Ver a imagem como matriz (didático)
python img7x7.py --no-show

# 2) Gerar dataset (opcional; o treino usa dados sintéticos on-the-fly)
python makedataset.py -n 100

# 3) Treinar o modelo
python trainblack7x7.py -e 15 -o modelo_rastreador.pth

# 4) Prever em uma imagem (ex.: uma gerada pelo makedataset)
python predictblackdot.py -i dataset_pontos/img_000.png -n modelo_rastreador.pth
```

---

## Estrutura de arquivos gerados

- `img7x7.py` → `pixel_central_7x7.png`, `pixel_central_7x7.raw`, `*_preview.png`
- `makedataset.py` → pasta `dataset_pontos/` (ou a que você indicar) com `img_*.png` e `labels.csv`
- `trainblack7x7.py` → `modelo_rastreador.pth` (ou o nome indicado em `-o`)
- `predictblackdot.py` → atualiza/gera `resultados_rastreamento.csv`
