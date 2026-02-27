# �� O Motor da Visão Computacional: Do Pixel à Rede Neural

Este diretório (`src`) contém uma série de *scripts* em Python criados para desmistificar a Inteligência Artificial e a Visão Computacional. O objetivo prático é demonstrar, do zero, como um computador enxerga uma imagem (como uma matriz numérica) e como treinamos uma Rede Neural Convolucional (CNN) para rastrear um objeto — neste caso, um ponto preto em uma matriz 7x7.

Ao invés de usar "caixas pretas" prontas do mercado, aqui nós construímos o algoritmo, geramos os dados, treinamos o modelo e fazemos a inferência.

---

## �� Dependências

Para rodar os *scripts* deste diretório, você precisará de um ambiente Python (recomendado o uso do `uv` ou `venv`) com as seguintes bibliotecas instaladas:

```bash
pip install numpy pillow torch matplotlib