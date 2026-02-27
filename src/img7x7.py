#!/usr/bin/env python3
"""
img7x7.py — Demonstração didática: imagem como matriz numérica.

Cria uma imagem 7x7 com um único pixel preto no centro, salva em PNG e RAW,
imprime a matriz no terminal (valores e representação em bits) e opcionalmente
exibe a visualização com matplotlib. Útil para mostrar como o computador
"enxerga" uma imagem (array de números).

Uso:
    python img7x7.py
    python img7x7.py -o minha_imagem.png --no-show
    python img7x7.py --help
"""

import argparse
import numpy as np
from PIL import Image


def parse_argumentos():
    """Configura e analisa os argumentos de linha de comando."""
    parser = argparse.ArgumentParser(
        description="Cria imagem 7x7 com pixel central preto e exibe a matriz (didático)."
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default="pixel_central_7x7",
        help="Nome base dos arquivos de saída, sem extensão (default: pixel_central_7x7)."
    )
    parser.add_argument(
        "-t", "--tamanho",
        type=int,
        default=7,
        help="Lado da matriz quadrada em pixels (default: 7)."
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Não abrir a janela do matplotlib (apenas salvar e imprimir no terminal)."
    )
    return parser.parse_args()


def main():
    args = parse_argumentos()
    tamanho = args.tamanho
    nome_base = args.output
    centro = tamanho // 2

    dados = np.full((tamanho, tamanho), 255, dtype=np.uint8)
    dados[centro, centro] = 0

    Image.fromarray(dados, mode='L').save(f"{nome_base}.png")
    dados.tofile(f"{nome_base}.raw")

    print("-" * 40)
    print("VISÃO DO COMPUTADOR (MATRIZ NUMÉRICA)")
    print("-" * 40)
    for linha in dados:
        print(" ".join(f"{pixel:3}" for pixel in linha))
    print("-" * 40)
    print(f"Dimensões: {dados.shape}")
    print(f"Tipo de dado: {dados.dtype} (8 bits por pixel)")
    print(f"Pixel Central (Preto): Valor {dados[centro, centro]} em ({centro},{centro})")
    print("-" * 40)

    print("\nVISÃO EM BITS (BASE 2):")
    print("-" * 60)
    for linha in dados:
        print(" ".join(f"{pixel:08b}" for pixel in linha))
    print("-" * 60)

    import matplotlib.pyplot as plt
    plt.figure(figsize=(6, 5))
    plt.imshow(dados, cmap='gray', interpolation='nearest')
    plt.title(f"Visualização da Imagem {tamanho}x{tamanho}")
    plt.colorbar(label="Intensidade de Cinza (0-255)")
    plt.savefig(f"{nome_base}_preview.png", dpi=100, bbox_inches="tight")
    if not args.no_show:
        plt.show()
    else:
        plt.close()
        print(f"Preview salvo em '{nome_base}_preview.png' (--no-show ativo).")


if __name__ == "__main__":
    main()
