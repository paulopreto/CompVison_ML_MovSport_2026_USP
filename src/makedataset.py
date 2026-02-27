#!/usr/bin/env python3
"""
makedataset.py — Geração de dataset sintético de imagens 7x7 com ponto preto.

Gera N imagens em escala de cinza (matriz 7x7), cada uma com um único "ponto"
em posição e intensidade aleatórias, e salva o gabarito (ground truth) em CSV.
Usado como combustível para treinar a CNN de rastreamento do ponto.

Uso:
    python makedataset.py
    python makedataset.py -n 200 -s dataset_custom
    python makedataset.py --help
"""

import argparse
import os
import random
import csv
import numpy as np
from PIL import Image


def parse_argumentos():
    """Configura e analisa os argumentos de linha de comando."""
    parser = argparse.ArgumentParser(
        description="Gera dataset sintético de imagens 7x7 com ponto preto e salva labels em CSV."
    )
    parser.add_argument(
        "-n", "--num-imagens",
        type=int,
        default=100,
        help="Número de imagens a gerar (default: 100)."
    )
    parser.add_argument(
        "-s", "--pasta-saida",
        type=str,
        default="dataset_pontos",
        help="Pasta onde salvar as imagens e o CSV de labels (default: dataset_pontos)."
    )
    parser.add_argument(
        "-t", "--tamanho",
        type=int,
        default=7,
        help="Lado da matriz quadrada em pixels (default: 7)."
    )
    return parser.parse_args()


def main():
    args = parse_argumentos()
    num_imagens = args.num_imagens
    tamanho = args.tamanho
    pasta_saida = args.pasta_saida

    if not os.path.exists(pasta_saida):
        os.makedirs(pasta_saida)

    anotacoes = [["arquivo", "posicao_x", "posicao_y", "intensidade"]]

    print("-" * 50)
    print(f"GERANDO DATASET DE {num_imagens} IMAGENS ({tamanho}x{tamanho})")
    print("-" * 50)

    for i in range(num_imagens):
        x = random.randint(0, tamanho - 1)
        y = random.randint(0, tamanho - 1)
        intensidade = random.randint(0, 120)
        dados = np.full((tamanho, tamanho), 255, dtype=np.uint8)
        dados[y, x] = intensidade

        nome_arquivo = f"img_{i:03d}.png"
        caminho_arquivo = os.path.join(pasta_saida, nome_arquivo)
        Image.fromarray(dados, mode='L').save(caminho_arquivo)
        anotacoes.append([nome_arquivo, x, y, intensidade])

    caminho_csv = os.path.join(pasta_saida, "labels.csv")
    with open(caminho_csv, mode="w", newline="") as arquivo_csv:
        escritor = csv.writer(arquivo_csv)
        escritor.writerows(anotacoes)

    print(f"Sucesso! {num_imagens} imagens salvas na pasta '{pasta_saida}'.")
    print(f"Gabarito salvo em '{caminho_csv}'.")
    print("-" * 50)


if __name__ == "__main__":
    main()
