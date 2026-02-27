#!/usr/bin/env python3
"""
png2raw.py — Converte imagem PNG para formato RAW (bytes brutos).

Lê uma imagem PNG, converte para escala de cinza (8 bits por pixel) e grava
os bytes da matriz num arquivo .raw (útil para inspeção em hex ou integração
com código que espera buffer bruto).

Uso:
    python png2raw.py -i imagem.png
    python png2raw.py --help
"""

import argparse
import numpy as np
from PIL import Image
import os

# ==========================================
# 1. Funções Modulares
# ==========================================
def parse_argumentos():
    """Configura e analisa os argumentos de linha de comando."""
    parser = argparse.ArgumentParser(description="Converte uma imagem PNG para formato RAW (bytes brutos).")
    parser.add_argument("-i", "--image", required=True, help="Caminho para a imagem PNG de entrada (ex: imagem.png)")
    return parser.parse_args()

def converter_png_para_raw(caminho_imagem):
    """Lê a imagem PNG, converte para matriz e salva os bytes brutos (.raw)."""
    # 1. Verifica se o arquivo existe
    if not os.path.exists(caminho_imagem):
        print(f"ERRO: O arquivo '{caminho_imagem}' não foi encontrado.")
        exit(1)

    print(f"Lendo a imagem: {caminho_imagem}")
    
    # 2. Carrega a imagem e força a conversão para escala de cinza (8 bits por pixel)
    img = Image.open(caminho_imagem).convert('L')
    
    # 3. Transforma em array do NumPy (uint8 = inteiro sem sinal de 8 bits)
    img_np = np.array(img, dtype=np.uint8)
    
    # 4. Cria o nome do arquivo de saída trocando a extensão original para .raw
    nome_base, _ = os.path.splitext(caminho_imagem)
    caminho_saida = f"{nome_base}.raw"
    
    # 5. Salva os dados brutos no disco
    img_np.tofile(caminho_saida)
    
    # 6. Exibe o resumo da operação
    print("-" * 50)
    print(f"Sucesso! Imagem convertida e salva como: '{caminho_saida}'")
    print(f"Dimensões da matriz: {img_np.shape}")
    print(f"Total de bytes gravados: {img_np.size} bytes")
    print("-" * 50)

# ==========================================
# 2. Fluxo Principal de Execução
# ==========================================
def main():
    # Lê os argumentos do terminal
    args = parse_argumentos()
    
    # Executa a conversão
    converter_png_para_raw(args.image)

if __name__ == "__main__":
    main()
