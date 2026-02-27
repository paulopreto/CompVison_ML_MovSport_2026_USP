#!/usr/bin/env python3
"""
predictblackdot.py — Inferência da coordenada do ponto preto via CNN treinada.

Carrega um modelo .pth (treinado com trainblack7x7.py), lê uma imagem PNG 7x7
em escala de cinza, normaliza e passa pela rede; exibe e opcionalmente salva
as coordenadas (x, y) preditas e em pixel inteiro.

Uso:
    python predictblackdot.py -i img_001.png -n modelo_rastreador.pth
    python predictblackdot.py --help
"""

import argparse
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import csv
import os

# ==========================================
# 1. Definição da Arquitetura do Modelo
# ==========================================
class RastreadorDePontoCNN(nn.Module):
    def __init__(self):
        super(RastreadorDePontoCNN, self).__init__()
        self.convolucao = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.densa = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 7 * 7, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )

    def forward(self, x):
        mapa_de_features = self.convolucao(x)
        return self.densa(mapa_de_features)

# ==========================================
# 2. Funções Modulares
# ==========================================
def parse_argumentos():
    """Configura e analisa os argumentos de linha de comando."""
    parser = argparse.ArgumentParser(description="Inferência da coordenada do ponto preto via CNN.")
    parser.add_argument("-i", "--image", required=True, help="Caminho para a imagem PNG de entrada.")
    parser.add_argument("-n", "--network", required=True, help="Caminho para o modelo treinado (.pth).")
    return parser.parse_args()

def carregar_modelo(caminho_pesos):
    """Instancia a rede e carrega os pesos salvos para a CPU."""
    print(f"Carregando a rede: {caminho_pesos}")
    modelo = RastreadorDePontoCNN()
    modelo.load_state_dict(torch.load(caminho_pesos, weights_only=True, map_location=torch.device('cpu')))
    modelo.eval()
    return modelo

def pre_processar_imagem(caminho_imagem):
    """Carrega a imagem, converte para escala de cinza, normaliza e transforma em Tensor."""
    print(f"Lendo a imagem: {caminho_imagem}")
    try:
        img = Image.open(caminho_imagem).convert('L')
    except FileNotFoundError:
        print(f"ERRO: A imagem '{caminho_imagem}' não foi encontrada.")
        exit(1)

    img_np = np.array(img, dtype=np.float32) / 255.0
    img_tensor = torch.tensor(img_np).unsqueeze(0).unsqueeze(0)
    return img_tensor

def realizar_inferencia(modelo, img_tensor):
    """Passa o tensor pelo modelo e retorna as coordenadas (float e int)."""
    with torch.no_grad():
        coord_predita = modelo(img_tensor).squeeze().numpy()

    x_pred, y_pred = coord_predita[0], coord_predita[1]
    x_pixel = round(float(x_pred))
    y_pixel = round(float(y_pred))
    
    return x_pred, y_pred, x_pixel, y_pixel

def salvar_resultados_csv(caminho_imagem, caminho_rede, x_pred, y_pred, x_pixel, y_pixel, arquivo_saida="resultados_rastreamento.csv"):
    """Salva os resultados da inferência em um arquivo CSV, criando o cabeçalho se necessário."""
    cabecalho = ["arquivo_imagem", "modelo_rede", "x_predito", "y_predito", "x_pixel", "y_pixel"]
    dados = [caminho_imagem, caminho_rede, round(float(x_pred), 4), round(float(y_pred), 4), x_pixel, y_pixel]

    escrever_cabecalho = not os.path.exists(arquivo_saida)

    with open(arquivo_saida, mode="a", newline="") as f:
        escritor = csv.writer(f)
        if escrever_cabecalho:
            escritor.writerow(cabecalho)
        escritor.writerow(dados)

    print(f"Resultado adicionado com sucesso em '{arquivo_saida}'.")

# ==========================================
# 3. Fluxo Principal de Execução
# ==========================================
def main():
    # 1. Lê os argumentos do terminal
    args = parse_argumentos()
    
    # 2. Carrega o modelo treinado
    modelo = carregar_modelo(args.network)
    
    # 3. Prepara a imagem
    img_tensor = pre_processar_imagem(args.image)
    
    # 4. Faz a previsão
    x_pred, y_pred, x_pixel, y_pixel = realizar_inferencia(modelo, img_tensor)
    
    # 5. Exibe os resultados na tela
    print("-" * 50)
    print(f"Detecção Bruta (Float) : X={x_pred:.4f}, Y={y_pred:.4f}")
    print(f"Detecção Final (Pixel) : X={x_pixel}, Y={y_pixel}")
    print("-" * 50)
    
    # 6. Salva no disco
    salvar_resultados_csv(args.image, args.network, x_pred, y_pred, x_pixel, y_pixel)

if __name__ == "__main__":
    main()
