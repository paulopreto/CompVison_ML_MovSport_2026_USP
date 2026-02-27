#!/usr/bin/env python3
"""
trainblack7x7.py — Treinamento da CNN para rastrear o ponto preto em imagens 7x7.

Gera dados sintéticos on-the-fly, treina uma rede convolucional (RastreadorDePontoCNN)
para regredir as coordenadas (x, y) do ponto e salva os pesos em arquivo .pth.
O modelo treinado é usado pelo script predictblackdot.py para inferência.

Uso:
    python trainblack7x7.py
    python trainblack7x7.py -e 20 -o modelo_custom.pth
    python trainblack7x7.py --help
"""

import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


class PontoDataset(Dataset):
    def __init__(self, num_amostras, tamanho=7):
        self.num_amostras = num_amostras
        self.tamanho = tamanho

    def __len__(self):
        return self.num_amostras

    def __getitem__(self, idx):
        img = np.full((1, self.tamanho, self.tamanho), 255.0, dtype=np.float32)
        x = random.randint(0, self.tamanho - 1)
        y = random.randint(0, self.tamanho - 1)
        intensidade = random.randint(0, 120)
        img[0, y, x] = float(intensidade)
        img = img / 255.0
        coordenadas = np.array([float(x), float(y)], dtype=np.float32)
        return torch.tensor(img), torch.tensor(coordenadas)


class RastreadorDePontoCNN(nn.Module):
    def __init__(self, tamanho=7):
        super(RastreadorDePontoCNN, self).__init__()
        self.tamanho = tamanho
        self.convolucao = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.densa = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * tamanho * tamanho, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )

    def forward(self, x):
        mapa_de_features = self.convolucao(x)
        return self.densa(mapa_de_features)


def parse_argumentos():
    """Configura e analisa os argumentos de linha de comando."""
    parser = argparse.ArgumentParser(
        description="Treina CNN para rastrear coordenadas do ponto preto em imagens 7x7."
    )
    parser.add_argument(
        "-n", "--num-amostras",
        type=int,
        default=5000,
        help="Número de amostras sintéticas por época (default: 5000)."
    )
    parser.add_argument(
        "-e", "--epocas",
        type=int,
        default=15,
        help="Número de épocas de treinamento (default: 15)."
    )
    parser.add_argument(
        "-b", "--batch-size",
        type=int,
        default=32,
        help="Tamanho do batch (default: 32)."
    )
    parser.add_argument(
        "-l", "--lr",
        type=float,
        default=0.005,
        help="Taxa de aprendizado (default: 0.005)."
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default="modelo_rastreador.pth",
        help="Caminho do arquivo de saída do modelo (.pth) (default: modelo_rastreador.pth)."
    )
    parser.add_argument(
        "-t", "--tamanho",
        type=int,
        default=7,
        help="Lado da matriz da imagem (default: 7)."
    )
    return parser.parse_args()


def main():
    args = parse_argumentos()
    device = torch.device("cpu")

    dataset_treino = PontoDataset(num_amostras=args.num_amostras, tamanho=args.tamanho)
    dataloader = DataLoader(dataset_treino, batch_size=args.batch_size, shuffle=True)

    modelo = RastreadorDePontoCNN(tamanho=args.tamanho).to(device)
    criterio_erro = nn.MSELoss()
    otimizador = optim.Adam(modelo.parameters(), lr=args.lr)

    print("Iniciando o Treinamento do Modelo (CPU)...\n")

    for epoca in range(args.epocas):
        erro_total = 0.0
        for imagens, coordenadas_reais in dataloader:
            imagens = imagens.to(device)
            coordenadas_reais = coordenadas_reais.to(device)
            previsoes = modelo(imagens)
            erro = criterio_erro(previsoes, coordenadas_reais)
            otimizador.zero_grad()
            erro.backward()
            otimizador.step()
            erro_total += erro.item()
        erro_medio = erro_total / len(dataloader)
        print(f"Época [{epoca+1}/{args.epocas}] | Erro Médio (MSE): {erro_medio:.4f}")

    print("\n--- TESTE DE RASTREAMENTO (INFERENCE) ---")
    modelo.eval()
    dataset_teste = PontoDataset(num_amostras=5, tamanho=args.tamanho)
    with torch.no_grad():
        for i in range(5):
            img_teste, coord_real = dataset_teste[i]
            img_input = img_teste.unsqueeze(0).to(device)
            coord_predita = modelo(img_input).squeeze().numpy()
            coord_real_np = coord_real.numpy()
            print(f"Teste {i+1}:")
            print(f"  Real     (x, y): ({coord_real_np[0]:.0f}, {coord_real_np[1]:.0f})")
            print(f"  Previsão (x, y): ({coord_predita[0]:.2f}, {coord_predita[1]:.2f})\n")

    torch.save(modelo.state_dict(), args.output)
    print(f"Modelo salvo com sucesso em '{args.output}'!")


if __name__ == "__main__":
    main()
