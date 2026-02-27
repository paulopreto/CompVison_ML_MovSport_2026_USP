import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random

# ==========================================
# 1. Criação do Dataset (O "Combustível" do Motor)
# ==========================================
class PontoDataset(Dataset):
    def __init__(self, num_amostras, tamanho=7):
        self.num_amostras = num_amostras
        self.tamanho = tamanho

    def __len__(self):
        return self.num_amostras

    def __getitem__(self, idx):
        # Cria imagem branca (1 canal de cor, 7x7)
        img = np.full((1, self.tamanho, self.tamanho), 255.0, dtype=np.float32)
        
        # Sorteia a posição (x, y) e a intensidade (cinza escuro a preto)
        x = random.randint(0, self.tamanho - 1)
        y = random.randint(0, self.tamanho - 1)
        intensidade = random.randint(0, 120)
        
        # Desenha o ponto
        img[0, y, x] = float(intensidade)
        
        # Normaliza a imagem para o intervalo [0, 1] (facilita o aprendizado)
        img = img / 255.0
        
        # O "Gabarito" (Ground Truth) que o modelo deve adivinhar
        coordenadas = np.array([float(x), float(y)], dtype=np.float32)
        
        return torch.tensor(img), torch.tensor(coordenadas)

# ==========================================
# 2. Arquitetura da Rede Neural (O "Motor")
# ==========================================
class RastreadorDePontoCNN(nn.Module):
    def __init__(self):
        super(RastreadorDePontoCNN, self).__init__()
        # Camadas de Convolução (Extração de padrões visuais)
        self.convolucao = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU()
        )
        # Camadas Densas (Decisão/Regressão para achar o x e y)
        self.densa = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 7 * 7, 32),
            nn.ReLU(),
            nn.Linear(32, 2) # Saída de 2 neurônios: coordenada X e coordenada Y
        )

    def forward(self, x):
        mapa_de_features = self.convolucao(x)
        coordenadas_preditas = self.densa(mapa_de_features)
        return coordenadas_preditas

# ==========================================
# 3. Configurações de Treinamento
# ==========================================
print("Iniciando o Treinamento do Modelo (CPU)...\n")

# Dispositivo
device = torch.device("cpu")

# Preparando os dados
dataset_treino = PontoDataset(num_amostras=5000, tamanho=7)
dataloader = DataLoader(dataset_treino, batch_size=32, shuffle=True)

# Instanciando o modelo, a função de erro (Loss) e o otimizador
modelo = RastreadorDePontoCNN().to(device)
criterio_erro = nn.MSELoss() # Erro Quadrático Médio (distância entre o real e a previsão)
otimizador = optim.Adam(modelo.parameters(), lr=0.005)

# ==========================================
# 4. O Loop de Treinamento (Deep Learning na Prática)
# ==========================================
epocas = 15

for epoca in range(epocas):
    erro_total = 0.0
    for imagens, coordenadas_reais in dataloader:
        imagens = imagens.to(device)
        coordenadas_reais = coordenadas_reais.to(device)
        
        # Passo 1: Previsão do modelo (Forward)
        previsoes = modelo(imagens)
        
        # Passo 2: Calcular o erro (Loss)
        erro = criterio_erro(previsoes, coordenadas_reais)
        
        # Passo 3: Ajustar os pesos (Backward + Optimizer Step)
        otimizador.zero_grad()
        erro.backward()
        otimizador.step()
        
        erro_total += erro.item()
        
    erro_medio = erro_total / len(dataloader)
    print(f"Época [{epoca+1}/{epocas}] | Erro Médio (MSE): {erro_medio:.4f}")

# ==========================================
# 5. Testando o Modelo com Novos Dados (Inference)
# ==========================================
print("\n--- TESTE DE RASTREAMENTO (INFERENCE) ---")
modelo.eval() # Coloca o modelo em modo de avaliação

dataset_teste = PontoDataset(num_amostras=5, tamanho=7)
with torch.no_grad(): # Desliga o cálculo de gradientes para economizar memória
    for i in range(5):
        img_teste, coord_real = dataset_teste[i]
        
        # Adiciona a dimensão do batch (1, 1, 7, 7) e envia para CPU
        img_input = img_teste.unsqueeze(0).to(device) 
        
        # O modelo adivinha onde está o ponto
        coord_predita = modelo(img_input).squeeze().numpy()
        coord_real = coord_real.numpy()
        
        print(f"Teste {i+1}:")
        print(f"  Real    (x, y): ({coord_real[0]:.0f}, {coord_real[1]:.0f})")
        print(f"  Previsão (x, y): ({coord_predita[0]:.2f}, {coord_predita[1]:.2f})\n")
# ==========================================
# 6. Salvando o Modelo Treinado
# ==========================================
torch.save(modelo.state_dict(), "modelo_rastreador.pth")
print("\nModelo salvo com sucesso em 'modelo_rastreador.pth'!")
