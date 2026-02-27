import os
import numpy as np
from PIL import Image
import random
import csv

# ==========================================
# 1. Configurações do Dataset
# ==========================================
num_imagens = 100
tamanho = 7
pasta_saida = "dataset_pontos"

# Cria a pasta para salvar as imagens, se não existir
if not os.path.exists(pasta_saida):
    os.makedirs(pasta_saida)

# Lista para guardar o "Gabarito" (Ground Truth)
anotacoes = [["arquivo", "posicao_x", "posicao_y", "intensidade"]]

print("-" * 50)
print(f"GERANDO DATASET DE {num_imagens} IMAGENS ({tamanho}x{tamanho})")
print("-" * 50)

# ==========================================
# 2. Gerando as Imagens e Variando Parâmetros
# ==========================================
for i in range(num_imagens):
    # a) Posição aleatória na matriz 7x7 (índices de 0 a 6)
    x = random.randint(0, tamanho - 1)
    y = random.randint(0, tamanho - 1)
    
    # b) Variação da cor (0 = Preto Absoluto, 120 = Cinza Escuro)
    # Valores próximos de 255 começariam a sumir no fundo branco
    intensidade = random.randint(0, 120)
    
    # c) Criando a matriz de fundo branco (255)
    dados = np.full((tamanho, tamanho), 255, dtype=np.uint8)
    
    # d) Inserindo o "ponto" (sinal) na posição sorteada com a cor sorteada
    dados[y, x] = intensidade
    
    # e) Salvando a imagem
    nome_arquivo = f"img_{i:03d}.png"
    caminho_arquivo = os.path.join(pasta_saida, nome_arquivo)
    Image.fromarray(dados, mode='L').save(caminho_arquivo)
    
    # f) Registrando na lista de anotações
    anotacoes.append([nome_arquivo, x, y, intensidade])

# ==========================================
# 3. Salvando o Arquivo de Labels (CSV)
# ==========================================
caminho_csv = os.path.join(pasta_saida, "labels.csv")
with open(caminho_csv, mode="w", newline="") as arquivo_csv:
    escritor = csv.writer(arquivo_csv)
    escritor.writerows(anotacoes)

print(f"Sucesso! {num_imagens} imagens salvas na pasta '{pasta_saida}'.")
print(f"Gabarito salvo em '{caminho_csv}'.")
print("-" * 50)
