import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# 1. Configurações
tamanho = 7
centro = tamanho // 2

# 2. Criando a matriz (255 = Branco, 0 = Preto)
dados = np.full((tamanho, tamanho), 255, dtype=np.uint8)
dados[centro, centro] = 0

# 3. Salvando os arquivos
Image.fromarray(dados, mode='L').save('pixel_central_7x7.png')
dados.tofile('pixel_central_7x7.raw')

# --- PARTE DIDÁTICA NO TERMINAL ---
print("-" * 40)
print("VISÃO DO COMPUTADOR (MATRIZ NUMÉRICA)")
print("-" * 40)

# Imprimindo a matriz de forma organizada
for linha in dados:
    # Formata cada número com 3 espaços para manter o alinhamento
    print(" ".join(f"{pixel:3}" for pixel in linha))

print("-" * 40)
print(f"Dimensões: {dados.shape}")
print(f"Tipo de dado: {dados.dtype} (8 bits por pixel)")
print(f"Pixel Central (Preto): Valor {dados[centro, centro]} em ({centro},{centro})")
print("-" * 40)

# No final do seu img7x7.py
print("\nVISÃO EM BITS (BASE 2):")
print("-" * 60)
for linha in dados:
    # Mostra o valor binário formatado com 8 dígitos para cada pixel
    print(" ".join(f"{pixel:08b}" for pixel in linha))
print("-" * 60)

# 4. Visualização Gráfica
plt.figure(figsize=(6, 5))
plt.imshow(dados, cmap='gray', interpolation='nearest')
plt.title(f"Visualização da Imagem {tamanho}x{tamanho}")
plt.colorbar(label="Intensidade de Cinza (0-255)")
plt.show()

