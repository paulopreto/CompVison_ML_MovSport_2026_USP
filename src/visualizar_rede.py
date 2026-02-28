#!/usr/bin/env python3
"""
visualizar_rede.py — Visualização didática da rede RastreadorDePontoCNN (.pth).

Para uso em aula: mostra a arquitetura em texto legível e gera um diagrama
visual (fluxo das camadas) para facilitar o entendimento humano.

O arquivo .pth contém apenas os pesos (state_dict). A estrutura da rede
vem da classe RastreadorDePontoCNN definida em trainblack7x7.py.

Uso:
    python visualizar_rede.py
    python visualizar_rede.py -m src/modelo_rastreador.pth -o diagrama_rede.png
    python visualizar_rede.py --help
"""

import argparse
import sys
from pathlib import Path

# Permite rodar de dentro de src/ ou da raiz do projeto
sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch

from trainblack7x7 import RastreadorDePontoCNN


def carregar_modelo(caminho_pth, tamanho=7):
    """Carrega apenas os pesos (state_dict) com segurança e instancia a rede."""
    caminho = Path(caminho_pth)
    if not caminho.exists():
        print(f"ERRO: Arquivo não encontrado: {caminho}")
        print("Dica: Treine antes com python trainblack7x7.py -o modelo_rastreador.pth")
        sys.exit(1)

    # Segurança: evita executar código arbitrário ao carregar .pth de origem desconhecida
    state_dict = torch.load(str(caminho), weights_only=True, map_location=torch.device("cpu"))
    modelo = RastreadorDePontoCNN(tamanho=tamanho)
    modelo.load_state_dict(state_dict, strict=True)
    modelo.eval()
    return modelo, state_dict


def resumo_texto(modelo, state_dict):
    """Imprime um resumo da arquitetura e dos pesos em formato fácil de ler em aula."""
    print("\n" + "=" * 60)
    print("  VISUALIZAÇÃO DIDÁTICA — RastreadorDePontoCNN")
    print("=" * 60)

    print("\n--- 1. ESTRUTURA DA REDE (como o dado flui) ---\n")
    print("  ENTRADA: imagem 7×7 em escala de cinza (1 canal)")
    print("       ↓")
    print("  Bloco Convolucional:")
    print("    • Conv2d: 1 canal → 8 canais, kernel 3×3, padding 1")
    print("    • ReLU")
    print("    • Conv2d: 8 canais → 16 canais, kernel 3×3, padding 1")
    print("    • ReLU")
    print("       ↓")
    print("  Bloco Denso (após Flatten):")
    print("    • Flatten: 16×7×7 = 784 valores")
    print("    • Linear: 784 → 32")
    print("    • ReLU")
    print("    • Linear: 32 → 2")
    print("       ↓")
    print("  SAÍDA: 2 números (coordenada x, coordenada y)")

    print("\n--- 2. PESOS (state_dict) — nome da camada e formato dos tensores ---\n")
    for nome, tensor in state_dict.items():
        forma = tuple(tensor.shape)
        num_params = tensor.numel()
        print(f"  {nome:45} | forma: {str(forma):20} | parâmetros: {num_params:,}")

    total = sum(p.numel() for p in modelo.parameters())
    print(f"\n  Total de parâmetros da rede: {total:,}")

    print("\n--- 3. RESUMO DO MODELO (PyTorch) ---\n")
    print(modelo)
    print("\n" + "=" * 60)


def desenhar_diagrama(caminho_saida, tamanho=7):
    """Gera um diagrama visual da arquitetura (sem depender dos pesos) para uso em aula."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        print("Dica: para gerar o diagrama, instale matplotlib: pip install matplotlib")
        return

    fig, ax = plt.subplots(1, 1, figsize=(14, 5))
    ax.set_xlim(-0.5, 13)
    ax.set_ylim(0, 5.5)
    ax.axis("off")

    cores = {
        "entrada": "#E3F2FD",
        "conv": "#BBDEFB",
        "relu": "#90CAF9",
        "flatten": "#FFF9C4",
        "linear": "#C8E6C9",
        "saida": "#FFCCBC",
    }

    def caixa(ax, x, y, w, h, texto, cor, fontsize=9):
        rect = mpatches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02",
                                        facecolor=cor, edgecolor="#333", linewidth=1.2)
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, texto, ha="center", va="center", fontsize=fontsize)

    def seta(ax, x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="->", color="#333", lw=1.5))

    # Linha única: fluxo da esquerda para a direita (fácil de seguir em aula)
    # Entrada
    caixa(ax, 0.2, 2.0, 1.0, 1.0, "Entrada\n1×7×7", cores["entrada"], 8)
    seta(ax, 1.2, 2.5, 1.5, 2.5)
    # Conv 1
    caixa(ax, 1.5, 2.1, 1.2, 0.8, "Conv2d\n1→8, 3×3", cores["conv"], 7)
    seta(ax, 2.7, 2.5, 3.0, 2.5)
    caixa(ax, 3.0, 2.1, 0.6, 0.8, "ReLU", cores["relu"], 8)
    seta(ax, 3.6, 2.5, 3.9, 2.5)
    # Conv 2
    caixa(ax, 3.9, 2.1, 1.2, 0.8, "Conv2d\n8→16, 3×3", cores["conv"], 7)
    seta(ax, 5.1, 2.5, 5.4, 2.5)
    caixa(ax, 5.4, 2.1, 0.6, 0.8, "ReLU", cores["relu"], 8)
    seta(ax, 6.0, 2.5, 6.3, 2.5)
    # Flatten
    n_flat = 16 * tamanho * tamanho
    caixa(ax, 6.3, 2.1, 1.0, 0.8, f"Flatten\n{n_flat}", cores["flatten"], 7)
    seta(ax, 7.3, 2.5, 7.6, 2.5)
    # Linear 1
    caixa(ax, 7.6, 2.1, 1.0, 0.8, "Linear\n784→32", cores["linear"], 7)
    seta(ax, 8.6, 2.5, 8.9, 2.5)
    caixa(ax, 8.9, 2.1, 0.6, 0.8, "ReLU", cores["relu"], 8)
    seta(ax, 9.5, 2.5, 9.8, 2.5)
    # Linear 2 e Saída
    caixa(ax, 9.8, 2.1, 1.0, 0.8, "Linear\n32→2", cores["linear"], 7)
    seta(ax, 10.8, 2.5, 11.2, 2.5)
    caixa(ax, 11.2, 2.0, 1.0, 1.0, "Saída\n(x, y)", cores["saida"], 8)

    ax.text(6.5, 4.2, "RastreadorDePontoCNN — fluxo dos dados (esquerda → direita)", ha="center", va="center", fontsize=13, fontweight="bold")
    ax.text(6.5, 3.7, "Uso em aula: cada caixa é uma camada; as setas indicam o fluxo da informação.", ha="center", va="center", fontsize=9, color="gray")

    out = Path(caminho_saida)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(str(out), dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Diagrama salvo em: {out}")


def main():
    parser = argparse.ArgumentParser(
        description="Visualização didática da rede neural (arquitetura e pesos) a partir de um .pth."
    )
    parser.add_argument(
        "-m", "--modelo",
        type=str,
        default="modelo_rastreador.pth",
        help="Caminho do arquivo .pth (default: modelo_rastreador.pth).",
    )
    parser.add_argument(
        "-t", "--tamanho",
        type=int,
        default=7,
        help="Lado da imagem (default: 7).",
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default="",
        help="Salvar diagrama da arquitetura neste arquivo (ex: diagrama_rede.png).",
    )
    parser.add_argument(
        "--sem-diagrama",
        action="store_true",
        help="Só mostrar resumo em texto, sem gerar figura.",
    )
    parser.add_argument(
        "--export-onnx",
        type=str,
        default="",
        metavar="ARQUIVO",
        help="Exporta o modelo para ONNX (ex: modelo.onnx) para abrir no Netron (netron.app).",
    )
    args = parser.parse_args()

    modelo, state_dict = carregar_modelo(args.modelo, tamanho=args.tamanho)
    resumo_texto(modelo, state_dict)

    if not args.sem_diagrama:
        destino = args.output or "diagrama_rastreador_rede.png"
        desenhar_diagrama(destino, tamanho=args.tamanho)

    if args.export_onnx:
        caminho_onnx = Path(args.export_onnx)
        try:
            dummy = torch.randn(1, 1, args.tamanho, args.tamanho)
            torch.onnx.export(modelo, dummy, str(caminho_onnx), input_names=["imagem_7x7"], output_names=["coordenadas_xy"])
            print(f"Modelo exportado para ONNX: {caminho_onnx}")
            print("  Abra em https://netron.app para ver o grafo da rede de forma interativa.")
        except ModuleNotFoundError as e:
            if "onnxscript" in str(e):
                print("Erro: exportação ONNX requer o pacote 'onnxscript'. Instale com:")
                print("  pip install onnxscript")
                print("  ou: uv add onnxscript")
                sys.exit(1)
            raise


if __name__ == "__main__":
    main()
