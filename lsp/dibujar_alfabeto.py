# Dibuja el alfabeto que el modelo aprendió, a partir del dataset grabado.
# Sirve para comparar contra la guía oficial del MINEDU y marcar diferencias.
import csv
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CSV = "asl_data_ANTIGUO.csv"
OUT = "alfabeto_actual_grabado.png"

# Conexiones de la mano (MediaPipe): dedo por dedo + la palma
CONN = [(0,1),(1,2),(2,3),(3,4),           # pulgar
        (0,5),(5,6),(6,7),(7,8),           # índice
        (9,10),(10,11),(11,12),            # medio
        (13,14),(14,15),(15,16),           # anular
        (0,17),(17,18),(18,19),(19,20),    # meñique
        (5,9),(9,13),(13,17)]              # palma

# letras con movimiento: el promedio no las representa bien
MOVIMIENTO = {"j", "z"}

por_letra = defaultdict(list)
with open(CSV) as f:
    r = csv.reader(f)
    next(r)
    for row in r:
        if row:
            por_letra[row[0]].append([float(v) for v in row[1:64]])  # 63 = forma

etiquetas = sorted(por_letra, key=lambda k: (not k.isalpha(), k))
n = len(etiquetas)
cols, filas = 7, (n + 6) // 7
fig, axes = plt.subplots(filas, cols, figsize=(cols * 2.1, filas * 2.4))
axes = np.atleast_2d(axes).ravel()

for ax, lab in zip(axes, etiquetas):
    pts = np.array(por_letra[lab]).mean(axis=0).reshape(21, 3)
    x, y = pts[:, 0], -pts[:, 1]          # y invertida: la imagen crece hacia abajo
    for a, b in CONN:
        ax.plot([x[a], x[b]], [y[a], y[b]], "-", color="#80252E", linewidth=2.2)
    ax.plot(x, y, "o", color="#E8B93B", markersize=3.5,
            markeredgecolor="#80252E", markeredgewidth=.6)
    titulo = lab.upper() if len(lab) == 1 else lab
    if lab in MOVIMIENTO:
        titulo += "  (movimiento)"
    ax.set_title(titulo, fontsize=11, fontweight="bold",
                 color="#999" if lab in MOVIMIENTO else "#222")
    ax.set_aspect("equal")
    ax.axis("off")
    m = 0.15
    ax.set_xlim(x.min() - m, x.max() + m)
    ax.set_ylim(y.min() - m, y.max() + m)

for ax in axes[n:]:
    ax.axis("off")

fig.suptitle("Señas que el modelo tiene grabadas actualmente\n"
             "(promedio de todas las muestras · compáralas con la guía del MINEDU)",
             fontsize=13, fontweight="bold", color="#80252E", y=0.995)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(OUT, dpi=130, facecolor="white")
print("guardado en", OUT, "|", n, "señas dibujadas")
