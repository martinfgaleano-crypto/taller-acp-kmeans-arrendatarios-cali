"""Reproduce, sin Streamlit, las tablas, métricas y figuras que se citan en el README.

Uso (desde la raíz del repositorio):  python src/reproducir_resultados.py
Salidas: resultados/*.csv y docs/figuras/*.png (configuración por defecto de la app).
"""
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import analisis as A  # noqa: E402

RAIZ = Path(__file__).resolve().parent.parent
OUT, FIG = RAIZ / "resultados", RAIZ / "docs" / "figuras"
OUT.mkdir(exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)
VERDE, NARANJA = "#0F5257", "#D1603D"
PALETA = ["#0F5257", "#D1603D", "#7A9E7E", "#E0A458"]

raw = A.cargar()
df, bitacora = A.preparar(raw)
res = A.acp(df, A.DEFECTO)
n = A.componentes_sugeridos(res)["sugerido"]
X, w = res.scores[:, :n], df.factor_expansion.values
etq, sil, _ = A.kmeans(X, 4, w)
df["cluster"], df["silueta"] = etq, sil
t = df.groupby("cluster")
roles = {"Hogares pequeños con ingreso formal": t.ingreso_pc.median().idxmax(),
         "Presión crítica del arriendo": t.carga_original.median().idxmax(),
         "Familias numerosas con ingreso diluido": t.personas.mean().idxmax(),
         "Jefatura mayor con baja cotización": t.edad_jefe.mean().idxmax()}
df["segmento"] = df.cluster.map({c: nmb for nmb, c in roles.items()})

# --- tablas
bitacora.to_csv(OUT / "01_preparacion.csv", index=False, encoding="utf-8-sig")
pd.DataFrame({"componente": [f"CP{i+1}" for i in range(len(A.DEFECTO))], "autovalor": res.autovalores,
              "varianza_pct": 100 * res.varianza, "acumulada_pct": 100 * np.cumsum(res.varianza)}).round(3).to_csv(OUT / "02_varianza_acp.csv", index=False, encoding="utf-8-sig")
res.cargas().round(3).to_csv(OUT / "03_cargas_acp.csv", encoding="utf-8-sig")
ev = A.evaluar_k(X, w)
ev.round(4).to_csv(OUT / "04_codo_silueta.csv", index=False, encoding="utf-8-sig")
A.estabilidad(X, w, 4, etq, res.Z).round(3).to_csv(OUT / "05_estabilidad.csv", index=False, encoding="utf-8-sig")
P = A.perfiles(df)
P.round(2).to_csv(OUT / "06_perfiles_segmentos.csv", index=False, encoding="utf-8-sig")
cr = df.groupby(["banda_smmlv", "segmento"], observed=True).factor_expansion.sum().unstack()
cr = 100 * cr.div(cr.sum(axis=1), axis=0)
cr.round(1).to_csv(OUT / "07_banda_por_segmento.csv", encoding="utf-8-sig")

# --- figuras
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
c = [f"CP{i+1}" for i in range(len(A.DEFECTO))]
ax[0].bar(c, 100 * res.varianza, color=VERDE)
ax[0].plot(c, 100 * np.cumsum(res.varianza), "o-", color=NARANJA)
ax[0].axhline(70, ls=":", c="gray")
ax[0].set(title="Varianza explicada y acumulada", ylabel="% de la varianza")
ax[1].plot(c, res.autovalores, "o-", color=VERDE)
ax[1].axhline(1, ls=":", c="gray")
ax[1].set(title="Sedimentación (Kaiser = 1)", ylabel="Autovalor")
fig.tight_layout()
fig.savefig(FIG / "acp_varianza.png", dpi=72)

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].plot(ev.K, ev.Inercia, "o-", color=VERDE)
ax[0].set(title="Método del codo", xlabel="K", ylabel="Inercia (ponderada)")
ax[1].plot(ev.K, ev.Silueta, "o-", color=NARANJA)
ax[1].set(title="Silueta promedio", xlabel="K", ylabel="Silueta")
for a in ax:
    a.axvline(4, ls=":", c="gray")
fig.tight_layout()
fig.savefig(FIG / "kmeans_k.png", dpi=72)

fig, ax = plt.subplots(figsize=(8, 6))
for i, (s, g) in enumerate(df.groupby("segmento")):
    ax.scatter(X[g.index, 0], X[g.index, 1], s=4, alpha=.45, color=PALETA[i], label=s)
ax.set(title=f"Segmentos en CP1–CP2 ({100*res.varianza[:2].sum():.1f} % de la varianza)", xlabel="CP1 · capacidad económica", ylabel="CP2 · tamaño familiar")
ax.legend(markerscale=4, fontsize=8)
fig.tight_layout()
fig.savefig(FIG / "segmentos_plano.png", dpi=72)

fig, ax = plt.subplots(figsize=(8, 4.5))
cr.plot(kind="bar", stacked=True, ax=ax, color=PALETA[:cr.shape[1]], rot=0)
ax.set(title="Composición de cada banda de ingreso por segmento (ponderado)", xlabel="Ingreso del hogar en SMMLV 2025", ylabel="% de hogares")
ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1, 1))
fig.tight_layout()
fig.savefig(FIG / "bandas_segmentos.png", dpi=72)
try:  # PNG livianos (paleta de 64 colores)
    from PIL import Image
    for f in FIG.glob("*.png"):
        Image.open(f).convert("RGB").quantize(64).save(f, optimize=True)
except ImportError:
    pass
print(f"Listo: {len(list(OUT.glob('*.csv')))} tablas en resultados/ y {len(list(FIG.glob('*.png')))} figuras en docs/figuras/")
