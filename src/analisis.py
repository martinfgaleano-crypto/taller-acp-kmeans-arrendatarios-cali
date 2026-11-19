"""Funciones de preparación, ACP y K-Means usadas por la aplicación.

Todo se calcula en vivo a partir de data/arrendatarios_cali_2025.parquet.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.preprocessing import StandardScaler

SMMLV_2025 = 1_423_500
SEMILLA = 42

VARIABLES = {
    "log_ingreso_pc": ("Ingreso por persona (log)", "Capacidad económica del hogar ajustada por tamaño"),
    "carga": ("Carga del arriendo (%)", "Canon / ingreso del hogar × 100, recortada en 100"),
    "personas": ("Personas en el hogar", "Tamaño del hogar"),
    "hacinamiento": ("Personas por dormitorio", "Densidad de ocupación de la vivienda"),
    "edad_jefe": ("Edad de la persona de referencia", "Etapa del ciclo de vida"),
    "tasa_perceptores": ("Perceptores laborales / personas", "Cuántos miembros generan ingreso laboral"),
    "formalidad": ("Perceptores que cotizan a pensión (%)", "Proxy de protección y verificabilidad del ingreso"),
}
DEFECTO = list(VARIABLES)


def etiqueta(v: str) -> str:
    return VARIABLES.get(v, (v,))[0]


def cargar(ruta=None) -> pd.DataFrame:
    from pathlib import Path
    ruta = ruta or Path(__file__).resolve().parent.parent / "data" / "arrendatarios_cali_2025.parquet"
    return pd.read_parquet(ruta)


def preparar(raw: pd.DataFrame, recorte_carga: float = 100.0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aplica filtros de calidad y construye las variables analíticas. Devuelve (base, bitácora)."""
    pasos = []

    def paso(nombre, df, motivo):
        pasos.append({"Paso": nombre, "Hogares-mes (muestra)": len(df),
                      "Hogares en un mes promedio": round(df.factor_expansion.sum() / 12), "Motivo": motivo})

    df = raw.copy()
    paso("Arrendatarios de Cali A.M. (GEIH 2025)", df, "AREA = 76 y P5090 = 3")
    df = df[df.canon.notna()]
    paso("Con canon válido", df, "Se excluye P5140 = 98/99 (no sabe / no informa)")
    df = df[df.ingreso_hogar.notna()]
    paso("Con ingreso completo", df, "Ocupado sin ingreso laboral o monto de otro ingreso sin dato")
    df = df[df.ingreso_hogar > 0].copy()
    paso("Con ingreso positivo", df, "Ingreso = 0 no permite calcular la carga del arriendo")
    dup = int(df.duplicated().sum())

    df["ingreso_pc"] = df.ingreso_hogar / df.personas
    df["log_ingreso_pc"] = np.log(df.ingreso_pc)
    df["carga_original"] = 100 * df.canon / df.ingreso_hogar
    df["carga"] = df.carga_original.clip(upper=recorte_carga)
    df["hacinamiento"] = df.personas / df.dormitorios
    df["tasa_perceptores"] = df.perceptores_laborales / df.personas
    df["formalidad"] = np.where(df.perceptores_laborales > 0,
                                100 * df.cotizantes / df.perceptores_laborales.replace(0, np.nan), 0.0)
    df["banda_smmlv"] = pd.cut(df.ingreso_hogar, [0, SMMLV_2025, 2 * SMMLV_2025 - 1, 4 * SMMLV_2025, np.inf],
                               labels=["≤1 SMMLV", "1–2 SMMLV", "2–4 SMMLV", ">4 SMMLV"])
    df["margen"] = df.ingreso_hogar - df.canon
    bit = pd.DataFrame(pasos)
    bit.attrs["duplicados"] = dup
    bit.attrs["recortados"] = int((df.carga_original > recorte_carga).sum())
    return df.reset_index(drop=True), bit


@dataclass
class ResultadoACP:
    escalador: StandardScaler
    pca: PCA
    Z: np.ndarray
    scores: np.ndarray
    variables: list

    @property
    def varianza(self):
        return self.pca.explained_variance_ratio_

    @property
    def autovalores(self):
        return self.pca.explained_variance_

    def cargas(self) -> pd.DataFrame:
        """Cargas = vectores propios × raíz del autovalor (correlación variable-componente)."""
        L = self.pca.components_.T * np.sqrt(self.pca.explained_variance_)
        return pd.DataFrame(L, index=[etiqueta(v) for v in self.variables],
                            columns=[f"CP{i+1}" for i in range(L.shape[1])])


def acp(df: pd.DataFrame, variables: list) -> ResultadoACP:
    esc = StandardScaler()
    Z = esc.fit_transform(df[variables])
    p = PCA(random_state=SEMILLA).fit(Z)
    return ResultadoACP(esc, p, Z, p.transform(Z), variables)


def componentes_sugeridos(res: ResultadoACP, umbral_acum=0.70) -> dict:
    kaiser = int((res.autovalores > 1).sum())
    acum = int(np.searchsorted(np.cumsum(res.varianza), umbral_acum) + 1)
    return {"kaiser": kaiser, "acumulado": acum, "sugerido": max(kaiser, 2)}


def evaluar_k(X: np.ndarray, pesos, ks=range(2, 9)) -> pd.DataFrame:
    filas = []
    for k in ks:
        km = KMeans(k, n_init=5, random_state=SEMILLA).fit(X, sample_weight=pesos)
        sil = silhouette_score(X, km.labels_, sample_size=min(3000, len(X)), random_state=SEMILLA)
        filas.append({"K": k, "Inercia": km.inertia_, "Silueta": sil})
    return pd.DataFrame(filas)


def kmeans(X: np.ndarray, k: int, pesos) -> tuple[np.ndarray, np.ndarray, KMeans]:
    km = KMeans(k, n_init=20, random_state=SEMILLA).fit(X, sample_weight=pesos)
    return km.labels_, silhouette_samples(X, km.labels_), km


def wq(x, w, q=0.5):
    x, w = np.asarray(x, float), np.asarray(w, float)
    o = np.argsort(x, kind="stable")
    c = np.cumsum(w[o])
    return float(x[o][np.searchsorted(c, q * c[-1])])


def perfiles(df: pd.DataFrame, col="segmento") -> pd.DataFrame:
    filas = []
    W = df.factor_expansion.sum()
    for s, g in df.groupby(col, observed=True):
        w = g.factor_expansion
        fin = g[g.respuesta_financiera]
        filas.append({
            col: s, "Hogares-mes (n)": len(g), "Hogares en un mes promedio": w.sum() / 12, "% de arrendatarios": 100 * w.sum() / W,
            "Ingreso del hogar (mediana)": wq(g.ingreso_hogar, w), "Ingreso por persona (mediana)": wq(g.ingreso_pc, w),
            "Canon (mediana)": wq(g.canon, w), "Carga del arriendo (mediana, %)": wq(g.carga_original, w),
            "Margen después del arriendo (mediana)": wq(g.margen, w),
            "Personas (promedio)": np.average(g.personas, weights=w), "Personas por dormitorio": np.average(g.hacinamiento, weights=w),
            "Edad persona de referencia": np.average(g.edad_jefe, weights=w),
            "Perceptores laborales": np.average(g.perceptores_laborales, weights=w),
            "Perceptores que cotizan (%)": np.average(g.formalidad, weights=w),
            "Productos financieros (promedio)*": np.average(fin.productos_financieros, weights=fin.factor_expansion) if len(fin) else np.nan,
            "% carga ≥ 30 %": 100 * w[g.carga_original >= 30].sum() / w.sum(),
            "% en banda 2–4 SMMLV": 100 * w[g.banda_smmlv == "2–4 SMMLV"].sum() / w.sum(),
            "Silueta media": g.silueta.mean() if "silueta" in g else np.nan,
            "% casos frontera (silueta < 0)": 100 * (g.silueta < 0).mean() if "silueta" in g else np.nan,
        })
    return pd.DataFrame(filas)


def nombrar(df: pd.DataFrame, variables: list, col="cluster") -> dict:
    """Nombre descriptivo por segmento: las dos variables que más lo distinguen (diferencia estandarizada)."""
    z = df[variables].apply(lambda s: (s - s.mean()) / s.std())
    medias = z.groupby(df[col]).mean()
    nombres = {}
    corto = {"log_ingreso_pc": "ingreso por persona", "carga": "carga del arriendo", "personas": "tamaño del hogar",
             "hacinamiento": "hacinamiento", "edad_jefe": "edad", "tasa_perceptores": "perceptores", "formalidad": "cotización"}
    for c, fila in medias.iterrows():
        top = fila.abs().sort_values(ascending=False).index[:2]
        partes = [f"{'↑' if fila[v] > 0 else '↓'} {corto.get(v, v)}" for v in top]
        nombres[c] = " · ".join(partes)
    # evitar nombres repetidos
    vistos = {}
    for c, n in nombres.items():
        vistos[n] = vistos.get(n, 0) + 1
        if vistos[n] > 1:
            nombres[c] = f"{n} ({vistos[n]})"
    return nombres


def estabilidad(X: np.ndarray, pesos, k: int, etiquetas: np.ndarray, Z_completo: np.ndarray | None = None) -> pd.DataFrame:
    """Robustez de la segmentación: índice de Rand ajustado (ARI) frente a la solución elegida.

    ARI = 1 → misma partición; ARI ≈ 0 → coincidencia por azar.
    """
    from sklearn.metrics import adjusted_rand_score as ari
    filas = []
    semillas = [ari(etiquetas, KMeans(k, n_init=5, random_state=s).fit(X, sample_weight=pesos).labels_) for s in range(1, 11)]
    filas.append({"Prueba": "Otras 10 semillas aleatorias", "ARI promedio": np.mean(semillas), "ARI mínimo": np.min(semillas)})
    rng = np.random.default_rng(SEMILLA)
    sub = []
    for b in range(20):
        idx = rng.choice(len(X), int(0.8 * len(X)), replace=False)
        w = None if pesos is None else np.asarray(pesos)[idx]
        sub.append(ari(etiquetas[idx], KMeans(k, n_init=5, random_state=b).fit(X[idx], sample_weight=w).labels_))
    filas.append({"Prueba": "20 submuestras del 80 % de los hogares", "ARI promedio": np.mean(sub), "ARI mínimo": np.min(sub)})
    alt = ari(etiquetas, KMeans(k, n_init=20, random_state=SEMILLA).fit(X, sample_weight=None if pesos is not None else np.ones(len(X))).labels_)
    filas.append({"Prueba": "Cambiar la ponderación (con ↔ sin FEX_C18)", "ARI promedio": alt, "ARI mínimo": alt})
    if Z_completo is not None:
        z = ari(etiquetas, KMeans(k, n_init=20, random_state=SEMILLA).fit(Z_completo, sample_weight=pesos).labels_)
        filas.append({"Prueba": "Usar todas las variables estandarizadas en vez de los componentes", "ARI promedio": z, "ARI mínimo": z})
    return pd.DataFrame(filas)


def inconsistencias(raw: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    """Revisión de valores atípicos o lógicamente dudosos (se reportan, no se eliminan salvo lo indicado)."""
    reglas = [
        ("Canon con código 98/99 (no sabe / no informa)", int(raw.canon.isna().sum()), "Excluido: no hay canon"),
        ("Ingreso del hogar sin dato completo", int(raw.ingreso_hogar.isna().sum()), "Excluido: no se imputa"),
        ("Ingreso del hogar = 0", int((raw.ingreso_hogar == 0).sum()), "Excluido: la carga no es calculable"),
        ("Carga del arriendo > 100 %", int((df.carga_original > 100).sum()), "Se conserva; se recorta en el modelo"),
        ("Canon mayor que 3 veces el ingreso", int((df.canon > 3 * df.ingreso_hogar).sum()), "Se conserva; probable ingreso corriente bajo o incompleto"),
        ("Persona de referencia menor de 18 años", int((df.edad_jefe < 18).sum()), "Se conserva; es posible y son muy pocos"),
        ("Hogar sin dormitorios registrados", int((df.dormitorios == 0).sum()), "No ocurre"),
        ("Más perceptores que personas", int((df.perceptores_laborales > df.personas).sum()), "No ocurre"),
        ("Filas duplicadas", int(raw.duplicated().sum()), "Revisar"),
    ]
    t = pd.DataFrame(reglas, columns=["Revisión", "Hogares-mes", "Decisión"])
    t.loc[t["Hogares-mes"] == 0, "Decisión"] = "No ocurre"
    return t
