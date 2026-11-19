"""Construye la base hogar-mes de arrendatarios de Cali A.M. a partir de la GEIH 2025.

Uso (desde la raíz del repositorio):

    python src/preparar_datos.py --geih <carpeta con personas_2025-MM.parquet y hogares_2025-MM.parquet>

Salida: data/arrendatarios_cali_2025.parquet (una fila por hogar-mes, sin identificadores).

Los Parquet mensuales se obtienen de los microdatos públicos del DANE (GEIH 2025, módulos
"Datos del hogar y la vivienda", "Características generales", "Ocupados" y "Otros ingresos
e impuestos"), unidos por PERIODO + DIRECTORIO + SECUENCIA_P + HOGAR (+ ORDEN en personas).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

KEY = ["PERIODO_AAAAMM", "DIRECTORIO", "SECUENCIA_P", "HOGAR"]
SMMLV_2025 = 1_423_500  # Decreto 1572 de 2024
NO_MIEMBROS = [10, 11, 12]  # servicio doméstico, pensionistas y trabajadores residentes

MENSUALES = [("P7495", "P7500S1", "P7500S1A1"), ("P7495", "P7500S2", "P7500S2A1"), ("P7495", "P7500S3", "P7500S3A1")]
ANUALES = [("P7505", "P7510S1", "P7510S1A1"), ("P7505", "P7510S2", "P7510S2A1"),
           ("P7505", "P7510S3", "P7510S3A1"), ("P7505", "P7510S5", "P7510S5A1")]
COLS_P = KEY + ["ORDEN", "P6040", "P6050", "OCI", "INGLABO", "P6920", "P7495", "P7505"] + \
    [c for x in MENSUALES + ANUALES for c in x[1:]]
COLS_H = KEY + ["AREA", "FEX_C18", "P5090", "P5140", "P5010"] + [f"P5222S{i}" for i in range(1, 10)]


def otro_ingreso(p: pd.DataFrame, filtro: str, resp: str, monto: str) -> pd.Series:
    """Aporte de un componente de 'otros ingresos'. NaN = dato incompleto.

    Personas <15 años no responden este módulo (el filtro es nulo para todas ellas):
    se tratan como aporte 0, igual que su ingreso laboral.
    """
    out = pd.Series(np.nan, index=p.index)
    fuera = (p.P6040 < 15) & p[filtro].isna() | p.P6050.isin(NO_MIEMBROS) | (p.P6040 < 10)
    out[fuera] = 0
    out[~fuera & (p[filtro] == 2)] = 0
    out[~fuera & (p[filtro] == 1) & (p[resp] == 2)] = 0
    ok = ~fuera & (p[filtro] == 1) & (p[resp] == 1) & p[monto].notna() & ~p[monto].isin([98, 99]) & (p[monto] >= 0)
    out[ok] = p.loc[ok, monto]
    return out


def construir(geih: Path) -> pd.DataFrame:
    hf = sorted(geih.rglob("hogares_2025-*.parquet"))
    pf = sorted(geih.rglob("personas_2025-*.parquet"))
    assert len(hf) == 12 and len(pf) == 12, f"Se esperaban 12+12 Parquet, hay {len(hf)}+{len(pf)}"
    H = pd.concat([pd.read_parquet(f, columns=COLS_H) for f in hf], ignore_index=True)
    H = H[(H.AREA.astype(str).str.strip() == "76") & (H.P5090 == 3)]          # Cali A.M. y arriendo
    P = pd.concat([pd.read_parquet(f, columns=COLS_P) for f in pf], ignore_index=True)
    P = P.merge(H[KEY], on=KEY)                                                 # personas de esos hogares
    P = P[~P.P6050.isin(NO_MIEMBROS)].copy()

    # Ingreso laboral individual (INGLABO) solo para ocupados; NaN si el ocupado no lo reporta
    P["laboral"] = np.where(P.OCI == 1, P.INGLABO.where(P.INGLABO >= 0), 0.0)
    mens = [otro_ingreso(P, *x) for x in MENSUALES]
    anu = [otro_ingreso(P, *x) for x in ANUALES]
    P["ing_persona"] = P.laboral + sum(mens) + sum(anu) / 12                  # anuales → mensual
    P["perceptor"] = P.laboral.fillna(0) > 0
    P["cotiza"] = P.perceptor & (P.P6920 == 1)

    g = P.groupby(KEY)
    hh = pd.DataFrame({
        "ingreso_hogar": g.ing_persona.sum(min_count=1).where(~g.ing_persona.apply(lambda s: s.isna().any())),
        "personas": g.size(),
        "edad_jefe": g.apply(lambda d: d.loc[d.P6050 == 1, "P6040"].max()),
        "perceptores_laborales": g.perceptor.sum(),
        "cotizantes": g.cotiza.sum(),
    }).reset_index()

    df = H.merge(hh, on=KEY, how="left", validate="one_to_one")
    df["canon"] = df.P5140.where((df.P5140 > 0) & ~df.P5140.isin([98, 99]))
    df["productos_financieros"] = (df[[f"P5222S{i}" for i in range(1, 9)]] == 1).sum(axis=1)
    df["respuesta_financiera"] = (df[[f"P5222S{i}" for i in range(1, 10)]] == 1).any(axis=1)
    df["dormitorios"] = df.P5010
    out = df[["PERIODO_AAAAMM", "FEX_C18", "ingreso_hogar", "canon", "personas", "dormitorios", "edad_jefe",
              "perceptores_laborales", "cotizantes", "productos_financieros", "respuesta_financiera"]].rename(
        columns={"PERIODO_AAAAMM": "mes", "FEX_C18": "factor_expansion"})
    return out.reset_index(drop=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--geih", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("data/arrendatarios_cali_2025.parquet"))
    a = ap.parse_args()
    df = construir(a.geih)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(a.out, index=False, compression="zstd")
    print(f"{len(df)} hogares-mes guardados en {a.out}  ·  {df.factor_expansion.sum()/12:,.0f} hogares en un mes promedio")
