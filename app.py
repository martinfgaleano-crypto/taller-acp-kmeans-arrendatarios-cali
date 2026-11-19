"""¿Quiénes son los hogares que viven en arriendo en Cali A.M.?
ACP y K-Means sobre la GEIH 2025 · Taller individual, Machine Learning aplicado a Economía y Finanzas.
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")  # menos picos de CPU en Streamlit Cloud
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src import analisis as A

st.set_page_config(page_title="Arrendatarios de Cali · ACP y K-Means", page_icon="🏠", layout="wide")

PALETA = ["#0F5257", "#D1603D", "#7A9E7E", "#E0A458", "#5B5F97", "#B56576", "#3D8EB9", "#8C6A5D"]
FUENTE = "Fuente: DANE, GEIH 2025 (12 meses). Cálculos propios. Cali A.M. = dominio Cali–Yumbo."

st.markdown("""
<style>
.block-container {padding-top: 2rem; max-width: 1200px;}
h1, h2, h3 {letter-spacing: -0.01em;}
.nota {font-size: 0.85rem; color: #5c6b6b;}
.kpi {background:#F3EFE7; border-radius:10px; padding:14px 16px; border-left:4px solid #0F5257;}
.kpi b {font-size:1.6rem; color:#0F5257; display:block;}
.kpi span {font-size:0.85rem; color:#4a4a4a;}
.caja {background:#F3EFE7; border-radius:10px; padding:14px 18px; margin:6px 0 14px 0;}
</style>""", unsafe_allow_html=True)


def kpi(col, valor, texto):
    col.markdown(f"<div class='kpi'><b>{valor}</b><span>{texto}</span></div>", unsafe_allow_html=True)


def pesos(x):
    return f"${x:,.0f}".replace(",", ".")


def num(x, d=0):
    s = f"{x:,.{d}f}"
    return s.replace(",", "X").replace(".", ",").replace("X", ".")


@st.cache_data
def datos_crudos():
    return A.cargar()


@st.cache_data(show_spinner=False)
def modelo(variables: tuple, recorte: float, n_comp: int | None, k: int, ponderar: bool):
    """Toda la cadena (preparación → ACP → K-Means) en caché por combinación de parámetros."""
    df, bit = A.preparar(datos_crudos(), recorte)
    res = A.acp(df, list(variables))
    sug = A.componentes_sugeridos(res)
    n = n_comp or sug["sugerido"]
    X = res.scores[:, :n]
    w = df.factor_expansion.values if ponderar else None
    etq, sil, _ = A.kmeans(X, k, w)
    return df, bit, res, sug, n, X, w, etq, sil


@st.cache_data(show_spinner=False)
def evaluacion_k(variables: tuple, recorte: float, n_comp: int, ponderar: bool):
    df, _, res, _, _, X, w, _, _ = modelo(variables, recorte, n_comp, 4, ponderar)
    return A.evaluar_k(X, w)


@st.cache_data(show_spinner=False)
def evaluacion_estabilidad(variables: tuple, recorte: float, n_comp: int, k: int, ponderar: bool):
    _, _, res, _, _, X, w, etq, _ = modelo(variables, recorte, n_comp, k, ponderar)
    return A.estabilidad(X, w, k, etq, res.Z)


# ----------------------------------------------------------------------------- controles
st.sidebar.title("Arrendatarios de Cali")
st.sidebar.caption("ACP + K-Means · GEIH 2025")
pagina = st.sidebar.radio("Navegación", ["1. Contexto", "2. Exploración de datos", "3. Preparación",
                                          "4. Componentes principales", "5. Segmentación K-Means",
                                          "6. Perfiles de los segmentos", "7. Conclusiones"])
st.sidebar.divider()
st.sidebar.subheader("Parámetros del modelo")
variables = st.sidebar.multiselect("Variables del modelo", A.DEFECTO, default=A.DEFECTO, format_func=A.etiqueta)
if len(variables) < 4:
    st.sidebar.error("Selecciona al menos 4 variables.")
    st.stop()
recorte = st.sidebar.slider("Recorte de la carga del arriendo (%)", 60, 300, 100, 10,
                            help="Cargas por encima de este valor se fijan en el límite para que pocos hogares con ingreso muy bajo no dominen la distancia.")
raw = datos_crudos()
_, _, _, sug, _, _, _, _, _ = modelo(tuple(variables), recorte, None, 4, True)
n_comp = st.sidebar.slider("Componentes que se conservan", 2, len(variables), sug["sugerido"],
                           help=f"Sugerido por los datos: {sug['sugerido']} (Kaiser = {sug['kaiser']}; ≥70 % acumulado = {sug['acumulado']}).")
k = st.sidebar.slider("Número de segmentos (K)", 2, 8, 4)
ponderar = st.sidebar.toggle("Ponderar K-Means con el factor de expansión", value=True)
st.sidebar.caption("Configuración por defecto: 7 variables, recorte 100 %, componentes de Kaiser, K = 4, ponderado.")

df, bitacora, res, sug, n_comp, X, w, etiquetas, sil = modelo(tuple(variables), recorte, n_comp, k, ponderar)
df = df.copy()
df["cluster"] = etiquetas
df["silueta"] = sil
nombres = A.nombrar(df, variables)
if k == 4:
    _t = df.assign(ipc=df.ingreso_pc).groupby("cluster")
    _roles = {"Hogares pequeños con ingreso formal": _t.ipc.median().idxmax(),
              "Presión crítica del arriendo": _t.carga_original.median().idxmax(),
              "Familias numerosas con ingreso diluido": _t.personas.mean().idxmax(),
              "Jefatura mayor con baja cotización": _t.edad_jefe.mean().idxmax()}
    if len(set(_roles.values())) == 4:
        nombres = {c: n for n, c in _roles.items()}
df["segmento"] = df.cluster.map(nombres)
colores = {nombres[c]: PALETA[i % len(PALETA)] for i, c in enumerate(sorted(nombres))}
var_ret = res.varianza[:n_comp].sum()
es_defecto = (variables == A.DEFECTO and recorte == 100 and n_comp == sug["sugerido"] and k == 4 and ponderar)


# ----------------------------------------------------------------------------- 1. contexto
if pagina == "1. Contexto":
    st.title("¿Quiénes son los hogares que viven en arriendo en Cali A.M.?")
    st.markdown("#### Perfiles económicos de los arrendatarios con Análisis de Componentes Principales y K-Means")
    c1, c2, c3 = st.columns(3)
    kpi(c1, num(raw.factor_expansion.sum() / 12), "hogares arrendatarios en un mes promedio de 2025")
    kpi(c2, num(len(raw)), "hogares-mes en la muestra (12 cortes mensuales)")
    kpi(c3, f"{len(A.DEFECTO)}", "variables cuantitativas en el modelo")
    st.markdown("### Pregunta")
    st.markdown("<div class='caja'><b>¿Qué perfiles de hogares arrendatarios existen en Cali A.M. y en qué se diferencian sus condiciones "
                "para avanzar hacia la vivienda propia?</b><br>La política de vivienda suele clasificar a los hogares solo por su ingreso "
                "en salarios mínimos. Este ejercicio pregunta si hogares dentro de la misma banda de ingreso enfrentan situaciones "
                "económicas distintas.</div>", unsafe_allow_html=True)
    a, b = st.columns(2)
    with a:
        st.markdown("### Datos")
        st.markdown("""
- **Fuente:** Gran Encuesta Integrada de Hogares (GEIH), DANE, enero–diciembre de 2025. Microdatos públicos y anonimizados.
- **Cobertura:** dominio Cali A.M. (Cali–Yumbo, `AREA = 76`); hogares que viven en arriendo (`P5090 = 3`).
- **Observación:** un hogar en un mes. Los 12 meses son cortes transversales independientes (no es un panel).
- **Ponderación:** factor de expansión `FEX_C18`. Un mes promedio = suma de factores de los 12 meses / 12.
- **Variables:** ingreso del hogar reconstruido (ingreso laboral `INGLABO` + arriendos, pensiones, alimentos, transferencias, ayudas y rentas financieras), canon `P5140`, composición del hogar, dormitorios `P5010`, edad, perceptores y cotización a pensión `P6920`.
""")
    with b:
        st.markdown("### ¿Por qué ACP y K-Means?")
        st.markdown("""
- Las condiciones de un hogar arrendatario se mueven **juntas**: más ingreso por persona suele venir con más perceptores, más cotización y menos carga del arriendo. El **ACP** resume esas variables correlacionadas en pocas dimensiones interpretables.
- La pregunta es si existen **grupos** de hogares con combinaciones parecidas. **K-Means** busca esos grupos sobre los componentes, sin imponer un corte de ingreso de antemano.
- Comparar los grupos con las bandas de salarios mínimos permite ver si la clasificación por ingreso esconde situaciones distintas.
""")
        st.markdown("### Limitaciones de la fuente")
        st.markdown("""
- La GEIH **no observa** ahorro acumulado, intención de compra, historial crediticio ni aprobación hipotecaria.
- El ingreso es una **reconstrucción propia**, no una variable oficial de ingreso total del DANE.
- La cotización a pensión es un proxy parcial de formalidad.
- El ingreso menos el arriendo **no es ahorro**: aún debe cubrir alimentación, transporte, salud y educación.
""")
    st.caption(FUENTE)

# ----------------------------------------------------------------------------- 2. exploración
elif pagina == "2. Exploración de datos":
    st.title("Exploración de la base")
    c1, c2, c3, c4 = st.columns(4)
    kpi(c1, num(len(raw)), "registros hogar-mes")
    kpi(c2, f"{raw.shape[1]}", "columnas en la base")
    kpi(c3, num(raw.isna().any(axis=1).sum()), "registros con algún faltante")
    kpi(c4, num(raw.duplicated().sum()), "filas duplicadas")
    st.markdown("### Estructura y faltantes")
    info = pd.DataFrame({"Tipo": raw.dtypes.astype(str), "Faltantes": raw.isna().sum(),
                         "% faltante": (100 * raw.isna().mean()).round(2), "Valores distintos": raw.nunique()})
    st.dataframe(info, width="stretch")
    st.markdown("<p class='nota'>Los faltantes de <code>ingreso_hogar</code> corresponden a hogares con un ocupado sin ingreso laboral "
                "reportado o con un monto de otro ingreso marcado como 'no sabe'. Los de <code>canon</code> son códigos 98/99.</p>", unsafe_allow_html=True)
    st.markdown("### Revisión de inconsistencias y valores extremos")
    st.dataframe(A.inconsistencias(raw, df), width="stretch", hide_index=True)
    st.markdown("### Estadísticas descriptivas de las variables del modelo (base preparada, sin ponderar)")
    st.dataframe(df[variables + ["ingreso_hogar", "canon"]].describe().T.round(2).rename(index=A.etiqueta), width="stretch")
    st.markdown("### Distribución de una variable")
    v = st.selectbox("Variable", variables + ["ingreso_hogar", "canon", "carga_original"], format_func=A.etiqueta)
    fig = px.histogram(df, x=v, nbins=60, color_discrete_sequence=[PALETA[0]],
                       title=f"Distribución de {A.etiqueta(v).lower()} · {num(len(df))} hogares-mes")
    fig.update_layout(xaxis_title=A.etiqueta(v), yaxis_title="Hogares-mes (muestra)", bargap=0.02)
    st.plotly_chart(fig, width="stretch")
    st.markdown("### Matriz de correlaciones")
    corr = df[variables].corr()
    corr.index = corr.columns = [A.etiqueta(c) for c in variables]
    fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1, aspect="auto",
                    title="Correlación de Pearson entre las variables del modelo")
    st.plotly_chart(fig, width="stretch")
    st.markdown("<div class='caja'>El ingreso por persona se asocia de forma negativa con la carga del arriendo y de forma positiva con la "
                "proporción de perceptores y la cotización. El tamaño del hogar y el hacinamiento se mueven juntos y en sentido opuesto al ingreso por persona. "
                "Estas correlaciones moderadas justifican el ACP: hay información redundante que puede resumirse.</div>", unsafe_allow_html=True)
    st.caption(FUENTE)

# ----------------------------------------------------------------------------- 3. preparación
elif pagina == "3. Preparación":
    st.title("Decisiones de preparación")
    st.markdown("### Del universo a la base analítica")
    st.dataframe(bitacora, width="stretch", hide_index=True)
    fig = go.Figure(go.Funnel(y=bitacora.Paso, x=bitacora["Hogares en un mes promedio"], marker_color=PALETA[0],
                              textinfo="value+percent initial"))
    fig.update_layout(title="Hogares en un mes promedio que permanecen en cada paso")
    st.plotly_chart(fig, width="stretch")
    st.markdown(f"""
**Decisiones y su justificación**

1. **Ingreso del hogar.** Se suma el ingreso laboral individual (`INGLABO`) de cada ocupado y los otros ingresos (arriendos recibidos, pensiones y alimentos del último mes; transferencias, ayudas y rentas de los últimos 12 meses divididas por 12). No se suman salario u honorarios aparte, porque ya están dentro de `INGLABO`.
2. **Personas de 10 a 14 años.** El módulo de otros ingresos no se aplica a ellas (su respuesta es nula en todos los casos). Se tratan como aporte 0. Tratarlas como dato faltante habría eliminado cerca de una de cada cinco familias, justamente las que tienen hijos adolescentes.
3. **Faltantes reales.** Un hogar con un ocupado sin ingreso reportado, o con un monto marcado como 98/99, queda fuera: no se imputa con cero.
4. **Duplicados:** {bitacora.attrs['duplicados']} filas idénticas. **Identificadores:** la base no contiene `DIRECTORIO` ni otras llaves; solo el mes, que no entra al modelo.
5. **Valores extremos.** La carga del arriendo tiene casos de más de 100 % (ingreso corriente muy bajo). Se recorta en **{recorte} %** ({bitacora.attrs['recortados']} hogares afectados). En los perfiles se reporta la carga original.
6. **Transformación.** El ingreso por persona entra en logaritmo, porque su distribución es muy asimétrica.
7. **Estandarización.** ACP y K-Means dependen de escalas y distancias, así que todas las variables se llevan a media 0 y desviación 1.
""")
    st.markdown("### Antes y después de estandarizar")
    a, b = st.columns(2)
    antes = df[variables].melt(var_name="Variable", value_name="Valor")
    antes["Variable"] = antes.Variable.map(A.etiqueta)
    despues = pd.DataFrame(res.Z, columns=[A.etiqueta(v) for v in variables]).melt(var_name="Variable", value_name="Valor")
    a.plotly_chart(px.box(antes, x="Variable", y="Valor", title="Escalas originales", color_discrete_sequence=[PALETA[1]]).update_xaxes(tickangle=30), width="stretch")
    b.plotly_chart(px.box(despues, x="Variable", y="Valor", title="Variables estandarizadas (media 0, desviación 1)", color_discrete_sequence=[PALETA[0]]).update_xaxes(tickangle=30), width="stretch")
    st.caption(FUENTE)

# ----------------------------------------------------------------------------- 4. ACP
elif pagina == "4. Componentes principales":
    st.title("Análisis de Componentes Principales")
    tabla = pd.DataFrame({"Componente": [f"CP{i+1}" for i in range(len(variables))], "Autovalor": res.autovalores,
                          "Varianza explicada (%)": 100 * res.varianza, "Acumulada (%)": 100 * np.cumsum(res.varianza)})
    c1, c2, c3 = st.columns(3)
    kpi(c1, f"{n_comp}", "componentes conservados")
    kpi(c2, f"{num(100*var_ret,1)} %", "de la varianza original conservada")
    kpi(c3, f"{sug['kaiser']}", "componentes con autovalor > 1 (criterio de Kaiser)")
    a, b = st.columns([3, 2])
    fig = go.Figure()
    fig.add_bar(x=tabla.Componente, y=tabla["Varianza explicada (%)"], name="Por componente", marker_color=PALETA[0])
    fig.add_scatter(x=tabla.Componente, y=tabla["Acumulada (%)"], name="Acumulada", mode="lines+markers", line_color=PALETA[1])
    fig.add_hline(y=70, line_dash="dot", annotation_text="70 %")
    fig.update_layout(title="Varianza explicada por componente y acumulada", yaxis_title="% de la varianza total", xaxis_title="Componente")
    a.plotly_chart(fig, width="stretch")
    fig = px.line(tabla, x="Componente", y="Autovalor", markers=True, title="Gráfico de sedimentación (autovalores)")
    fig.add_hline(y=1, line_dash="dot", annotation_text="Kaiser = 1")
    fig.update_traces(line_color=PALETA[0])
    b.plotly_chart(fig, width="stretch")
    st.dataframe(tabla.round(2), width="stretch", hide_index=True)
    st.markdown(f"<div class='caja'><b>¿Cuántos componentes?</b> Los datos sugieren <b>{sug['sugerido']}</b>: {sug['kaiser']} tienen autovalor mayor que 1 "
                f"y el gráfico de sedimentación se aplana después de ese punto. Para llegar al 70 % de la varianza se necesitan {sug['acumulado']}. "
                f"Se conservan {n_comp}, que resumen el {num(100*var_ret,1)} % de la información de las {len(variables)} variables.</div>", unsafe_allow_html=True)

    st.markdown("### Cargas: ¿qué significa cada componente?")
    L = res.cargas().iloc[:, :n_comp]
    fig = px.imshow(L, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1, aspect="auto",
                    title="Correlación entre cada variable y cada componente conservado")
    st.plotly_chart(fig, width="stretch")
    for j in range(n_comp):
        col = L.iloc[:, j]
        pos = ", ".join(col[col > 0.4].sort_values(ascending=False).index.str.lower()) or "—"
        neg = ", ".join(col[col < -0.4].sort_values().index.str.lower()) or "—"
        st.markdown(f"- **CP{j+1}** ({num(100*res.varianza[j],1)} %): valores altos en *{pos}*; bajos en *{neg}*.")
    if es_defecto:
        st.markdown("<div class='caja'><b>Interpretación (configuración por defecto).</b><br>"
                    "<b>CP1 · Capacidad económica:</b> más ingreso por persona, más perceptores y más cotización, con menos carga del arriendo.<br>"
                    "<b>CP2 · Tamaño familiar frente a presión:</b> hogares grandes y con más hacinamiento, pero con carga relativa menor (el ingreso se reparte entre más personas, el canon no crece al mismo ritmo).<br>"
                    "<b>CP3 · Ciclo de vida:</b> está dominado por la edad de la persona de referencia.</div>", unsafe_allow_html=True)

    st.markdown("### Hogares en el espacio de componentes")
    a, b = st.columns(2)
    cx = a.selectbox("Eje X", [f"CP{i+1}" for i in range(n_comp)], 0)
    cy = b.selectbox("Eje Y", [f"CP{i+1}" for i in range(n_comp)], 1)
    ix, iy = int(cx[2:]) - 1, int(cy[2:]) - 1
    pct = 100 * (res.varianza[ix] + res.varianza[iy])
    plot = pd.DataFrame({cx: res.scores[:, ix], cy: res.scores[:, iy], "Banda de ingreso": df.banda_smmlv.astype(str)})
    fig = px.scatter(plot, x=cx, y=cy, color="Banda de ingreso", opacity=0.45, color_discrete_sequence=PALETA,
                     title=f"Hogares en {cx} y {cy}: este plano resume el {num(pct,1)} % de la varianza")
    esc = 3
    for i, v in enumerate(variables):
        fig.add_annotation(x=res.pca.components_[ix, i] * esc, y=res.pca.components_[iy, i] * esc, ax=0, ay=0, xref="x", yref="y",
                           axref="x", ayref="y", showarrow=True, arrowhead=2, arrowcolor="#333", text=A.etiqueta(v), font=dict(size=10))
    fig.update_traces(marker_size=4)
    st.plotly_chart(fig, width="stretch")
    st.markdown("<p class='nota'>Las flechas muestran la dirección de cada variable (biplot). El color indica la banda de ingreso en salarios mínimos de 2025: "
                "las bandas se superponen en el plano, lo que anticipa que el ingreso por sí solo no separa a los hogares.</p>", unsafe_allow_html=True)
    st.caption(FUENTE)

# ----------------------------------------------------------------------------- 5. K-Means
elif pagina == "5. Segmentación K-Means":
    st.title("Segmentación con K-Means")
    st.markdown(f"K-Means se aplica sobre los **{n_comp} componentes conservados** ({num(100*var_ret,1)} % de la varianza), "
                f"{'ponderando cada hogar con su factor de expansión' if ponderar else 'sin ponderar'}. Semilla fija = {A.SEMILLA}, 20 inicializaciones.")
    with st.spinner("Comparando valores de K…"):
        ev = evaluacion_k(tuple(variables), recorte, n_comp, ponderar)
    a, b = st.columns(2)
    fig = px.line(ev, x="K", y="Inercia", markers=True, title="Método del codo: inercia intra-segmento" + (" (ponderada)" if ponderar else ""))
    fig.add_vline(x=k, line_dash="dot", annotation_text=f"K = {k}")
    fig.update_traces(line_color=PALETA[0])
    a.plotly_chart(fig, width="stretch")
    fig = px.line(ev, x="K", y="Silueta", markers=True, title="Coeficiente de silueta promedio")
    fig.add_vline(x=k, line_dash="dot", annotation_text=f"K = {k}")
    fig.update_traces(line_color=PALETA[1])
    b.plotly_chart(fig, width="stretch")
    mejor = int(ev.loc[ev.Silueta.idxmax(), "K"])
    st.markdown(f"<div class='caja'><b>¿Cómo se elige K?</b> La silueta más alta se alcanza con <b>K = {mejor}</b> "
                f"({num(ev.Silueta.max(),3)}); con K = 4 es {num(ev.set_index('K').Silueta[4],3)}. El codo se suaviza a partir de K = 4. "
                "Se eligió <b>K = 4</b> porque tiene una silueta prácticamente igual a la mejor y separa un grupo que con K = 3 quedaba mezclado "
                "(hogares con persona de referencia mayor y baja cotización). Una silueta cercana a 0,3 indica una <b>estructura moderada</b>: "
                "los segmentos existen, pero sus bordes son difusos.</div>", unsafe_allow_html=True)

    st.markdown("### ¿Son estables los segmentos?")
    st.markdown("Se vuelve a correr K-Means con otras semillas, con submuestras del 80 % de los hogares, sin ponderación y con todas las variables, "
                "y se compara cada resultado con la segmentación elegida.")
    if st.button("Ejecutar la prueba de estabilidad", type="primary") or st.session_state.get("estabilidad_vista"):
        st.session_state["estabilidad_vista"] = True
        with st.spinner("Probando la estabilidad de la segmentación (unos segundos)…"):
            est = evaluacion_estabilidad(tuple(variables), recorte, n_comp, k, ponderar)
        st.dataframe(est.style.format({"ARI promedio": "{:.3f}", "ARI mínimo": "{:.3f}"}), width="stretch", hide_index=True)
    st.markdown("<p class='nota'>Índice de Rand ajustado (ARI): 1 = misma partición; 0 = coincidencia por azar. Una silueta moderada no implica grupos inestables: "
                "la partición se repite casi igual con otras semillas y submuestras.</p>", unsafe_allow_html=True)

    st.markdown("### Segmentos en el plano CP1–CP2")
    pct = 100 * res.varianza[:2].sum()
    plot = pd.DataFrame({"CP1": X[:, 0], "CP2": X[:, 1], "Segmento": df.segmento})
    fig = px.scatter(plot, x="CP1", y="CP2", color="Segmento", opacity=0.5, color_discrete_map=colores,
                     title=f"Segmentos K-Means (K = {k}) · el plano CP1–CP2 resume el {num(pct,1)} % de la varianza")
    fig.update_traces(marker_size=4)
    st.plotly_chart(fig, width="stretch")
    if n_comp >= 3:
        plot["CP3"] = X[:, 2]
        fig = px.scatter_3d(plot, x="CP1", y="CP2", z="CP3", color="Segmento", opacity=0.6, color_discrete_map=colores,
                            title=f"Vista en tres componentes ({num(100*res.varianza[:3].sum(),1)} % de la varianza)")
        fig.update_traces(marker_size=2)
        st.plotly_chart(fig, width="stretch")

    st.markdown("### Calidad de la separación: silueta por hogar")
    orden = df.sort_values(["segmento", "silueta"], ascending=[True, False]).reset_index(drop=True)
    fig = px.bar(orden, x=orden.index, y="silueta", color="segmento", color_discrete_map=colores,
                 title="Silueta de cada hogar, agrupada por segmento (valores < 0 = más cerca de otro segmento)")
    fig.update_layout(xaxis_title="Hogares ordenados por segmento", yaxis_title="Silueta", bargap=0, xaxis_showticklabels=False)
    fig.update_traces(marker_line_width=0)
    st.plotly_chart(fig, width="stretch")
    st.markdown(f"<p class='nota'>{num(100*(df.silueta<0).mean(),1)} % de los hogares tienen silueta negativa: son casos frontera. "
                "Las etiquetas numéricas de K-Means no tienen orden; por eso cada segmento se nombra con las dos variables que más lo distinguen.</p>", unsafe_allow_html=True)
    st.caption(FUENTE)

# ----------------------------------------------------------------------------- 6. perfiles
elif pagina == "6. Perfiles de los segmentos":
    st.title("¿Quiénes están en cada segmento?")
    st.markdown("Los segmentos se describen con las **variables originales**, ponderadas con el factor de expansión. Medianas para montos; promedios para composición.")
    P = A.perfiles(df)
    cols_pesos = [c for c in P.columns if "(mediana)" in c and "Carga" not in c]
    mostrar = P.copy()
    for c in cols_pesos:
        mostrar[c] = mostrar[c].map(pesos)
    mostrar["Hogares en un mes promedio"] = mostrar["Hogares en un mes promedio"].map(num)
    for c in mostrar.columns:
        if c not in cols_pesos + ["segmento", "Hogares en un mes promedio"]:
            mostrar[c] = mostrar[c].map(lambda x: num(x, 0) if c == "Hogares-mes (n)" else num(x, 2 if "Silueta" in c else 1))
    st.dataframe(mostrar.set_index("segmento").T.astype(str), width="stretch", height=720)
    d1, d2 = st.columns(2)
    d1.download_button("Descargar perfiles (CSV)", P.to_csv(index=False).encode("utf-8-sig"), "perfiles_segmentos.csv", "text/csv")
    d2.download_button("Descargar hogares con su segmento (CSV)", df[["mes", "factor_expansion"] + variables + ["segmento", "silueta"]].to_csv(index=False).encode("utf-8-sig"),
                       "hogares_segmentados.csv", "text/csv")
    st.markdown("<p class='nota'>* Productos financieros: solo hogares que respondieron la pregunta P5222 (83 %). Margen = ingreso − canon; no es ahorro.</p>", unsafe_allow_html=True)

    st.markdown("### Huella de cada segmento (diferencia frente al promedio, en desviaciones estándar)")
    z = df[variables].apply(lambda s: (s - s.mean()) / s.std()).groupby(df.segmento).mean()
    z.columns = [A.etiqueta(c) for c in variables]
    fig = px.imshow(z, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1.5, zmax=1.5, aspect="auto",
                    title="Promedio estandarizado de cada variable por segmento")
    st.plotly_chart(fig, width="stretch")

    st.markdown("### La banda de 2 a 4 salarios mínimos no es un grupo homogéneo")
    cruce = df.groupby(["banda_smmlv", "segmento"], observed=True).factor_expansion.sum().reset_index()
    cruce["%"] = 100 * cruce.factor_expansion / cruce.groupby("banda_smmlv", observed=True).factor_expansion.transform("sum")
    fig = px.bar(cruce, x="banda_smmlv", y="%", color="segmento", color_discrete_map=colores,
                 title="Composición de cada banda de ingreso según segmento (% de hogares, ponderado)")
    fig.update_layout(xaxis_title="Ingreso del hogar en SMMLV de 2025", yaxis_title="% de hogares de la banda")
    st.plotly_chart(fig, width="stretch")

    st.markdown("### Comparar una variable entre segmentos")
    v = st.selectbox("Variable original", ["ingreso_pc", "carga_original", "canon", "margen", "personas", "hacinamiento", "edad_jefe", "formalidad"],
                     format_func=lambda x: {"ingreso_pc": "Ingreso por persona", "carga_original": "Carga del arriendo (%)", "canon": "Canon",
                                            "margen": "Margen después del arriendo", "personas": "Personas", "hacinamiento": "Personas por dormitorio",
                                            "edad_jefe": "Edad persona de referencia", "formalidad": "Perceptores que cotizan (%)"}[x])
    fig = px.box(df, x="segmento", y=v, color="segmento", color_discrete_map=colores, points=False,
                 title="Distribución por segmento (muestra, sin ponderar)")
    if v in ["ingreso_pc", "canon", "carga_original"]:
        fig.update_yaxes(type="log")
    fig.update_layout(showlegend=False, xaxis_title="Segmento")
    st.plotly_chart(fig, width="stretch")
    st.caption(FUENTE)

# ----------------------------------------------------------------------------- 7. conclusiones
else:
    st.title("Conclusiones")
    P = A.perfiles(df).set_index("segmento")
    if not es_defecto:
        st.warning("Cambiaste los parámetros. Las cifras se recalculan, pero el texto interpretativo fue escrito para la configuración por defecto (7 variables, recorte 100 %, componentes de Kaiser, K = 4, ponderado).")
    if k >= 4:
        presion = P["Carga del arriendo (mediana, %)"].idxmax()
        grande = P["Personas (promedio)"].idxmax()
        mayor = P["Edad persona de referencia"].idxmax()
        holgado = P["Ingreso por persona (mediana)"].idxmax()
        banda = df[df.banda_smmlv == "2–4 SMMLV"]
        comp = (banda.groupby("segmento").factor_expansion.sum() / banda.factor_expansion.sum() * 100).sort_values(ascending=False)
        f = lambda s, c: P.loc[s, c]
        st.markdown(f"""
### 1. ¿Qué define a los componentes?
El primer componente es **capacidad económica**: ingreso por persona, número de perceptores y cotización a pensión van juntos y en sentido opuesto a la carga del arriendo. El segundo separa **hogares grandes y hacinados** de hogares pequeños con más presión relativa. El tercero recoge el **ciclo de vida** (edad). Con {n_comp} componentes se conserva el **{num(100*var_ret,1)} %** de la variación de las {len(variables)} variables.

### 2. ¿Qué segmentos aparecen?
- **{holgado}**: {num(f(holgado,'% de arrendatarios'),1)} % de los arrendatarios. Hogares pequeños ({num(f(holgado,'Personas (promedio)'),1)} personas), ingreso por persona de {pesos(f(holgado,'Ingreso por persona (mediana)'))} y {num(f(holgado,'Perceptores que cotizan (%)'),0)} % de perceptores cotizando. Carga mediana de {num(f(holgado,'Carga del arriendo (mediana, %)'),1)} %.
- **{grande}**: {num(f(grande,'% de arrendatarios'),1)} %. Familias de {num(f(grande,'Personas (promedio)'),1)} personas y {num(f(grande,'Personas por dormitorio'),1)} por dormitorio. Su ingreso del hogar ({pesos(f(grande,'Ingreso del hogar (mediana)'))}) no es bajo, pero por persona cae a {pesos(f(grande,'Ingreso por persona (mediana)'))}.
- **{mayor}**: {num(f(mayor,'% de arrendatarios'),1)} %. Persona de referencia de {num(f(mayor,'Edad persona de referencia'),0)} años en promedio y solo {num(f(mayor,'Perceptores que cotizan (%)'),0)} % de perceptores cotizando. Carga mediana de {num(f(mayor,'Carga del arriendo (mediana, %)'),1)} %.
- **{presion}**: {num(f(presion,'% de arrendatarios'),1)} %. Ingreso del hogar de {pesos(f(presion,'Ingreso del hogar (mediana)'))}, carga mediana de **{num(f(presion,'Carga del arriendo (mediana, %)'),0)} %** y cotización casi nula ({num(f(presion,'Perceptores que cotizan (%)'),0)} %).

### 3. La misma banda de ingreso esconde situaciones distintas
De los hogares con ingreso entre 2 y 4 salarios mínimos, **{num(comp.iloc[0],0)} %** pertenecen a «{comp.index[0]}», **{num(comp.iloc[1],0)} %** a «{comp.index[1]}» y **{num(comp.iloc[2],0)} %** a «{comp.index[2]}». Dos hogares con el mismo ingreso total pueden tener un ingreso por persona que difiere en más del doble y una cotización muy diferente. Para una política de acceso a vivienda, esto significa que un subsidio definido solo por ingreso trataría igual a hogares con capacidades de pago y de acceso al crédito muy distintas.

**¿Qué implica cada perfil para la transición a vivienda propia?** (interpretación; la GEIH no mide ahorro ni crédito)
- *{holgado}*: los que más se acercan a las condiciones de financiación, por ingreso por persona y cotización.
- *{grande}*: el ingreso del hogar entra en la banda, pero repartido entre más personas limita la cuota que pueden pagar sin sacrificar consumo esencial.
- *{mayor}*: con baja cotización, probablemente les cuesta más demostrar su ingreso ante un banco, y tienen menos años de plazo para un crédito.
- *{presion}*: necesitan alivio del arriendo antes que apoyo para comprar.

### 4. ¿Qué tan claros son los grupos?
La silueta promedio es **{num(df.silueta.mean(),2)}** y el {num(100*(df.silueta<0).mean(),1)} % de los hogares son casos frontera. Los segmentos son útiles para describir perfiles, pero **no son categorías nítidas**: la población arrendataria se parece más a un continuo con zonas de concentración que a grupos aislados. Aun así, la partición es **estable**: se repite casi igual con otras semillas y con submuestras del 80 % de los hogares (sección 5). Los bordes son difusos, pero los perfiles no son un accidente del algoritmo.

### 5. Limitaciones
- La GEIH no observa ahorro, deudas, historial crediticio ni intención de compra: los segmentos describen **condiciones**, no elegibilidad para comprar.
- El ingreso es una reconstrucción propia. Los otros ingresos de 12 meses se promedian por mes, y {num(len(raw)-len(df))} hogares-mes con canon o ingreso incompleto, o ingreso cero, quedan fuera.
- El ACP se estima sobre la muestra (sin ponderar); los tamaños y perfiles sí se expanden con el factor de expansión. La GEIH no trae variables de diseño para calcular errores estándar.
- K-Means supone grupos de forma aproximadamente esférica y es sensible a valores extremos; por eso se recortó la carga del arriendo.
- Los resultados son descriptivos y no implican causalidad.
""")
    else:
        st.info("Con K < 4 no se distinguen los cuatro perfiles descritos en la interpretación. Usa K = 4 para leer las conclusiones.")
    st.caption(FUENTE)
