from streamlit.testing.v1 import AppTest
import sys
pages=["1. Contexto","2. Exploración de datos","3. Preparación","4. Componentes principales","5. Segmentación K-Means","6. Perfiles de los segmentos","7. Conclusiones"]
for p in pages:
    at=AppTest.from_file("../app.py",default_timeout=120)
    at.run()
    at.sidebar.radio[0].set_value(p).run()
    errs=[e.value for e in at.exception]
    print(p, "EXC" if errs else "ok", errs[:1], "| warnings:", [w.value[:80] for w in at.warning])
    if p.startswith("7"):
        print("\n".join(m.value[:1200] for m in at.markdown if "segmentos aparecen" in m.value))
    if p.startswith("5"):
        print([m.value[:500] for m in at.markdown if "elige" in m.value])
