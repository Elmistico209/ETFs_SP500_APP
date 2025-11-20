# pages/01_Aula_de_estrategias_con_opciones.py
# Módulo educativo: Aula de estrategias con opciones

import os
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
import pandas as pd
import streamlit as st

try:
    import yfinance as yf
except Exception:
    yf = None

import plotly.graph_objects as go

# Opcional: cargar .env por si lo necesitas en un futuro aquí
try:
    from dotenv import load_dotenv
    for p in [Path.cwd()/".env", Path(__file__).resolve().parent.parent/".env"]:
        if p.exists():
            load_dotenv(p, override=False)
            break
except Exception:
    pass

def _normalize_ticker(t: str) -> str:
    t = (t or "").strip().upper()
    if t == "APPL":
        t = "AAPL"
    return t

st.set_page_config(page_title="Aula de estrategias con opciones", layout="wide")
st.title("Aula de estrategias con opciones (educativo)")

st.caption(
    "Ejemplos simplificados de payoff al vencimiento por acción. "
    "No son precios reales de mercado ni recomendaciones."
)

@st.cache_data(show_spinner=False, ttl=600)
def get_spot_price_options(ticker: str) -> float:
    """Obtiene un precio de referencia para el subyacente."""
    if yf is None:
        return 100.0
    try:
        t = yf.Ticker(_normalize_ticker(ticker))
        hist = t.history(period="5d", auto_adjust=True)
        if not hist.empty:
            px = float(hist["Close"].iloc[-1])
            if px > 0:
                return px
    except Exception:
        pass
    return 100.0

# Información base de cada estrategia
estrategias_info = {
    "Covered Call": {
        "objetivo": "Generar ingreso extra sobre una posición larga en acciones/ETF vendiendo calls OTM.",
        "estructura": "Largo 1 acción/ETF + corto 1 call out-of-the-money.",
        "riesgo": "Riesgo bajista similar a tener la acción; upside limitado al strike de la call + prima.",
        "cuando": "Visión neutral o ligeramente alcista; no te importa que te asignen/vendan el subyacente."
    },
    "Cash-Secured Put": {
        "objetivo": "Cobrar prima a cambio de estar dispuesto a comprar el activo a un precio inferior al actual.",
        "estructura": "Venta de 1 put OTM con efectivo suficiente para cubrir una posible asignación.",
        "riesgo": "Riesgo bajista similar a comprar las acciones; pérdida crece si el precio cae muy por debajo del strike.",
        "cuando": "Visión moderadamente alcista o quieres entrar al activo con 'descuento'."
    },
    "Protective Put": {
        "objetivo": "Proteger una posición larga mediante la compra de una put, creando un seguro a la baja.",
        "estructura": "Largo 1 acción/ETF + largo 1 put (ATM u OTM cercana).",
        "riesgo": "La pérdida se acota alrededor del strike menos el precio actual + prima, aunque pagas ese seguro.",
        "cuando": "Tienes una posición que no quieres vender pero te preocupa una corrección fuerte."
    },
    "Call Debit Spread (Bull Call)": {
        "objetivo": "Tomar una posición alcista con riesgo limitado y menor costo que una call desnuda.",
        "estructura": "Compra 1 call (strike bajo) + venta 1 call (strike alto), mismo vencimiento.",
        "riesgo": "Riesgo limitado a la prima neta pagada; beneficio máximo limitado entre strikes menos dicha prima.",
        "cuando": "Visión alcista moderada y quieres definir riesgo/beneficio desde el inicio."
    },
    "Bear Put Spread": {
        "objetivo": "Apostar a una caída moderada del subyacente con riesgo limitado.",
        "estructura": "Compra 1 put (strike alto) + venta 1 put (strike bajo), mismo vencimiento.",
        "riesgo": "Riesgo limitado a la prima neta pagada; beneficio máximo si el subyacente cae por debajo del strike bajo.",
        "cuando": "Visión bajista moderada y buscas una forma más eficiente que un simple short de acciones."
    },
    "Iron Condor": {
        "objetivo": "Cobrar prima esperando que el precio se mantenga dentro de un rango.",
        "estructura": "Venta de un put spread OTM + venta de un call spread OTM (mismo vencimiento).",
        "riesgo": "Riesgo limitado al ancho de los spreads menos el crédito recibido; sufre si el precio sale con fuerza del rango.",
        "cuando": "Visión neutral y volatilidad implícita relativamente alta; esperas menos movimiento del que descuenta el mercado."
    },
}

def payoff_profile(strategy: str, S: np.ndarray, S0: float) -> Tuple[np.ndarray, dict]:
    """
    Calcula payoff al vencimiento por acción para distintas estrategias.
    S: vector de posibles precios al vencimiento.
    S0: precio de referencia (spot).
    """
    S0 = max(S0, 1e-6)

    if strategy == "Covered Call":
        K = 1.05 * S0
        premium = 0.03 * S0
        stock_payoff = S - S0
        call_short = premium - np.maximum(S - K, 0)
        payoff = stock_payoff + call_short
        params = {
            "Strike call": K,
            "Prima recibida": premium,
            "Coste base acción": S0,
        }

    elif strategy == "Cash-Secured Put":
        K = 0.95 * S0
        premium = 0.03 * S0
        payoff = premium - np.maximum(K - S, 0)
        params = {
            "Strike put": K,
            "Prima recibida": premium,
        }

    elif strategy == "Protective Put":
        K = 0.95 * S0
        premium = 0.03 * S0
        stock_payoff = S - S0
        put_long = np.maximum(K - S, 0) - premium
        payoff = stock_payoff + put_long
        params = {
            "Strike put": K,
            "Prima pagada": premium,
            "Coste base acción": S0,
        }

    elif strategy == "Call Debit Spread (Bull Call)":
        K1 = 1.00 * S0
        K2 = 1.10 * S0
        c1 = 0.06 * S0
        c2 = 0.03 * S0
        long_call = np.maximum(S - K1, 0) - c1
        short_call = c2 - np.maximum(S - K2, 0)
        payoff = long_call + short_call
        params = {
            "Strike call larga": K1,
            "Strike call corta": K2,
            "Prima pagada call larga": c1,
            "Prima recibida call corta": c2,
        }

    elif strategy == "Bear Put Spread":
        K1 = 1.00 * S0
        K2 = 0.90 * S0
        p1 = 0.06 * S0
        p2 = 0.03 * S0
        long_put = np.maximum(K1 - S, 0) - p1
        short_put = p2 - np.maximum(K2 - S, 0)
        payoff = long_put + short_put
        params = {
            "Strike put larga": K1,
            "Strike put corta": K2,
            "Prima pagada put larga": p1,
            "Prima recibida put corta": p2,
        }

    elif strategy == "Iron Condor":
        Kp1 = 0.80 * S0
        Kp2 = 0.90 * S0
        Kc1 = 1.10 * S0
        Kc2 = 1.20 * S0
        credit_put_spread = 0.02 * S0
        credit_call_spread = 0.02 * S0

        short_put = credit_put_spread - np.maximum(Kp2 - S, 0) + np.maximum(Kp1 - S, 0)
        short_call = credit_call_spread - np.maximum(S - Kc1, 0) + np.maximum(S - Kc2, 0)

        payoff = short_put + short_call
        params = {
            "Strike put larga": Kp1,
            "Strike put corta": Kp2,
            "Strike call corta": Kc1,
            "Strike call larga": Kc2,
            "Crédito neto aproximado": credit_put_spread + credit_call_spread,
        }

    else:
        payoff = np.zeros_like(S)
        params = {}

    return payoff, params

# --- Controles de la sección ---
c1, c2 = st.columns([1.2, 1])

with c1:
    opt_ticker = st.text_input(
        "Ticker subyacente para el ejemplo (Yahoo)",
        value="SPY",
        key="opt_ticker_aula"
    )
    spot = get_spot_price_options(opt_ticker)
    st.metric("Precio de referencia (spot aprox.)", f"${spot:,.2f}")

    strategy_name = st.selectbox(
        "Estrategia a estudiar",
        options=list(estrategias_info.keys()),
        index=0,
        key="estrategia_aula"
    )

with c2:
    info = estrategias_info[strategy_name]
    st.markdown(f"### {strategy_name}")
    st.markdown(f"**Objetivo principal:** {info['objetivo']}")
    st.markdown(f"**Estructura típica:** {info['estructura']}")
    st.markdown(f"**Riesgo principal:** {info['riesgo']}")
    st.markdown(f"**Cuándo se suele usar:** {info['cuando']}")

# --- Cálculo del payoff y gráfico ---
if opt_ticker:
    try:
        S_min = max(0.3 * spot, 0.01)
        S_max = 1.7 * spot
        S_grid = np.linspace(S_min, S_max, 220)

        payoff, params = payoff_profile(strategy_name, S_grid, spot)

        max_gain = float(np.max(payoff))
        max_loss = float(np.min(payoff))

        m1, m2 = st.columns(2)
        m1.metric("Ganancia máxima (aprox., por acción)", f"${max_gain:,.2f}")
        m2.metric("Pérdida máxima (aprox., por acción)", f"${max_loss:,.2f}")

        fig_opt = go.Figure()
        fig_opt.add_trace(go.Scatter(
            x=S_grid,
            y=payoff,
            mode="lines",
            name="Payoff al vencimiento (por acción)",
            line=dict(color="#ff6f00", width=2)
        ))

        fig_opt.add_hline(
            y=0,
            line=dict(color="rgba(120,120,120,0.7)", width=1, dash="dash"),
            annotation_text="Break-even",
            annotation_position="bottom left"
        )

        fig_opt.add_vline(
            x=spot,
            line=dict(color="#2196f3", width=1, dash="dot"),
            annotation_text="Spot",
            annotation_position="top right"
        )

        fig_opt.update_layout(
            title=f"Perfil de payoff al vencimiento — {strategy_name} sobre {opt_ticker.upper()}",
            xaxis_title="Precio del subyacente al vencimiento",
            yaxis_title="Payoff por acción",
            hovermode="x unified",
            margin=dict(l=10, r=10, t=50, b=40)
        )

        st.plotly_chart(fig_opt, use_container_width=True)

        if params:
            st.markdown("**Supuestos del ejemplo (por acción):**")
            df_params = pd.DataFrame(
                {"Parámetro": list(params.keys()), "Valor": list(params.values())}
            )
            st.dataframe(
                df_params,
                use_container_width=True,
                height=min(260, 40 * (len(df_params) + 1))
            )

    except Exception as e:
        st.error(f"No fue posible generar el perfil de payoff: {e}")

# --- Fuentes para estudiar más ---
st.subheader("Fuentes recomendadas para seguir aprendiendo opciones")
st.markdown(
    "- **Tastytrade / Tastylive** — contenido educativo sobre estrategias con opciones.\n"
    "- **Cboe Options Institute** — material formativo del principal mercado de opciones de EE.UU.\n"
    "- **The Options Industry Council (OIC)** — cursos y glosarios sobre opciones.\n"
    "- **Investopedia** — artículos introductorios sobre opciones y estrategias.\n"
    "- Documentación y material educativo de tu broker (simuladores, PDFs y webinars)."
)

st.caption(
    "Este módulo es estrictamente educativo. Los ejemplos de strikes y primas son supuestos "
    "simplificados y no reflejan precios reales de mercado. No constituye asesoría financiera."
)
