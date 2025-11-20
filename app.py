# app.py
# app ingenieria financiera — Yahoo Finance + Finviz + Plotly (sin SARIMAX)
# Secciones: Sectores, Resumen+Velas+Finviz, Pronósticos, Backtesting
# + IA explicando secciones + Asistente IA de preguntas sobre los datos
# -----------------------------------------------------------------------------
# Requisitos (requirements.txt):
# streamlit>=1.34,<2
# yfinance>=0.2.43
# pandas>=2.2
# numpy>=1.26,<3
# requests>=2.31
# python-dotenv>=1.0.1
# beautifulsoup4>=4.12
# lxml>=4.9.3
# plotly>=5.22
# google-genai>=0.3.0        # (y/o) google-generativeai>=0.7.0
# -----------------------------------------------------------------------------

import os
import re
import time
import math
import json
import unicodedata
from datetime import date
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

# Plotly
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ====== .env (opcional) ======
try:
    from dotenv import load_dotenv
    for p in [Path.cwd()/".env", Path(__file__).resolve().parent/".env"]:
        if p.exists():
            load_dotenv(p, override=False)
            break
except Exception:
    pass

# ====== yfinance ======
try:
    import yfinance as yf
except Exception:
    yf = None

# ====== Gemini SDKs (opcional) ======
_GENAI_NEW = None
_GENAI_OLD = None
try:
    from google import genai as _genai_new
    _GENAI_NEW = _genai_new
except Exception:
    _GENAI_NEW = None
try:
    import google.generativeai as _genai_old
    _GENAI_OLD = _genai_old
except Exception:
    _GENAI_OLD = None

# ====== BeautifulSoup (Finviz) ======
import requests
from bs4 import BeautifulSoup

# ========= Config =========
st.set_page_config(page_title="app ingenieria financiera", layout="wide")
st.title("App ingenieria financiera")

DEFAULT_TICKERS = ["XLY", "XLP", "XLE", "XLF", "XLV", "XLI", "XLK", "XLU"]
SECTOR_LABELS = {
    "XLY": "Consumo Discrec.", "XLP": "Consumo Básico", "XLE": "Energía", "XLF": "Financiero",
    "XLV": "Salud", "XLI": "Industriales", "XLK": "Tecnología", "XLU": "Utilities"
}
BENCH = "SPY"

# ========= Utilidades de mercado =========
@st.cache_data(show_spinner=True)
def _download_block(tickers: List[str], start: date, end: date, field: str) -> pd.DataFrame:
    """
    Descarga precios con yfinance y devuelve un DataFrame ancho con columnas=tickers.
    field: 'Close', 'Adj Close', 'Volume', etc.
    """
    if yf is None:
        raise ImportError("Instala yfinance: pip install yfinance")
    try:
        df = yf.download(
            tickers, start=start, end=end,
            interval="1d", auto_adjust=True,
            progress=False, threads=True, actions=False,
        )
    except Exception:
        df = pd.DataFrame()

    def _extract_field(df_):
        if df_ is None or df_.empty:
            return None
        if not isinstance(df_.columns, pd.MultiIndex):
            if field in df_.columns:
                out = df_[[field]].copy()
                if isinstance(tickers, list) and len(tickers) == 1:
                    out.columns = [tickers[0]]
                return out
            return None
        if field in df_.columns.get_level_values(0):
            return df_.xs(field, axis=1, level=0)
        if field in df_.columns.get_level_values(1):
            return df_.xs(field, axis=1, level=1)
        return None

    out = _extract_field(df)
    if out is not None and not out.empty:
        keep = [t for t in tickers if t in out.columns]
        return out[keep].dropna(how="all")

    # Fallback por ticker
    frames = []
    for t in tickers:
        try:
            h = yf.Ticker(t).history(start=start, end=end, interval="1d", auto_adjust=True)
            if not h.empty and field in h.columns:
                frames.append(h[field].rename(t))
        except Exception:
            pass
    if frames:
        return pd.concat(frames, axis=1).dropna(how="all")
    return pd.DataFrame()

def _annualize_vol(std_daily: float, periods_per_year: int = 252) -> float:
    return std_daily * math.sqrt(periods_per_year)

def _max_drawdown(series: pd.Series) -> float:
    if series.empty:
        return np.nan
    cummax = series.cummax()
    dd = series / cummax - 1.0
    return float(dd.min())

def _normalize_ticker(t: str) -> str:
    t = (t or "").strip().upper()
    if t == "APPL":
        t = "AAPL"
    return t

# ========= Indicadores técnicos =========
def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    line = ema_fast - ema_slow
    sig = line.ewm(span=signal, adjust=False).mean()
    hist = line - sig
    return line, sig, hist

# --- OHLCV helpers para Plotly ---
def prepare_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    cols = ["Open", "High", "Low", "Close", "Volume"]
    df = df.copy()
    cols_presentes = [c for c in cols if c in df.columns]
    df = df[cols_presentes]
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, errors="coerce")
    df = df.dropna()
    if hasattr(df.index, "tz"):
        df.index = df.index.tz_localize(None)
    return df.sort_index()

# ========= Beta (CAPM) =========
def _beta_vs_market(ri: pd.Series, rm: pd.Series) -> float:
    """
    Calcula beta como la pendiente de la regresión OLS: ri = alpha + beta * rm.
    Devuelve np.nan si no hay datos suficientes.
    """
    ri = pd.to_numeric(ri, errors="coerce").dropna()
    rm = pd.to_numeric(rm, errors="coerce").dropna()
    df = pd.concat([ri.rename("ri"), rm.rename("rm")], axis=1).dropna()
    if len(df) < 20 or df["rm"].std() == 0:
        return float("nan")
    # Pendiente por polyfit (ri en función de rm)
    beta, alpha = np.polyfit(df["rm"].values, df["ri"].values, deg=1)
    return float(beta)

# ========= Finviz scraping =========
FINVIZ_URL = "https://finviz.com/quote.ashx?t={ticker}&p=d"
_session_finviz = requests.Session()
_session_finviz.headers.update({
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                   "AppleWebKit/537.36 (KHTML, like Gecko) "
                   "Chrome/124.0 Safari/537.36")
})

def _norm_text_bs(s: Optional[str]) -> str:
    if s is None:
        return ""
    s = unicodedata.normalize("NFKC", s)
    s = s.replace("\xa0", " ").strip()
    s = re.sub(r"\s+", " ", s)
    return s

@st.cache_data(show_spinner=False, ttl=600)
def _finviz_snapshot_raw(ticker: str) -> dict:
    url = FINVIZ_URL.format(ticker=_normalize_ticker(ticker))
    resp = _session_finviz.get(url, timeout=12)
    if resp.status_code != 200:
        raise RuntimeError(f"Finviz HTTP {resp.status_code}")
    soup = BeautifulSoup(resp.text, "lxml")
    table = soup.find("table", class_="snapshot-table2")
    if not table:
        raise RuntimeError("Tabla snapshot-table2 no encontrada (Finviz).")
    data = {}
    for row in table.find_all("tr"):
        tds = row.find_all("td")
        for i in range(0, len(tds) - 1, 2):
            key = _norm_text_bs(tds[i].get_text(strip=True))
            val = _norm_text_bs(tds[i+1].get_text(strip=True))
            data[key] = val
    return data

def _to_number_bs(x: Optional[str]):
    if x is None:
        return None
    s = _norm_text_bs(x).replace(",", "")
    if s in {"-", "—", "N/A", ""}:
        return None
    if s.endswith("%"):
        try:
            return float(s[:-1]) / 100.0
        except ValueError:
            return None
    m = re.match(r"^([+-]?\d*\.?\d+)\s*([KMBT])?$", s, re.I)
    if m:
        val = float(m.group(1))
        mult = {"K": 1e3, "M": 1e6, "B": 1e9, "T": 1e12}.get((m.group(2) or "").upper(), 1.0)
        return val * mult
    try:
        return float(s)
    except ValueError:
        return s

def finviz_blocks(ticker: str) -> Dict[str, Dict[str, Any]]:
    raw = _finviz_snapshot_raw(ticker)
    def getn(k): return _to_number_bs(raw.get(k))
    def get(k):  return raw.get(k)

    basics = {
        "Market Cap": getn("Market Cap"),
        "Enterprise Value": getn("Enterprise Value"),
        "Income": getn("Income"),
        "Sales": getn("Sales"),
    }
    valuation = {
        "P/E": getn("P/E"), "Forward P/E": getn("Forward P/E"), "PEG": getn("PEG"),
        "EV/Sales": getn("EV/Sales"),
        "P/B": getn("P/B"), "P/S": getn("P/S"),
        # "P/FCF": getn("P/FCF"),  # habilitar si Finviz lo expone para el ticker
    }
    growth = {
        "ROE": getn("ROE"), "ROA": getn("ROA"),
        "Gross Margin": getn("Gross Margin"), "Oper. Margin": getn("Oper. Margin"),
        "Profit Margin": getn("Profit Margin"),
        "EPS next Y": getn("EPS next Y"), "EPS next Q": getn("EPS next Q"),
        "EPS this Y": getn("EPS this Y"), "EPS Y/Y TTM": getn("EPS Y/Y TTM"),
        "Sales Y/Y TTM": getn("Sales Y/Y TTM"),
    }
    structure = {
        "Debt/Eq": getn("Debt/Eq"), "LT Debt/Eq": getn("LT Debt/Eq"),
        "Current Ratio": getn("Current Ratio"), "Quick Ratio": getn("Quick Ratio"),
    }
    dividends = {"Payout": getn("Payout")}
    dv_ttm = get("Dividend TTM")
    if dv_ttm:
        m_amt = re.search(r"([+-]?\d*\.?\d+)", dv_ttm)
        m_y   = re.search(r"\(([^)]*)\)", dv_ttm)
        dividends["Dividend TTM (amount)"] = float(m_amt.group(1)) if m_amt else None
        dividends["Dividend TTM (yield)"]  = _to_number_bs(m_y.group(1)) if m_y else None
    performance = {
        "Perf YTD": getn("Perf YTD"), "Perf Year": getn("Perf Year"),
        "Perf 3Y": getn("Perf 3Y"), "Perf 5Y": getn("Perf 5Y"),
        "Beta": getn("Beta"), "Price": getn("Price"), "Prev Close": getn("Prev Close"),
    }
    return {
        "Básicos": basics, "Valuación": valuation, "Crecimiento y Rentabilidad": growth,
        "Estructura": structure, "Dividendos": dividends, "Desempeño": performance
    }

def valuation_diagnosis(blocks: Dict[str, Dict[str, Any]], ticker: str) -> dict:
    t = _normalize_ticker(ticker)
    V = blocks["Valuación"]; G = blocks["Crecimiento y Rentabilidad"]
    pe = V.get("P/E"); fpe = V.get("Forward P/E"); peg = V.get("PEG")
    ps = V.get("P/S"); pb = V.get("P/B"); pfcf = V.get("P/FCF")
    roe = G.get("ROE"); opm = G.get("Oper. Margin"); pm = G.get("Profit Margin")
    score = 0; reasons = []
    if peg is not None:
        if peg <= 1.0: score -= 2; reasons.append(f"PEG={peg:.2f} (crecimiento a precio atractivo)")
        elif peg >= 1.5: score += 2; reasons.append(f"PEG={peg:.2f} (precio > crecimiento)")
    if pe is not None:
        if pe <= 20: score -= 1; reasons.append(f"P/E={pe:.1f} (razonable)")
        elif pe >= 35: score += 2; reasons.append(f"P/E={pe:.1f} (elevado)")
    if fpe is not None and pe is not None and fpe > pe*1.05:
        score += 1; reasons.append(f"Forward P/E {fpe:.1f} > P/E {pe:.1f} (expansión múltiplo)")
    if ps is not None and ps >= 10: score += 1; reasons.append(f"P/S={ps:.1f} (ventas caras)")
    if pb is not None and pb >= 6: score += 1; reasons.append(f"P/B={pb:.1f} (valor contable caro)")
    if pfcf is not None and pfcf >= 40: score += 1; reasons.append(f"P/FCF={pfcf:.1f} (flujo libre caro)")
    quality_bonus = 0
    if roe is not None and roe >= 0.20: quality_bonus -= 1; reasons.append(f"ROE={roe:.0%} (calidad alta)")
    if opm is not None and opm >= 0.25: quality_bonus -= 1; reasons.append(f"Margen operativo={opm:.0%}")
    if pm is not None and pm >= 0.20: quality_bonus -= 0.5; reasons.append(f"Margen neto={pm:.0%}")
    score += quality_bonus
    verdict = "Sobrevaluada" if score >= 3 else ("No sobrevaluada / razonable" if score <= -1 else "Neutral (mixta / en línea)")
    return {"ticker": t, "score": round(score, 2), "veredicto": verdict, "razones": reasons}

def render_valuation_verdict(diag: dict):
    verdict = diag.get("veredicto", "Neutral")
    score   = diag.get("score", 0.0)
    color_map = {
        "Sobrevaluada": ("#8B0000", "#FDEAEA"),
        "No sobrevaluada / razonable": ("#0F5132", "#E7F6EE"),
        "Neutral (mixta / en línea)": ("#664D03", "#FFF3CD"),
    }
    fg, bg = color_map.get(verdict, ("#1F1F1F", "#F1F1F1"))
    html = f"""
    <div style="border:1px solid {fg}; background:{bg}; color:{fg};
        padding:14px 16px; border-radius:10px; font-family:Inter,system-ui,Segoe UI,Arial;">
        <div style="display:flex; justify-content:space-between; align-items:center;">
            <div style="font-size:18px; font-weight:700;">Veredicto de valuación</div>
            <div style="font-size:14px; font-weight:600;">Score: {score:+.2f}</div>
        </div>
        <div style="margin-top:4px; font-size:16px;">{verdict}</div>
    </div>"""
    st.markdown(html, unsafe_allow_html=True)

# ========= Gráfico simple de líneas =========
def fig_line(df: pd.DataFrame, ytitle: str, title: str) -> go.Figure:
    fig = go.Figure()
    for col in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df[col], mode="lines", name=col))
    fig.update_layout(
        title=title, hovermode="x unified",
        legend=dict(orientation="h", y=-0.2),
        margin=dict(l=10, r=10, t=40, b=40)
    )
    fig.update_yaxes(title=ytitle)
    fig.update_xaxes(rangeslider=dict(visible=True), rangeselector=dict(
        buttons=list([
            dict(count=1, label="1M", step="month", stepmode="backward"),
            dict(count=3, label="3M", step="month", stepmode="backward"),
            dict(count=6, label="6M", step="month", stepmode="backward"),
            dict(count=1, label="1Y", step="year", stepmode="backward"),
            dict(step="all")
        ])
    ))
    return fig

# =============================================================================
# #Sectores
# =============================================================================
st.header("Sectores")

# Controles
c1, c2, c3 = st.columns([1.3, 1, 1])
with c1:
    lookback = st.selectbox("Años de historial (series y métricas)", [5, 10, 15, 20, "Máx"], index=1)
with c2:
    selected = st.radio(
        "ETF sectorial",
        options=DEFAULT_TICKERS,
        index=0, horizontal=True,
        format_func=lambda x: f"{x} · {SECTOR_LABELS.get(x, x)}"
    )
with c3:
    compare_spy = st.toggle("Comparar con SPY", value=True)

TODAY = date.today()
start_date = date(2000, 1, 1) if lookback == "Máx" else date(max(2000, TODAY.year - int(lookback)), 1, 1)
invest = st.number_input("Monto a invertir (simulador YTD)", min_value=0.0, value=10000.0, step=100.0)

tickers = [selected] + ([BENCH] if compare_spy and selected != BENCH else [])

# Descarga
with st.spinner("Descargando datos desde Yahoo Finance..."):
    px = _download_block(tickers, start_date, TODAY, field="Close").ffill()
    vol = _download_block(tickers, start_date, TODAY, field="Volume").ffill()

if px.empty:
    st.error("No se obtuvieron precios. Cambia rango o revisa red.")
    st.stop()

ret = px.pct_change().dropna(how="all")
cumret = (1 + ret).cumprod() - 1
ann_returns = ret.resample("Y").apply(lambda x: (1 + x).prod() - 1)
ann_returns.index = ann_returns.index.year

# Métricas rápidas + simulador (incluye BETA vs SPY)
st.subheader("Métricas rápidas")

def metrics_box(tkr: str, rm: Optional[pd.Series] = None):
    p = px[tkr].dropna()
    r = ret[tkr].dropna()
    years = (p.index[-1] - p.index[0]).days / 365.25
    cagr = (p.iloc[-1] / p.iloc[0]) ** (1 / max(years, 1e-9)) - 1 if len(p) > 1 else np.nan
    vol_ann = _annualize_vol(r.std()) if not r.empty else np.nan
    mdd = _max_drawdown((1 + r).cumprod())

    # --- Beta vs mercado (si se proporciona rm) ---
    beta_val = np.nan
    if rm is not None and not r.empty:
        rr = pd.concat([r.rename("ri"), rm.rename("rm")], axis=1).dropna()
        if len(rr) >= 20:
            beta_val = _beta_vs_market(rr["ri"], rr["rm"])

    # YTD y proyección basada en YTD
    start_ytd = date(TODAY.year, 1, 1)
    pr_ytd = p[p.index.date >= start_ytd]
    r_ytd = r[r.index.date >= start_ytd]
    if len(pr_ytd) >= 2:
        ytd = float(pr_ytd.iloc[-1] / pr_ytd.iloc[0] - 1)
        exp_full = (1 + r_ytd.mean()) ** 252 - 1
    else:
        ytd, exp_full = np.nan, np.nan

    # Tarjetas: agrega columna para Beta
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(f"{tkr} · CAGR", f"{cagr:,.2%}")
    c2.metric("Vol anual", f"{vol_ann:,.2%}")
    c3.metric("Máx. DD", f"{mdd:,.2%}")
    c4.metric("Beta vs SPY", "N/D" if pd.isna(beta_val) else f"{beta_val:,.2f}")

    d1, d2, d3 = st.columns(3)
    d1.metric("YTD", f"{(ytd if pd.notna(ytd) else 0):,.1%}" if pd.notna(ytd) else "N/D")
    d2.metric("Proy. anual (YTD)", f"{(exp_full if pd.notna(exp_full) else 0):,.1%}" if pd.notna(exp_full) else "N/D")
    if pd.notna(exp_full):
        gain = invest * exp_full
        final = invest + gain
        d3.metric("Ganancia aprox.", f"${gain:,.0f}")
        st.caption(f"Monto final estimado: **${final:,.0f}**")

cols = st.columns(2)
with cols[0]:
    rm_arg = ret[BENCH] if (BENCH in ret.columns and compare_spy and selected != BENCH) else None
    metrics_box(selected, rm=rm_arg)
with cols[1]:
    if BENCH in px.columns and compare_spy:
        rm_bench = ret[BENCH] if BENCH in ret.columns else None
        metrics_box(BENCH, rm=rm_bench)   # beta de SPY vs SPY ≈ 1
    else:
        st.info("Activa *Comparar con SPY* para ver sus métricas.")

# Gráficos interactivos (Plotly)
st.subheader("Series de tiempo (interactivo)")
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Precio normalizado", "Precio (cierre)", "Volumen", "Retorno diario (%)", "Rendimiento acumulado (%)"]
)
with tab1:
    base0 = (px.divide(px.iloc[0]) - 1.0) * 100.0
    st.plotly_chart(
        fig_line(base0, "Variación (%)", "Precio normalizado (base 0%)"),
        use_container_width=True
    )
with tab2:
    st.plotly_chart(fig_line(px, "Precio", "Precio de cierre"), use_container_width=True)
with tab3:
    if vol.empty:
        st.info("No hay volumen para este rango.")
    else:
        st.plotly_chart(fig_line(vol, "Volumen", "Volumen negociado"), use_container_width=True)
with tab4:
    st.plotly_chart(fig_line(ret*100, "Retorno diario (%)", "Retorno diario (%)"), use_container_width=True)
with tab5:
    st.plotly_chart(fig_line(cumret*100, "Rend. acumulado (%)", "Rendimiento acumulado (%)"), use_container_width=True)

# Tabla de rendimientos anuales
st.subheader("Rendimientos por año (%)")
annual_pct = (ann_returns * 100).round(2)
annual_pct.index.name = "Año"
st.dataframe(
    annual_pct.style.set_properties(**{"padding": "8px"}),
    use_container_width=True,
    height=min(480, 52*(len(annual_pct)+1))
)
st.download_button("⬇️ CSV Rendimientos anuales", data=ann_returns.to_csv(index=True).encode(),
                   file_name="annual_returns.csv", mime="text/csv")
st.caption("Fuente de datos: Yahoo Finance (vía yfinance).")

# =============================================================================
# Resumen del activo, velas e indicadores (Finviz)
# =============================================================================
st.header("Resumen del activo, velas e indicadores (Finviz)")

def get_api_key_default() -> str:
    key = os.getenv("GEMINI_API_KEY", "") or os.getenv("GOOGLE_API_KEY", "")
    try:
        if not key and "GEMINI_API_KEY" in st.secrets:
            key = st.secrets["GEMINI_API_KEY"] or key
    except Exception:
        pass
    return key

def list_gemini_models(api_key: str) -> List[str]:
    base = ["gemini-2.5-flash", "gemini-1.5-flash", "gemini-1.5-flash-8b", "gemini-1.5-pro"]
    if not api_key:
        return base
    try:
        from google import genai as _genai_new_check
        client = _genai_new_check.Client()
        names = []
        for m in client.models.list():
            name = getattr(m, "name", "")
            if "/" in name:
                name = name.split("/")[-1]
            names.append(name)
        return names or base
    except Exception:
        try:
            import google.generativeai as _genai_old_check
            _genai_old_check.configure(api_key=api_key)
            names = []
            for m in _genai_old_check.list_models():
                name = getattr(m, "name", "")
                if "/" in name:
                    name = name.split("/")[-1]
                names.append(name)
            return names or base
        except Exception:
            return base

def generate_with_gemini(api_key: str, model_name: str, prompt: str) -> str:
    try:
        from google import genai as _genai_new_use
        client = _genai_new_use.Client()
        resp = client.models.generate_content(model=model_name, contents=prompt)
        return getattr(resp, "text", "") or ""
    except Exception:
        import google.generativeai as _genai_old_use
        _genai_old_use.configure(api_key=api_key)
        model = _genai_old_use.GenerativeModel(model_name)
        resp = model.generate_content(prompt)
        return getattr(resp, "text", "") or ""

def force_exact_words_spanish(text: str, n: int = 500) -> str:
    text = re.sub(r"[•\-\*]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    words = text.split(" ") if text else []
    if len(words) == n:
        return text
    if len(words) > n:
        return " ".join(words[:n])
    filler = (" Esta síntesis se limita a los datos disponibles, "
              "está redactada en español y no constituye asesoría financiera.")
    filler_words = filler.strip().split(" ")
    while len(words) < n:
        take = min(n - len(words), len(filler_words))
        words.extend(filler_words[:take])
    return " ".join(words)

def gemini_summarize_ticker(ticker: str, api_key: str, model_name: str) -> Tuple[str, dict]:
    if not api_key:
        raise ValueError("Falta la API Key de Gemini/Google.")
    T = yf.Ticker(_normalize_ticker(ticker))
    raw = {"ticker": _normalize_ticker(ticker)}
    try:
        info = T.info
    except Exception:
        info = {}
    raw["info"] = {
        "longName": info.get("longName"),
        "shortName": info.get("shortName"),
        "category": info.get("category"),
        "fundFamily": info.get("fundFamily"),
        "annualReportExpenseRatio": info.get("annualReportExpenseRatio"),
        "totalAssets": info.get("totalAssets"),
        "yield": info.get("yield"),
        "navPrice": info.get("navPrice"),
        "underlyingIndex": info.get("underlyingIndex"),
        "website": info.get("website"),
        "longBusinessSummary": info.get("longBusinessSummary"),
    }

    try:
        hist = T.history(period="max", auto_adjust=True)
    except Exception:
        hist = pd.DataFrame(index=pd.DatetimeIndex([]))

    def ret_period_days(series: pd.Series, days: int):
        if series is None or series.empty or len(series) < 2:
            return None
        if len(series) <= days:
            return float(series.iloc[-1] / series.iloc[0] - 1)
        return float(series.iloc[-1] / series.iloc[-days] - 1)

    if not hist.empty and "Close" in hist.columns:
        close = hist["Close"]
    elif not hist.empty:
        close = hist.iloc[:, 0]
    else:
        close = pd.Series(dtype=float)

    ytd = None
    if not close.empty:
        start_y = date.today().replace(month=1, day=1)
        pr_ytd = close[close.index.date >= start_y]
        if len(pr_ytd) >= 2:
            ytd = float(pr_ytd.iloc[-1] / pr_ytd.iloc[0] - 1)

    r1 = ret_period_days(close, 252)
    r3 = ret_period_days(close, 252*3)
    r5 = ret_period_days(close, 252*5)
    raw["returns"] = {"YTD": ytd, "1Y": r1, "3Y": r3, "5Y": r5}

    # (Opcional) Calcular beta vs SPY y adjuntarla al JSON usado por el resumen
    try:
        start_beta = date(max(2000, date.today().year - 1), 1, 1)
        prices_beta = _download_block([_normalize_ticker(ticker), BENCH], start_beta, date.today(), field="Close").ffill().dropna()
        r_all = prices_beta.pct_change().dropna()
        if set([_normalize_ticker(ticker), BENCH]).issubset(r_all.columns):
            raw["beta_vs_SPY"] = _beta_vs_market(r_all[_normalize_ticker(ticker)], r_all[BENCH])
        else:
            raw["beta_vs_SPY"] = None
    except Exception:
        raw["beta_vs_SPY"] = None

    prompt = f"""
Eres un analista financiero. Redacta en español un resumen exclusivamente con los datos proporcionados (no inventes).
Usa párrafos corridos, sin viñetas ni encabezados. Longitud objetivo: entre 520 y 560 palabras (después se ajustará a 500).
Incluye: nombre del instrumento, sector/categoría, proveedor, índice subyacente (si aplica), expense ratio, activos,
rendimientos YTD/1Y/3Y/5Y (si faltan, indícalo como N/D) y una breve interpretación de riesgos/volatilidad.
Datos crudos:
{raw}
""".strip()

    text = ""
    for m in [model_name, "gemini-2.5-flash", "gemini-1.5-flash", "gemini-1.5-flash-8b"]:
        try:
            text = generate_with_gemini(api_key, m, prompt)
            if text:
                break
        except Exception:
            time.sleep(1)
            continue
    if not text:
        raise RuntimeError("No fue posible generar el texto (modelo/cuota).")
    return force_exact_words_spanish(text, n=500), raw

def ai_short_section_summary(
    section_name: str,
    context: dict,
    api_key: str,
    model_name: str,
    max_words: int = 160,
) -> str:
    if not api_key:
        return "⚠️ Configura primero tu API Key de Gemini/Google."
    safe_ctx = json.dumps(context, default=str)[:6000]
    prompt = f"""
Eres un analista cuantitativo que explica resultados de una app de ingeniería financiera.
Sección: {section_name}
Objetivo:
- Explicar en lenguaje sencillo qué significan los datos.
- Máx. {max_words} palabras.
- Sin viñetas, en 1–2 párrafos.
- No des recomendaciones directas de compra/venta ni frases tipo "debes hacer".
Contexto numérico y tablas (JSON):
{safe_ctx}
""".strip()
    try:
        text = generate_with_gemini(api_key, model_name, prompt)
    except Exception as e:
        text = f"No se pudo generar el resumen: {e}"
    if not text:
        text = "No fue posible generar un resumen con la IA en este momento."
    return text

# === Configuración segura de la API Key (SOLO desde .env / st.secrets) ===
api_key_default = get_api_key_default()
gemini_api_key = api_key_default or ""

with st.expander("⚙️ Estado de la IA (Gemini)", expanded=False):
    if gemini_api_key:
        st.success("Gemini API Key cargada desde .env / st.secrets. La clave NO se muestra por seguridad.")
        st.caption("Si compartes esta app, los usuarios podrán usar la IA pero nunca verán tu clave.")
    else:
        st.error("No se encontró GEMINI_API_KEY en .env ni en st.secrets. Los módulos de IA no funcionarán.")

# Modelos disponibles (no se expone la key)
models = list_gemini_models(gemini_api_key.strip() if gemini_api_key else "")
preferred = "gemini-2.5-flash" if "gemini-2.5-flash" in models else (
    "gemini-1.5-flash" if "gemini-1.5-flash" in models else models[0]
)
model_name = st.selectbox("Modelo de Gemini", options=models, index=models.index(preferred))
ticker_to_describe = st.text_input("Ticker (Yahoo) para el resumen / velas / Finviz", value=selected)

if st.button("Generar resumen con Gemini"):
    try:
        summary_text, raw_used = gemini_summarize_ticker(ticker_to_describe, gemini_api_key.strip(), model_name)
        st.success("Resumen generado (500 palabras)")
        st.write(summary_text)
        with st.expander("Datos usados (Yahoo)"):
            st.json(raw_used)
    except Exception as e:
        st.error(f"Error al generar el resumen: {e}")

# =============================================================================
# Velas + Indicadores (EMAs 20/21, RSI, MACD)
# =============================================================================
if ticker_to_describe:
    tt = _normalize_ticker(ticker_to_describe)
    st.subheader(f"Velas de {tt} (últimos 365 días)")

    # Controles de visualización
    cema1, cema2, cper = st.columns([1, 1, 2])
    with cema1:
        show_ema20 = st.toggle("Mostrar EMA 20", value=True)
    with cema2:
        show_ema21 = st.toggle("Mostrar EMA 21", value=True)
    with cper:
        rsi_period = st.slider("Periodo RSI", min_value=5, max_value=30, value=14, step=1)

    try:
        T = yf.Ticker(tt)
        hist = T.history(period="max", auto_adjust=False)
        if hasattr(hist.index, "tz"):
            hist.index = hist.index.tz_localize(None)
        last_year = (pd.Timestamp.today().tz_localize(None) - pd.Timedelta(days=365))
        hist_1y = hist.loc[hist.index >= last_year]
        dfv = prepare_ohlcv(hist_1y)

        if dfv.empty or any(c not in dfv.columns for c in ["Open", "High", "Low", "Close"]):
            st.info("No hay datos OHLCV válidos.")
        else:
            # EMAs y técnicos
            dfv["EMA20"] = dfv["Close"].ewm(span=20, adjust=False).mean()
            dfv["EMA21"] = dfv["Close"].ewm(span=21, adjust=False).mean()
            dfv["RSI"] = rsi(dfv["Close"], period=rsi_period)
            macd_line, macd_sig, macd_hist = macd(dfv["Close"])
            dfv["MACD_line"], dfv["MACD_signal"], dfv["MACD_hist"] = macd_line, macd_sig, macd_hist

            # Subplots: Velas + RSI + MACD
            fig = make_subplots(
                rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.035,
                row_heights=[0.50, 0.28, 0.22],
                specs=[[{"secondary_y": True}], [{"secondary_y": False}], [{"secondary_y": False}]]
            )

            # Velas
            fig.add_trace(go.Candlestick(
                x=dfv.index,
                open=dfv["Open"], high=dfv["High"], low=dfv["Low"], close=dfv["Close"],
                name=f"{tt} (Velas)",
                increasing_line_color="#26a69a", decreasing_line_color="#ef5350",
                showlegend=True
            ), row=1, col=1, secondary_y=False)

            # Volumen
            if "Volume" in dfv.columns:
                fig.add_trace(go.Bar(
                    x=dfv.index, y=dfv["Volume"],
                    name="Volumen", marker_color="rgba(100, 149, 237, 0.35)", opacity=0.5
                ), row=1, col=1, secondary_y=True)

            # EMAs
            if show_ema20:
                fig.add_trace(go.Scatter(
                    x=dfv.index, y=dfv["EMA20"], mode="lines",
                    line=dict(color="#ff9800", width=1.8),
                    name="EMA 20",
                ), row=1, col=1, secondary_y=False)
            if show_ema21:
                fig.add_trace(go.Scatter(
                    x=dfv.index, y=dfv["EMA21"], mode="lines",
                    line=dict(color="#7e57c2", width=1.8, dash="dot"),
                    name="EMA 21",
                ), row=1, col=1, secondary_y=False)

            # RSI
            fig.add_trace(go.Scatter(
                x=dfv.index, y=dfv["RSI"], mode="lines",
                line=dict(color="#009688", width=2.0),
                name=f"RSI ({rsi_period})"
            ), row=2, col=1)
            fig.add_hrect(y0=30, y1=70, fillcolor="rgba(200,200,200,0.12)", line_width=0, row=2, col=1)
            fig.add_hline(y=70, line=dict(color="#ef5350", width=1, dash="dot"), row=2, col=1)
            fig.add_hline(y=30, line=dict(color="#26a69a", width=1, dash="dot"), row=2, col=1)

            # MACD
            fig.add_trace(go.Bar(
                x=dfv.index, y=dfv["MACD_hist"],
                name="MACD Hist", marker_color="rgba(158,158,158,0.55)"
            ), row=3, col=1)
            fig.add_trace(go.Scatter(
                x=dfv.index, y=dfv["MACD_line"], mode="lines",
                line=dict(color="#2196f3", width=2.0),
                name="MACD"
            ), row=3, col=1)
            fig.add_trace(go.Scatter(
                x=dfv.index, y=dfv["MACD_signal"], mode="lines",
                line=dict(color="#ff5722", width=1.6, dash="dot"),
                name="Signal"
            ), row=3, col=1)
            fig.add_hline(y=0, line=dict(color="rgba(120,120,120,0.6)", width=1), row=3, col=1)

            fig.update_layout(
                hovermode="x unified",
                legend=dict(orientation="h", y=-0.12),
                margin=dict(l=10, r=10, t=40, b=10),
                xaxis_rangeslider_visible=True,
                height=920
            )
            fig.update_yaxes(title_text="Precio", row=1, col=1, secondary_y=False)
            fig.update_yaxes(title_text="Volumen", row=1, col=1, secondary_y=True)
            fig.update_yaxes(title_text="RSI", range=[0, 100], row=2, col=1)
            fig.update_yaxes(title_text="MACD", row=3, col=1)

            st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"No fue posible mostrar velas/indicadores: {e}")

    # Finviz + diagnóstico
    st.subheader("Indicadores Finviz y valuación")
    try:
        blocks = finviz_blocks(tt)

        # Presentación por apartados
        tabs = st.tabs(list(blocks.keys()))
        for i, (section, data_dict) in enumerate(blocks.items()):
            with tabs[i]:
                df_kv = pd.DataFrame(
                    {
                        "Métrica": list(data_dict.keys()),
                        "Valor": [
                            data_dict[k] if not isinstance(data_dict[k], float)
                            else (round(data_dict[k]*100, 2) if (0 < abs(data_dict[k]) < 1 and any(x in k for x in ["Margin", "ROE", "ROA", "Perf", "Yield"]))
                                  else round(data_dict[k], 2))
                            for k in data_dict.keys()
                        ],
                    }
                )

                def fmt_val(row):
                    k, v = row["Métrica"], row["Valor"]
                    if isinstance(v, (int, float)):
                        if any(x in k for x in ["Price", "Market Cap", "Enterprise Value", "Income", "Sales"]):
                            return f"{v:,.0f}"
                        if any(x in k for x in ["Margin", "ROE", "ROA", "Perf", "Yield"]) and abs(v) <= 100:
                            return f"{v:.2f}%"
                        return f"{v:,.2f}"
                    return str(v)

                df_kv["Valor"] = df_kv.apply(fmt_val, axis=1)
                st.dataframe(
                    df_kv.style.set_properties(**{"padding": "10px"}),
                    use_container_width=True,
                    height=min(520, 42*(len(df_kv)+1))
                )

        diag = valuation_diagnosis(blocks, tt)
        render_valuation_verdict(diag)
        with st.expander("Ver razones del veredicto"):
            for r in diag["razones"]:
                st.write("• " + r)

    except Exception as e:
        st.error(f"No se pudo obtener Finviz para {tt}: {e}")

# =============================================================================
# Pronóstico de precio (3-6-9-12 meses) usando alfa (CAPM simplificado)
# =============================================================================
st.header("Pronóstico por alfa (CAPM simplificado)")

fc1, fc2, fc3, fc4 = st.columns([1.2, 1, 1, 1])
with fc1:
    fc_ticker = st.text_input("Ticker a pronosticar", value=ticker_to_describe or selected, key="fc_ticker")
with fc2:
    fc_bench = st.text_input("Benchmark (mercado)", value=BENCH, key="fc_bench")
with fc3:
    fc_years = st.selectbox("Años de lookback (regresión)", [1, 3, 5, 10], index=1)
with fc4:
    fc_mkt_mean_window = st.selectbox("Media esperada del mercado", ["Último año", "Todo lookback"], index=0)

if fc_ticker:
    try:
        t_norm = _normalize_ticker(fc_ticker)
        b_norm = _normalize_ticker(fc_bench)
        start_fc = date(max(2000, TODAY.year - int(fc_years)), 1, 1)
        prices_fc = _download_block([t_norm, b_norm], start_fc, TODAY, field="Close").ffill().dropna()
        if prices_fc.empty or any(c not in prices_fc.columns for c in [t_norm, b_norm]):
            st.warning("No hay suficientes precios para el pronóstico.")
        else:
            # Retornos diarios
            r_all = prices_fc.pct_change().dropna()
            ri = r_all[t_norm]
            rm = r_all[b_norm]

            # OLS simple: ri = alpha + beta * rm
            beta, alpha = np.polyfit(rm.values, ri.values, deg=1)

            # Medias esperadas del mercado (diaria)
            if fc_mkt_mean_window == "Último año":
                cutoff = r_all.index.max() - pd.Timedelta(days=365)
                mu_m = rm.loc[rm.index >= cutoff].mean()
            else:
                mu_m = rm.mean()

            exp_r_daily = float(alpha + beta * mu_m)

            # Días de horizonte (trading ~ 21/mes): 3,6,9,12 meses
            horizons = {"3M": 63, "6M": 126, "9M": 189, "12M": 252}

            p0 = float(prices_fc[t_norm].iloc[-1])
            rows = []
            for label, n in horizons.items():
                gross = (1.0 + exp_r_daily) ** n
                ret_h = gross - 1.0
                p_fore = p0 * gross
                rows.append({"Horizonte": label, "Días": n, "Retorno esp.": ret_h, "Precio esp.": p_fore})
            df_fc = pd.DataFrame(rows)

            # Métricas clave
            alpha_ann = alpha * 252
            mu_m_ann = (1 + mu_m) ** 252 - 1

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Alpha (diario)", f"{alpha:,.4%}")
            m2.metric("Alpha (anual aprox.)", f"{alpha_ann:,.2%}")
            m3.metric("Beta", f"{beta:,.2f}")
            m4.metric("E[r_m] anual (hist.)", f"{mu_m_ann:,.2%}")

            # Tabla de pronóstico
            df_show = df_fc.copy()
            df_show["Retorno esp. (%)"] = (df_show["Retorno esp."] * 100).round(2)
            df_show["Precio esp. ($)"] = df_show["Precio esp."].map(lambda x: f"${x:,.2f}")
            df_show = df_show[["Horizonte", "Días", "Retorno esp. (%)", "Precio esp. ($)"]]
            st.subheader(f"Pronóstico de {t_norm} basado en alfa/beta vs {b_norm}")
            st.dataframe(df_show, use_container_width=True, height=220)

            # Curva del precio esperado (diaria) para 1 año
            nmax = max(horizons.values())
            proj_index = pd.date_range(start=prices_fc.index[-1] + pd.Timedelta(days=1), periods=nmax, freq="B")
            proj_curve = pd.Series((1.0 + exp_r_daily) ** np.arange(1, nmax+1), index=proj_index) * p0
            chart_df = pd.concat([
                prices_fc[[t_norm]].rename(columns={t_norm: "Precio real"}).iloc[-200:],
                proj_curve.rename("Precio esperado").to_frame()
            ], axis=0)
            figp = go.Figure()
            figp.add_trace(go.Scatter(x=chart_df.index, y=chart_df["Precio real"], name="Precio real", mode="lines", line=dict(color="#2196f3")))
            figp.add_trace(go.Scatter(x=proj_curve.index, y=proj_curve.values, name="Precio esperado", mode="lines", line=dict(color="#ff6f00", dash="dash")))
            figp.update_layout(title=f"Proyección 12M (constante E[r] diario={exp_r_daily:.4%})",
                               hovermode="x unified", margin=dict(l=10, r=10, t=40, b=10))
            figp.update_yaxes(title="Precio")
            st.plotly_chart(figp, use_container_width=True)

            st.caption("Metodología: ri = α + β·rm con datos diarios (Yahoo). E[rm] = media histórica seleccionada. Se asume persistencia de α y β y retorno diario constante para proyectar por capitalización. Esto es educativo, no asesoría financiera.")

            # Botón de descarga
            out_csv = df_fc.copy()
            out_csv["ret_esp"] = out_csv["Retorno esp."]
            out_csv["precio_esp"] = out_csv["Precio esp."]
            st.download_button("⬇️ CSV Pronóstico (α-CAPM)", data=out_csv.to_csv(index=False).encode(),
                               file_name=f"forecast_alpha_{t_norm}_vs_{b_norm}.csv", mime="text/csv")

            # Explicación IA
            with st.expander("🧠 Explicación IA de este pronóstico (α-CAPM)"):
                if st.button("Explicar pronóstico con IA", key="explain_capm"):
                    ctx_capm = {
                        "ticker": t_norm,
                        "benchmark": b_norm,
                        "alpha_diario": float(alpha),
                        "alpha_anual_aprox": float(alpha_ann),
                        "beta": float(beta),
                        "ret_mercado_diario_medio": float(mu_m),
                        "ret_mercado_anual_hist": float(mu_m_ann),
                        "precio_actual": float(p0),
                        "horizontes": df_fc.to_dict(orient="records"),
                    }
                    resumen_capm = ai_short_section_summary(
                        section_name="Pronóstico α-CAPM",
                        context=ctx_capm,
                        api_key=gemini_api_key.strip(),
                        model_name=model_name,
                        max_words=140,
                    )
                    st.write(resumen_capm)

    except Exception as e:
        st.error(f"No se pudo calcular el pronóstico: {e}")

# =============================================================================
# Pronóstico de precio por REGRESIÓN LINEAL (tendencia)
# =============================================================================
st.header("Pronóstico por regresión lineal (tendencia)")

lr1, lr2, lr3, lr4 = st.columns([1.2, 1, 1, 1])
with lr1:
    lr_ticker = st.text_input("Ticker (regresión lineal)", value=fc_ticker or ticker_to_describe or selected, key="lr_ticker")
with lr2:
    lr_years = st.selectbox("Años de lookback (tendencia)", [1, 3, 5, 10, "Máx"], index=2)
with lr3:
    lr_use_log = st.toggle("Usar log-precio (recomendado)", value=True)
with lr4:
    lr_clip_floor = st.number_input("Piso precio (evitar negativos)", min_value=0.0, value=0.01, step=0.01)

if lr_ticker:
    try:
        lr_t = _normalize_ticker(lr_ticker)
        Tlr = yf.Ticker(lr_t)
        hist_lr = Tlr.history(period="max", auto_adjust=True)
        if hist_lr.empty:
            st.warning("No hay historial para el ticker.")
        else:
            pr = hist_lr["Close"].dropna().copy()
            if lr_years != "Máx":
                start_lr = pr.index.max() - pd.DateOffset(years=int(lr_years))
                pr = pr[pr.index >= start_lr]
            t = np.arange(len(pr), dtype=float)
            y = np.log(pr.values) if lr_use_log else pr.values
            b, a = np.polyfit(t, y, deg=1)  # devuelve [b, a]
            y_fit = a + b * t
            resid = y - y_fit
            sigma = float(np.std(resid, ddof=1)) if len(resid) > 1 else 0.0

            horizons = {"3M": 63, "6M": 126, "9M": 189, "12M": 252}
            t0 = len(pr) - 1
            last_price = float(pr.iloc[-1])
            rows = []
            for label, n in horizons.items():
                tf = t0 + n
                y_pred = a + b * tf
                y_lo = y_pred - 1.96 * sigma
                y_hi = y_pred + 1.96 * sigma
                if lr_use_log:
                    p_pred = float(np.exp(y_pred))
                    p_lo = float(np.exp(y_lo))
                    p_hi = float(np.exp(y_hi))
                else:
                    p_pred = float(max(y_pred, lr_clip_floor))
                    p_lo = float(max(y_lo, lr_clip_floor))
                    p_hi = float(max(y_hi, lr_clip_floor))
                ret_exp = (p_pred / last_price) - 1.0 if last_price > 0 else np.nan
                rows.append({"Horizonte": label, "Días": n, "Precio esp.": p_pred, "Precio (lo)": p_lo, "Precio (hi)": p_hi, "Retorno esp.": ret_exp})
            df_lr = pd.DataFrame(rows)

            slope_day = b
            if lr_use_log:
                growth_daily = np.exp(slope_day) - 1
                growth_ann = (1 + growth_daily) ** 252 - 1
                m1, m2 = st.columns(2)
                m1.metric("Pendiente diaria (log)", f"{growth_daily:,.4%}")
                m2.metric("Crec. anual implícito", f"{growth_ann:,.2%}")
            else:
                m1, m2 = st.columns(2)
                m1.metric("Pendiente diaria ($)", f"{slope_day:,.4f}")
                m2.metric("Precio actual", f"${last_price:,.2f}")

            df_show = df_lr.copy()
            df_show["Retorno esp. (%)"] = (df_show["Retorno esp."] * 100).round(2)
            for c in ["Precio esp.", "Precio (lo)", "Precio (hi)"]:
                df_show[c] = df_show[c].map(lambda x: f"${x:,.2f}")
            df_show = df_show[["Horizonte", "Días", "Retorno esp. (%)", "Precio esp.", "Precio (lo)", "Precio (hi)"]]
            st.subheader(f"Pronóstico por regresión lineal — {lr_t}")
            st.dataframe(df_show, use_container_width=True, height=240)

            nmax = max(horizons.values())
            fut_index = pd.date_range(start=pr.index[-1] + pd.Timedelta(days=1), periods=nmax, freq="B")
            t_future = np.arange(len(pr)+1, len(pr)+1+nmax, dtype=float)
            y_pred_curve = a + b * t_future
            y_lo_curve = y_pred_curve - 1.96 * sigma
            y_hi_curve = y_pred_curve + 1.96 * sigma
            if lr_use_log:
                p_curve = np.exp(y_pred_curve)
                p_lo_c = np.exp(y_lo_curve)
                p_hi_c = np.exp(y_hi_curve)
            else:
                p_curve = np.maximum(y_pred_curve, lr_clip_floor)
                p_lo_c = np.maximum(y_lo_curve, lr_clip_floor)
                p_hi_c = np.maximum(y_hi_curve, lr_clip_floor)

            chart_df = pd.concat([
                pr.rename("Precio real").iloc[-200:],
                pd.Series(p_curve, index=fut_index, name="Precio esperado"),
                pd.Series(p_lo_c, index=fut_index, name="Banda 95% (lo)"),
                pd.Series(p_hi_c, index=fut_index, name="Banda 95% (hi)")
            ], axis=1)
            figlr = go.Figure()
            figlr.add_trace(go.Scatter(x=chart_df.index, y=chart_df["Precio real"], name="Precio real", mode="lines", line=dict(color="#1f77b4")))
            figlr.add_trace(go.Scatter(x=fut_index, y=p_curve, name="Precio esperado", mode="lines", line=dict(color="#ff7f0e", dash="dash")))
            figlr.add_trace(go.Scatter(x=fut_index, y=p_lo_c, name="Banda 95% (lo)", mode="lines", line=dict(color="rgba(255,127,14,0.35)", width=1)))
            figlr.add_trace(go.Scatter(x=fut_index, y=p_hi_c, name="Banda 95% (hi)", mode="lines", line=dict(color="rgba(255,127,14,0.35)", width=1), fill='tonexty', fillcolor='rgba(255,127,14,0.12)'))
            figlr.update_layout(title="Proyección 12M — Regresión lineal (con bandas 95%)",
                                hovermode="x unified", margin=dict(l=10, r=10, t=40, b=10))
            figlr.update_yaxes(title="Precio")
            st.plotly_chart(figlr, use_container_width=True)

            st.download_button("⬇️ CSV Pronóstico (Regresión Lineal)", data=df_lr.to_csv(index=False).encode(),
                               file_name=f"forecast_linear_{lr_t}.csv", mime="text/csv")

            st.caption("Metodología: tendencia lineal en precio o log-precio vs tiempo hábil. Bandas≈±1.96·σ de residuales. Limitaciones: no modela estacionalidad, shocks ni cambios de régimen; es un enfoque ilustrativo.")

            with st.expander("🧠 Explicación IA de la regresión lineal"):
                if st.button("Explicar regresión con IA", key="explain_regresion"):
                    ctx_lr = {
                        "ticker": lr_t,
                        "usa_log_precio": bool(lr_use_log),
                        "pendiente_diaria": float(slope_day),
                        "precio_actual": float(last_price),
                        "tabla_pronostico": df_lr.to_dict(orient="records"),
                    }
                    if lr_use_log:
                        ctx_lr["crecimiento_diario_implicito"] = float(growth_daily)
                        ctx_lr["crecimiento_anual_implicito"] = float(growth_ann)

                    resumen_lr = ai_short_section_summary(
                        section_name="Pronóstico por regresión lineal",
                        context=ctx_lr,
                        api_key=gemini_api_key.strip(),
                        model_name=model_name,
                        max_words=150,
                    )
                    st.write(resumen_lr)
    except Exception as e:
        st.error(f"No se pudo ejecutar la regresión lineal: {e}")

# =============================================================================
# Backtesting educativo — Cruce de EMAs
# =============================================================================
st.header("Backtesting (educativo) — Cruces de EMAs")
bt1, bt2, bt3, bt4 = st.columns([1, 1, 1, 2])
with bt1:
    bt_ticker = st.text_input("Ticker backtest", value=selected, key="bt_ticker")
with bt2:
    ema_fast = st.number_input("EMA rápida", min_value=3, max_value=100, value=20, step=1)
with bt3:
    ema_slow = st.number_input("EMA lenta", min_value=5, max_value=250, value=50, step=1)
with bt4:
    bt_years = st.selectbox("Años a evaluar", [3, 5, 10, "Máx"], index=1)

if bt_ticker:
    try:
        Tbt = yf.Ticker(_normalize_ticker(bt_ticker))
        hist_bt = Tbt.history(period="max", auto_adjust=True)
        pr = hist_bt["Close"].dropna()

        if bt_years != "Máx":
            start_bt = pr.index.max() - pd.DateOffset(years=int(bt_years))
            pr = pr[pr.index >= start_bt]

        # Indicadores para estrategia
        ema_f = pr.ewm(span=int(ema_fast), adjust=False).mean()
        ema_s = pr.ewm(span=int(ema_slow), adjust=False).mean()

        # Señales (1: largo cuando EMAf > EMAs, 0: fuera)
        position = (ema_f > ema_s).astype(int)

        # Retornos diarios del activo
        r = pr.pct_change().fillna(0.0)

        # Evitar look-ahead: aplicar posición del día anterior
        strat_ret = r * position.shift(1).fillna(0)

        # Equity curves (base 1.0)
        bench_curve = (1 + r).cumprod()
        strat_curve = (1 + strat_ret).cumprod()

        # Métricas
        def cagr_from_curve(curve: pd.Series) -> float:
            if len(curve) < 2:
                return np.nan
            years = (curve.index[-1] - curve.index[0]).days / 365.25
            return curve.iloc[-1] ** (1 / max(years, 1e-9)) - 1

        strat_cagr = cagr_from_curve(strat_curve)
        bench_cagr = cagr_from_curve(bench_curve)
        sharpe = (strat_ret.mean() / strat_ret.std() * np.sqrt(252)) if strat_ret.std() != 0 else np.nan
        mdd_strat = _max_drawdown(strat_curve)
        mdd_bench = _max_drawdown(bench_curve)
        inpos = strat_ret[position.shift(1) == 1]
        win_rate = (inpos > 0).mean() if len(inpos) else np.nan
        crosses = ((position.diff() != 0) & (position == 1)).sum()

        m1, m2, m3, m4, m5, m6 = st.columns(6)
        m1.metric("CAGR Estrategia", f"{strat_cagr:,.2%}" if pd.notna(strat_cagr) else "N/D")
        m2.metric("CAGR Buy&Hold", f"{bench_cagr:,.2%}" if pd.notna(bench_cagr) else "N/D")
        m3.metric("Sharpe (≈)", f"{sharpe:,.2f}" if pd.notna(sharpe) else "N/D")
        m4.metric("Máx DD Estrategia", f"{mdd_strat:,.2%}" if pd.notna(mdd_strat) else "N/D")
        m5.metric("Máx DD Bench", f"{mdd_bench:,.2%}" if pd.notna(mdd_bench) else "N/D")
        m6.metric("# Trades (entradas)", f"{int(crosses)}")

        figb = go.Figure()
        figb.add_trace(go.Scatter(x=strat_curve.index, y=strat_curve, mode="lines",
                                  name="Estrategia EMAs", line=dict(color="#ff6f00", width=2)))
        figb.add_trace(go.Scatter(x=bench_curve.index, y=bench_curve, mode="lines",
                                  name="Buy & Hold", line=dict(color="#2196f3", width=1.5)))
        entries = (position.diff() == 1)
        figb.add_trace(go.Scatter(
            x=pr.index[entries], y=bench_curve[entries],
            mode="markers", marker=dict(symbol="triangle-up", color="#2e7d32", size=8),
            name="Entrada (cruce EMA)"
        ))
        figb.update_layout(
            hovermode="x unified",
            margin=dict(l=10, r=10, t=40, b=10),
            xaxis_rangeslider_visible=True,
            title=f"Backtest EMAs — {bt_ticker} (EMA {ema_fast}/{ema_slow})"
        )
        figb.update_yaxes(title="Crecimiento (base 1.0)")
        st.plotly_chart(figb, use_container_width=True)

        # === Precio + EMAs + Cruces (visual) ===
        with st.expander("🧠 Explicación IA del backtest EMAs"):
            if st.button("Explicar backtest con IA", key="explain_backtest"):
                ctx_bt = {
                    "ticker": bt_ticker,
                    "ema_rapida": int(ema_fast),
                    "ema_lenta": int(ema_slow),
                    "anios_evaluados": bt_years,
                    "cagr_estrategia": float(strat_cagr) if pd.notna(strat_cagr) else None,
                    "cagr_buy_hold": float(bench_cagr) if pd.notna(bench_cagr) else None,
                    "sharpe_aprox": float(sharpe) if pd.notna(sharpe) else None,
                    "max_dd_estrategia": float(mdd_strat) if pd.notna(mdd_strat) else None,
                    "max_dd_bench": float(mdd_bench) if pd.notna(mdd_bench) else None,
                    "win_rate_trades": float(win_rate) if pd.notna(win_rate) else None,
                    "numero_trades": int(crosses),
                }

                resumen_bt = ai_short_section_summary(
                    section_name="Backtest de cruces de EMAs",
                    context=ctx_bt,
                    api_key=gemini_api_key.strip(),
                    model_name=model_name,
                    max_words=150,
                )
                st.write(resumen_bt)

        show_price_emas = st.toggle("Ver precio + EMAs y cruces", value=True, key="show_price_emas")
        if show_price_emas:
            figc = go.Figure()
            figc.add_trace(go.Scatter(x=pr.index, y=pr.values, name="Precio", mode="lines", line=dict(color="#444", width=1.5)))
            figc.add_trace(go.Scatter(x=ema_f.index, y=ema_f.values, name=f"EMA {ema_fast}", mode="lines", line=dict(color="#ff6f00", width=2)))
            figc.add_trace(go.Scatter(x=ema_s.index, y=ema_s.values, name=f"EMA {ema_slow}", mode="lines", line=dict(color="#2196f3", width=2)))

            # Cruces: arriba (entrada) y abajo (salida)
            cross_up = (ema_f > ema_s) & (ema_f.shift(1) <= ema_s.shift(1))
            cross_dn = (ema_f < ema_s) & (ema_f.shift(1) >= ema_s.shift(1))

            figc.add_trace(go.Scatter(
                x=pr.index[cross_up], y=pr[cross_up],
                mode="markers", name="Cruce ↑ (entrada)",
                marker=dict(symbol="triangle-up", color="#2e7d32", size=10, line=dict(color="#1b5e20", width=1))
            ))
            figc.add_trace(go.Scatter(
                x=pr.index[cross_dn], y=pr[cross_dn],
                mode="markers", name="Cruce ↓ (salida)",
                marker=dict(symbol="triangle-down", color="#c62828", size=10, line=dict(color="#8e0000", width=1))
            ))

            figc.update_layout(
                title=f"{bt_ticker}: Precio + EMAs y cruces",
                hovermode="x unified",
                margin=dict(l=10, r=10, t=40, b=10),
                xaxis_rangeslider_visible=True,
                legend=dict(orientation="h", y=-0.15)
            )
            figc.update_yaxes(title="Precio")
            st.plotly_chart(figc, use_container_width=True)

        df_bt = pd.DataFrame({
            "Close": pr,
            f"EMA_{ema_fast}": ema_f,
            f"EMA_{ema_slow}": ema_s,
            "Position": position,
            "Return": r,
            "Strategy_Return": strat_ret,
            "Equity_Strategy": strat_curve,
            "Equity_Bench": bench_curve
        }).dropna()
        st.dataframe(df_bt.tail(500), use_container_width=True, height=360)

        st.download_button(
            "⬇️ CSV Backtest (completo)",
            data=df_bt.to_csv().encode(),
            file_name=f"backtest_{bt_ticker}_ema{ema_fast}_{ema_slow}.csv",
            mime="text/csv"
        )

        st.caption("Backtest educativo: ignora costos, deslizamientos e impuestos. No constituye asesoría de inversión.")
    except Exception as e:
        st.error(f"No se pudo ejecutar el backtest: {e}")

# =============================================================================
# Asistente IA de la app — Preguntas sobre los datos
# =============================================================================
st.header("Asistente IA sobre los datos de la app")

st.caption(
    "Haz preguntas en español sobre los rendimientos, volatilidad, diferencias entre ETFs, etc. "
    "El asistente responde solo con base en los datos cargados en esta página."
)

def ai_data_qa(question: str, context: dict, api_key: str, model_name: str, max_words: int = 220) -> str:
    if not api_key:
        return "⚠️ Falta la API Key de Gemini/Google. Configúrala en el servidor (.env / secrets)."

    safe_ctx = json.dumps(context, default=str)[:7000]
    prompt = f"""
Eres un asistente de análisis cuantitativo dentro de una app de ingeniería financiera.
El usuario hará una pregunta sobre los datos de la app.

Reglas:
- Responde SIEMPRE en español.
- Usa solo la información que viene en el JSON (no inventes datos nuevos).
- Máximo {max_words} palabras.
- No des recomendaciones de compra/venta ni lenguaje tipo 'debes invertir'.

Pregunta del usuario:
{question}

Datos disponibles (JSON):
{safe_ctx}
""".strip()

    try:
        answer = generate_with_gemini(api_key, model_name, prompt)
    except Exception as e:
        answer = f"No se pudo contestar con la IA: {e}"
    if not answer:
        answer = "No pude generar una respuesta con la IA en este momento."
    return answer

# Construimos contexto con lo que ya calculaste
try:
    # métricas por ticker (CAGR, vol, DD)
    metrics_context = {}
    for tkr in px.columns:
        p = px[tkr].dropna()
        r_ = ret[tkr].dropna()
        if len(p) < 2:
            continue
        years = (p.index[-1] - p.index[0]).days / 365.25
        cagr = (p.iloc[-1] / p.iloc[0]) ** (1 / max(years, 1e-9)) - 1
        vol_ann = _annualize_vol(r_.std()) if not r_.empty else np.nan
        mdd = _max_drawdown((1 + r_).cumprod())
        metrics_context[tkr] = {
            "CAGR": float(cagr),
            "vol_anualizada": float(vol_ann),
            "max_drawdown": float(mdd),
        }

    ctx_app = {
        "fecha_hoy": str(TODAY),
        "etf_principal_seleccionado": selected,
        "comparando_con_SPY": bool(compare_spy),
        "tickers_cargados": list(px.columns),
        "metrics_por_ticker": metrics_context,
        "rendimientos_anuales_pct": annual_pct.to_dict(),
        "estadisticas_retornos_diarios": ret.describe().to_dict(),
    }
except Exception:
    ctx_app = {"error": "No se pudo construir el contexto, revisa los datos."}

col_q1, col_q2 = st.columns([2, 1])
with col_q1:
    user_question = st.text_area(
        "Escribe tu pregunta (ej. *¿qué sector ha tenido mejor rendimiento en los últimos años?*)",
        key="qa_question",
        height=90,
    )
with col_q2:
    st.write("Contexto actual:")
    st.json(
        {
            "ETF principal": selected,
            "Tickers": list(px.columns),
            "Años_historial": lookback,
        },
        expanded=False,
    )

if st.button("Preguntar a la IA sobre los datos", key="qa_ask"):
    if not user_question.strip():
        st.warning("Escribe primero una pregunta.")
    else:
        answer = ai_data_qa(
            question=user_question.strip(),
            context=ctx_app,
            api_key=gemini_api_key.strip(),
            model_name=model_name,
            max_words=220,
        )
        st.markdown("### Respuesta de la IA")
        st.write(answer)

# ========= Fin app.py =========
