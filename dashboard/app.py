import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from config.settings import RISK, TRADING, settings
from database.models import init_db
from database.repository import (
    get_all_trades,
    get_latest_snapshot,
    get_snapshots_last_days,
    get_today_pnl,
)
from models.predictor import is_model_trained

# ── Página ────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AsistenteTrading",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",   # colapsado por defecto en móvil
)

# ── CSS mobile-first ──────────────────────────────────────────────────────────
st.markdown("""
<style>
  /* Fuente base más grande en móvil */
  html, body, [class*="css"] { font-size: 15px; }

  /* Tarjeta métrica personalizada */
  .card {
    background: #1a1a2e;
    border-radius: 14px;
    padding: 16px 12px;
    text-align: center;
    margin-bottom: 8px;
    border: 1px solid #2a2a4a;
  }
  .card-label { font-size: 0.78rem; color: #aaa; text-transform: uppercase; letter-spacing: 1px; }
  .card-value { font-size: 1.7rem; font-weight: 800; margin: 4px 0; }
  .card-sub   { font-size: 0.85rem; color: #888; }
  .green { color: #00e676; }
  .red   { color: #ff5252; }
  .gold  { color: #FFD700; }
  .blue  { color: #40c4ff; }

  /* Sección separadora */
  .section-title {
    font-size: 1rem; font-weight: 700;
    color: #aaa; text-transform: uppercase;
    letter-spacing: 1.5px; margin: 20px 0 8px 0;
    border-bottom: 1px solid #2a2a4a; padding-bottom: 4px;
  }

  /* Botón de actualizar */
  div[data-testid="stButton"] > button {
    width: 100%; border-radius: 10px;
    font-weight: 600; padding: 10px;
  }

  /* Tabla más compacta en móvil */
  .dataframe td, .dataframe th { font-size: 0.78rem !important; padding: 4px 6px !important; }

  /* Ocultar menú hamburguesa y footer de Streamlit */
  #MainMenu {visibility: hidden;}
  footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


def card(label: str, value: str, sub: str = "", color: str = "gold") -> str:
    return f"""
    <div class="card">
      <div class="card-label">{label}</div>
      <div class="card-value {color}">{value}</div>
      {"<div class='card-sub'>" + sub + "</div>" if sub else ""}
    </div>"""


def pnl_color(val: float) -> str:
    return "green" if val >= 0 else "red"


@st.cache_data(ttl=60)
def load_data():
    init_db()
    return (
        get_latest_snapshot(),
        get_all_trades(),
        get_snapshots_last_days(30),
        get_today_pnl(),
    )


# ── Datos ─────────────────────────────────────────────────────────────────────
snapshot, trades, snapshots, today_pnl = load_data()
initial_capital = float(TRADING.get("capital", 1000))
mode = TRADING.get("mode", "paper")

if snapshot:
    total_val     = snapshot.total_value
    total_pnl     = snapshot.total_pnl
    total_pnl_pct = snapshot.total_pnl_pct
    drawdown      = snapshot.drawdown_pct
    cash          = snapshot.cash_usd
    pos_val       = snapshot.position_value
else:
    total_val = cash = initial_capital
    total_pnl = total_pnl_pct = drawdown = pos_val = 0.0

# ── Header ────────────────────────────────────────────────────────────────────
mode_badge = "🟡 PAPER" if mode == "paper" else "🔴 LIVE"
st.markdown(f"## 📈 AsistenteTrading &nbsp; <small>{mode_badge}</small>", unsafe_allow_html=True)
st.caption(f"BTC/USDT · {pd.Timestamp.now().strftime('%d %b %Y, %H:%M')}")

col_refresh, col_status = st.columns([1, 1])
with col_refresh:
    if st.button("🔄 Actualizar", use_container_width=True):
        st.cache_data.clear()
        st.rerun()
with col_status:
    model_ok = is_model_trained()
    st.markdown(
        f"<div style='text-align:center;padding:8px;background:#1a1a2e;border-radius:10px;"
        f"border:1px solid #2a2a4a'>"
        f"{'✅ Modelo listo' if model_ok else '⚠️ Sin modelo'}</div>",
        unsafe_allow_html=True,
    )

# ── Métricas principales (cards 2x2 en móvil) ────────────────────────────────
st.markdown("<div class='section-title'>Portfolio</div>", unsafe_allow_html=True)

r1c1, r1c2 = st.columns(2)
r2c1, r2c2 = st.columns(2)

with r1c1:
    st.markdown(card("Total", f"${total_val:,.0f}", f"inicial: ${initial_capital:,.0f}", "gold"), unsafe_allow_html=True)
with r1c2:
    c = pnl_color(total_pnl)
    st.markdown(card("PnL Total", f"${total_pnl:+.2f}", f"{total_pnl_pct:+.2f}%", c), unsafe_allow_html=True)
with r2c1:
    c = pnl_color(today_pnl)
    st.markdown(card("PnL Hoy", f"${today_pnl:+.2f}", "", c), unsafe_allow_html=True)
with r2c2:
    dd_c = "green" if drawdown > -10 else "red"
    st.markdown(card("Drawdown", f"{drawdown:.1f}%", "máx caída", dd_c), unsafe_allow_html=True)

# ── Posición abierta ──────────────────────────────────────────────────────────
open_trades = [t for t in trades if t.is_open]
if open_trades:
    t = open_trades[0]
    st.markdown("<div class='section-title'>Posición Abierta</div>", unsafe_allow_html=True)
    pa1, pa2 = st.columns(2)
    with pa1:
        st.markdown(card("Entrada", f"${t.price:,.2f}", TRADING['symbol'], "blue"), unsafe_allow_html=True)
    with pa2:
        st.markdown(card("Cantidad", f"{t.quantity:.5f}", "BTC", "blue"), unsafe_allow_html=True)
    sl1, sl2 = st.columns(2)
    with sl1:
        st.markdown(card("Stop Loss", f"${t.stop_loss:,.2f}", "-2%", "red"), unsafe_allow_html=True)
    with sl2:
        st.markdown(card("Take Profit", f"${t.take_profit:,.2f}", "+5%", "green"), unsafe_allow_html=True)

# ── Curva de equity ───────────────────────────────────────────────────────────
st.markdown("<div class='section-title'>Evolución del Portfolio</div>", unsafe_allow_html=True)

if snapshots:
    eq_df = pd.DataFrame([{"fecha": s.timestamp, "valor": s.total_value} for s in snapshots])
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=eq_df["fecha"], y=eq_df["valor"],
        mode="lines", line=dict(color="#FFD700", width=2.5),
        fill="tozeroy", fillcolor="rgba(255,215,0,0.06)",
    ))
    fig.add_hline(y=initial_capital, line_dash="dash", line_color="#555",
                  annotation_text="Capital inicial", annotation_font_color="#888")
    fig.update_layout(
        template="plotly_dark", height=220,
        margin=dict(l=0, r=0, t=4, b=0),
        showlegend=False,
        xaxis=dict(showgrid=False),
        yaxis=dict(title="USD", showgrid=True, gridcolor="#222"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
else:
    st.info("Sin datos aún. El bot debe completar al menos un ciclo.")

# ── Estadísticas rápidas ──────────────────────────────────────────────────────
closed = [t for t in trades if not t.is_open]
if closed:
    st.markdown("<div class='section-title'>Estadísticas</div>", unsafe_allow_html=True)
    wins = [t for t in closed if (t.pnl or 0) > 0]
    s1, s2, s3 = st.columns(3)
    with s1:
        st.markdown(card("Operaciones", str(len(closed)), "cerradas", "blue"), unsafe_allow_html=True)
    with s2:
        wr = len(wins) / len(closed) * 100
        st.markdown(card("Win Rate", f"{wr:.0f}%", f"{len(wins)} ganadoras", pnl_color(wr - 50)), unsafe_allow_html=True)
    with s3:
        total_closed_pnl = sum(t.pnl or 0 for t in closed)
        st.markdown(card("PnL cerradas", f"${total_closed_pnl:+.2f}", "", pnl_color(total_closed_pnl)), unsafe_allow_html=True)

# ── Últimas operaciones ───────────────────────────────────────────────────────
st.markdown("<div class='section-title'>Últimas Operaciones</div>", unsafe_allow_html=True)

if trades:
    recent = sorted(trades, key=lambda t: t.timestamp or pd.Timestamp.min, reverse=True)[:15]
    rows = []
    for t in recent:
        pnl_str = f"${t.pnl:+.2f}" if t.pnl is not None else "Abierta"
        rows.append({
            "Fecha": t.timestamp.strftime("%d/%m %H:%M") if t.timestamp else "-",
            "Estado": "🟢 Abierta" if t.is_open else ("✅" if (t.pnl or 0) >= 0 else "❌"),
            "Entrada": f"${t.price:,.1f}",
            "PnL": pnl_str,
            "Motivo": t.close_reason or "-",
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True, height=280)
else:
    st.info("Sin operaciones registradas aún.")

# ── Backtesting ───────────────────────────────────────────────────────────────
with st.expander("📊 Ver último resultado de Backtesting"):
    backtest_img = ROOT / "backtesting" / "results" / "backtest_result.png"
    if backtest_img.exists():
        st.image(str(backtest_img), use_container_width=True)
    else:
        st.info("Ejecuta el entrenamiento primero (py run.py → opción 1).")

# ── Configuración de Telegram ─────────────────────────────────────────────────
with st.expander("📱 Configurar Telegram (alertas en celular)"):
    st.markdown("""
**Pasos para recibir alertas en tu celular:**

1. Abre Telegram y busca **@BotFather**
2. Escribe `/newbot` y sigue las instrucciones
3. Guarda el **token** que te da BotFather
4. Busca **@userinfobot** en Telegram y escríbele cualquier cosa
5. Te responderá con tu **Chat ID**
6. En tu archivo `.env` del VPS, configura:
   ```
   TELEGRAM_TOKEN=tu_token_aqui
   TELEGRAM_CHAT_ID=tu_chat_id_aqui
   TELEGRAM_ENABLED=true
   ```
7. Reinicia el bot en EasyPanel

Después recibirás alertas automáticas por cada operación.
""")

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    f"AsistenteTrading · Modo {mode.upper()} · "
    "Los resultados pasados no garantizan resultados futuros."
)
