import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import yfinance as yf
from datetime import datetime

from config.settings import TRADING
from database.models import init_db
from database.repository import (
    get_all_trades,
    get_latest_snapshot,
    get_snapshots_last_days,
    get_today_pnl,
)
from models.predictor import is_model_trained

# ── Página ─────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AsistenteTrading Pro",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── CSS profesional ────────────────────────────────────────────────────────────
st.markdown("""
<style>
  html, body, [class*="css"] {
    background-color: #0d1117 !important;
    font-family: 'Segoe UI', system-ui, sans-serif;
    color: #e6edf3;
  }
  .metric-card {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 10px;
    padding: 14px 16px;
    text-align: center;
    margin-bottom: 8px;
  }
  .metric-label { font-size: 0.68rem; color: #8b949e; text-transform: uppercase; letter-spacing: 1.5px; }
  .metric-value { font-size: 1.45rem; font-weight: 800; margin: 5px 0; }
  .metric-sub   { font-size: 0.78rem; color: #8b949e; }

  .signal-box {
    border-radius: 12px;
    padding: 18px 14px;
    text-align: center;
    margin-bottom: 10px;
    border: 2px solid;
  }
  .signal-buy  { background: rgba(63,185,80,.10); border-color: #3fb950; }
  .signal-sell { background: rgba(248,81,73,.10);  border-color: #f85149; }
  .signal-wait { background: rgba(210,153,34,.10); border-color: #d29922; }
  .signal-label { font-size: 0.68rem; color: #8b949e; text-transform: uppercase; letter-spacing: 2px; }
  .signal-value { font-size: 1.7rem; font-weight: 900; margin: 6px 0; }
  .signal-conf  { font-size: 0.8rem; color: #8b949e; }

  .stat-row {
    display: flex; justify-content: space-between;
    padding: 7px 0; border-bottom: 1px solid #21262d; font-size: 0.83rem;
  }
  .stat-key { color: #8b949e; }
  .stat-val { color: #e6edf3; font-weight: 600; }

  .section-hdr {
    font-size: 0.68rem; font-weight: 700; color: #8b949e;
    text-transform: uppercase; letter-spacing: 2px;
    border-bottom: 1px solid #30363d;
    padding-bottom: 6px; margin: 18px 0 10px 0;
  }
  .green { color: #3fb950; } .red { color: #f85149; }
  .gold  { color: #d29922; } .blue { color: #58a6ff; }

  #MainMenu { visibility: hidden; } footer { visibility: hidden; }
  .stDeployButton { display: none; }
  div[data-testid="stButton"] > button {
    border-radius: 8px; font-weight: 600;
    background: #21262d; border: 1px solid #30363d; color: #e6edf3;
  }
  div[data-testid="stButton"] > button:hover { background: #30363d; }
  .dataframe td, .dataframe th { font-size: 0.79rem !important; }
</style>
""", unsafe_allow_html=True)


# ── Helpers ────────────────────────────────────────────────────────────────────
def _squeeze(series):
    """Convierte Series o DataFrame de una columna a Series 1D."""
    if hasattr(series, 'squeeze'):
        s = series.squeeze()
        if isinstance(s, pd.DataFrame):
            s = s.iloc[:, 0]
        return s
    return series


@st.cache_data(ttl=60)
def get_market_data():
    try:
        df = yf.download("BTC-USD", period="7d", interval="1h",
                         progress=False, auto_adjust=True)
        if hasattr(df.columns, 'levels'):
            df.columns = df.columns.get_level_values(0)
        if df.empty:
            return pd.DataFrame(), 0, 0, 0, 0, 0, 0

        df2 = yf.download("BTC-USD", period="3d", interval="1d",
                          progress=False, auto_adjust=True)
        if hasattr(df2.columns, 'levels'):
            df2.columns = df2.columns.get_level_values(0)

        close   = _squeeze(df["Close"])
        current = float(close.iloc[-1])
        prev    = float(_squeeze(df2["Close"]).iloc[-2]) if len(df2) >= 2 else current
        ch      = current - prev
        ch_pct  = ch / prev * 100 if prev else 0

        high_24h = float(_squeeze(df["High"]).tail(24).max())
        low_24h  = float(_squeeze(df["Low"]).tail(24).min())
        vol_24h  = float(_squeeze(df["Volume"]).tail(24).sum())

        # EMA
        df["EMA20"]  = close.ewm(span=20).mean()
        df["EMA50"]  = close.ewm(span=50).mean()
        df["EMA200"] = close.ewm(span=200).mean()

        # RSI manual
        delta = close.diff()
        gain  = delta.where(delta > 0, 0.0).rolling(14).mean()
        loss  = (-delta.where(delta < 0, 0.0)).rolling(14).mean()
        rs    = gain / loss.replace(0, float("nan"))
        df["RSI"] = 100 - (100 / (1 + rs))

        return df, current, ch, ch_pct, high_24h, low_24h, vol_24h
    except Exception:
        return pd.DataFrame(), 0, 0, 0, 0, 0, 0


@st.cache_data(ttl=60)
def load_db_data():
    init_db()
    return (
        get_latest_snapshot(),
        get_all_trades(),
        get_snapshots_last_days(30),
        get_today_pnl(),
    )


def build_chart(df: pd.DataFrame):
    if df.empty or len(df) < 5:
        return None

    open_  = _squeeze(df["Open"])
    high_  = _squeeze(df["High"])
    low_   = _squeeze(df["Low"])
    close_ = _squeeze(df["Close"])
    vol_   = _squeeze(df["Volume"])

    colors_vol = [
        "#3fb950" if float(c) >= float(o) else "#f85149"
        for c, o in zip(close_, open_)
    ]

    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.025,
        row_heights=[0.60, 0.18, 0.22],
    )

    # Velas
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=open_, high=high_, low=low_, close=close_,
        name="BTC/USDT",
        increasing_line_color="#3fb950", increasing_fillcolor="#3fb950",
        decreasing_line_color="#f85149", decreasing_fillcolor="#f85149",
        line_width=1,
    ), row=1, col=1)

    # EMAs
    for col_name, color, label in [
        ("EMA20", "#58a6ff", "EMA 20"),
        ("EMA50", "#d29922", "EMA 50"),
        ("EMA200", "#6e7681", "EMA 200"),
    ]:
        if col_name in df.columns:
            fig.add_trace(go.Scatter(
                x=df.index, y=_squeeze(df[col_name]),
                name=label, line=dict(color=color, width=1.3),
                hovertemplate=f"{label}: $%{{y:,.0f}}<extra></extra>",
            ), row=1, col=1)

    # Volumen
    fig.add_trace(go.Bar(
        x=df.index, y=vol_,
        name="Vol", marker_color=colors_vol, opacity=0.65, showlegend=False,
    ), row=2, col=1)

    # RSI
    if "RSI" in df.columns:
        rsi_ = _squeeze(df["RSI"])
        fig.add_trace(go.Scatter(
            x=df.index, y=rsi_,
            name="RSI 14", line=dict(color="#a371f7", width=1.5),
            hovertemplate="RSI: %{y:.1f}<extra></extra>",
        ), row=3, col=1)
        fig.add_hrect(y0=70, y1=100, fillcolor="rgba(248,81,73,.07)",
                      line_width=0, row=3, col=1)
        fig.add_hrect(y0=0, y1=30, fillcolor="rgba(63,185,80,.07)",
                      line_width=0, row=3, col=1)
        for y_val, color in [(70, "#f85149"), (30, "#3fb950")]:
            fig.add_hline(y=y_val, line_dash="dot", line_color=color,
                          line_width=0.8, row=3, col=1)

    grid = "#21262d"
    fig.update_layout(
        template="plotly_dark",
        height=560,
        paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
        margin=dict(l=0, r=60, t=10, b=0),
        showlegend=True,
        legend=dict(orientation="h", x=0, y=1.01,
                    font=dict(size=10, color="#8b949e"),
                    bgcolor="rgba(0,0,0,0)"),
        xaxis=dict(showgrid=True, gridcolor=grid, rangeslider_visible=False,
                   showspikes=True, spikecolor="#8b949e", spikethickness=1),
        xaxis2=dict(showgrid=True, gridcolor=grid),
        xaxis3=dict(showgrid=True, gridcolor=grid),
        yaxis=dict(showgrid=True, gridcolor=grid, side="right",
                   title="USD", title_font_size=10),
        yaxis2=dict(showgrid=False, side="right", title="Vol", title_font_size=9),
        yaxis3=dict(showgrid=True, gridcolor=grid, side="right",
                    title="RSI", title_font_size=9, range=[0, 100]),
    )
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# DATOS
# ══════════════════════════════════════════════════════════════════════════════
df_mkt, price, ch, ch_pct, high_24h, low_24h, vol_24h = get_market_data()
snapshot, trades, snapshots, today_pnl = load_db_data()

initial_capital = float(TRADING.get("capital", 1000))
mode   = TRADING.get("mode", "paper")
symbol = TRADING.get("symbol", "BTC/USDT")

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

open_trades = [t for t in trades if t.is_open]
closed      = [t for t in trades if not t.is_open]

# ══════════════════════════════════════════════════════════════════════════════
# HEADER — BARRA DE PRECIO EN VIVO
# ══════════════════════════════════════════════════════════════════════════════
mode_badge  = "🟡 PAPER" if mode == "paper" else "🔴 LIVE"
ch_color    = "#3fb950" if ch >= 0 else "#f85149"
ch_arrow    = "▲" if ch >= 0 else "▼"
price_str   = f"${price:,.2f}" if price > 0 else "—"
model_ok    = is_model_trained()

cA, cB, cC, cD = st.columns([2, 3, 5, 2])

with cA:
    st.markdown(f"""
    <div style="padding-top:6px">
      <div style="font-size:1.3rem;font-weight:900;color:#e6edf3">AsistenteTrading</div>
      <div style="font-size:0.72rem;color:#8b949e">{symbol} &nbsp;·&nbsp; {mode_badge}</div>
    </div>""", unsafe_allow_html=True)

with cB:
    st.markdown(f"""
    <div style="padding-top:2px">
      <span style="font-size:2rem;font-weight:900;color:#fff;font-family:monospace">{price_str}</span><br>
      <span style="font-size:0.95rem;color:{ch_color};font-weight:700">
        {ch_arrow} ${abs(ch):,.2f} &nbsp;({ch_pct:+.2f}%)
      </span>
    </div>""", unsafe_allow_html=True)

with cC:
    model_color = "#3fb950" if model_ok else "#f85149"
    model_text  = "✅ IA Activa" if model_ok else "⚠️ Sin modelo"
    st.markdown(f"""
    <div style="display:flex;gap:28px;padding-top:8px;font-size:0.82rem;flex-wrap:wrap">
      <div><div style="color:#8b949e">MÁX 24H</div>
           <div style="color:#3fb950;font-weight:700">${high_24h:,.0f}</div></div>
      <div><div style="color:#8b949e">MÍN 24H</div>
           <div style="color:#f85149;font-weight:700">${low_24h:,.0f}</div></div>
      <div><div style="color:#8b949e">VOLUMEN 24H</div>
           <div style="color:#e6edf3;font-weight:700">${vol_24h/1e9:.2f}B</div></div>
      <div><div style="color:#8b949e">CAPITAL</div>
           <div style="color:#58a6ff;font-weight:700">${total_val:,.2f}</div></div>
      <div><div style="color:#8b949e">PNL TOTAL</div>
           <div style="color:{'#3fb950' if total_pnl >= 0 else '#f85149'};font-weight:700">${total_pnl:+.2f}</div></div>
      <div><div style="color:#8b949e">MODELO IA</div>
           <div style="color:{model_color};font-weight:700">{model_text}</div></div>
    </div>""", unsafe_allow_html=True)

with cD:
    st.markdown("<div style='padding-top:8px'></div>", unsafe_allow_html=True)
    if st.button("🔄 Actualizar", use_container_width=True):
        st.cache_data.clear()
        st.rerun()
    st.caption(datetime.now().strftime("%d %b %Y  %H:%M"))

st.markdown("<hr style='border-color:#30363d;margin:8px 0 12px 0'>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# LAYOUT PRINCIPAL: Gráfico (izq) + Panel de control (der)
# ══════════════════════════════════════════════════════════════════════════════
col_chart, col_panel = st.columns([7, 3])

with col_chart:
    fig = build_chart(df_mkt)
    if fig:
        st.plotly_chart(fig, use_container_width=True, config={
            "displayModeBar": True,
            "displaylogo": False,
            "modeBarButtonsToRemove": ["lasso2d", "select2d", "autoScale2d"],
            "toImageButtonOptions": {"format": "png"},
        })
    else:
        st.warning("No se pudieron cargar los datos del mercado.")

with col_panel:
    # ── Señal de IA ──────────────────────────────────────────────────────────
    st.markdown("<div class='section-hdr'>Señal de IA</div>", unsafe_allow_html=True)

    if open_trades:
        box_cls, sig_txt, sig_icon, sig_color = "signal-buy", "EN POSICIÓN", "🟢", "#3fb950"
    elif closed:
        last_cl = sorted(closed, key=lambda t: t.timestamp or pd.Timestamp.min)[-1]
        if last_cl.close_reason == "take_profit":
            box_cls, sig_txt, sig_icon, sig_color = "signal-buy", "TAKE PROFIT ✓", "💰", "#3fb950"
        elif last_cl.close_reason == "stop_loss":
            box_cls, sig_txt, sig_icon, sig_color = "signal-sell", "STOP LOSS", "🛑", "#f85149"
        else:
            box_cls, sig_txt, sig_icon, sig_color = "signal-wait", "ESPERANDO SEÑAL", "⏳", "#d29922"
    else:
        box_cls, sig_txt, sig_icon, sig_color = "signal-wait", "ANALIZANDO", "🔍", "#d29922"

    n_ops    = len(closed)
    n_wins   = len([t for t in closed if (t.pnl or 0) > 0])
    win_rate = n_wins / n_ops * 100 if n_ops > 0 else 0

    st.markdown(f"""
    <div class="signal-box {box_cls}">
      <div class="signal-label">Estado del Bot</div>
      <div class="signal-value" style="color:{sig_color}">{sig_icon} {sig_txt}</div>
      <div class="signal-conf">Análisis cada hora en punto &nbsp;·&nbsp; Win Rate: {win_rate:.0f}%</div>
    </div>""", unsafe_allow_html=True)

    # ── Portfolio ─────────────────────────────────────────────────────────────
    st.markdown("<div class='section-hdr'>Portfolio</div>", unsafe_allow_html=True)

    pnl_c = "#3fb950" if total_pnl >= 0 else "#f85149"
    hoy_c = "#3fb950" if today_pnl >= 0 else "#f85149"
    dd_c  = "#3fb950" if drawdown > -10 else "#f85149"

    st.markdown(f"""
    <div class="metric-card" title="Todo tu dinero actual: lo que está libre + lo que está invertido en BTC ahora mismo.">
      <div class="metric-label">Valor Total ℹ️</div>
      <div class="metric-value blue">${total_val:,.2f}</div>
      <div class="metric-sub">Capital inicial: ${initial_capital:,.0f}</div>
    </div>
    <div class="metric-card" title="Ganancia o pérdida total desde que arrancó el bot. Verde = ganando. Rojo = perdiendo.">
      <div class="metric-label">PnL Total ℹ️</div>
      <div class="metric-value" style="color:{pnl_c}">${total_pnl:+.2f}</div>
      <div class="metric-sub">{total_pnl_pct:+.2f}%</div>
    </div>
    <div class="metric-card" title="Cuánto ganaste o perdiste solo hoy. Se reinicia cada día a las 12 de la noche.">
      <div class="metric-label">PnL Hoy ℹ️</div>
      <div class="metric-value" style="color:{hoy_c}">${today_pnl:+.2f}</div>
    </div>
    <div class="metric-card" title="Cuánto bajó tu capital desde su punto más alto. Si llega a -20% el bot se detiene automáticamente para proteger tu dinero.">
      <div class="metric-label">Drawdown ℹ️</div>
      <div class="metric-value" style="color:{dd_c}">{drawdown:.1f}%</div>
      <div class="metric-sub">límite: −20%</div>
    </div>""", unsafe_allow_html=True)

    # ── Posición Abierta ──────────────────────────────────────────────────────
    if open_trades:
        t = open_trades[0]
        unreal = (price / t.price - 1) * 100 if price > 0 and t.price > 0 else 0
        u_c = "#3fb950" if unreal >= 0 else "#f85149"
        st.markdown("<div class='section-hdr'>Posición Abierta</div>", unsafe_allow_html=True)
        st.markdown(f"""
        <div style="background:#161b22;border:1px solid #30363d;border-radius:10px;padding:12px">
          <div class="stat-row">
            <span class="stat-key" title="Precio al que el bot compró Bitcoin">Entrada ℹ️</span>
            <span class="stat-val">${t.price:,.2f}</span></div>
          <div class="stat-row">
            <span class="stat-key" title="Cuántos Bitcoin compró el bot con tu dinero">Cantidad ℹ️</span>
            <span class="stat-val">{t.quantity:.5f} BTC</span></div>
          <div class="stat-row">
            <span class="stat-key" title="Si el precio cae hasta aquí, el bot vende automáticamente para evitar perder más. Pérdida máxima: 2%">Stop Loss ℹ️</span>
            <span class="stat-val red">${t.stop_loss:,.2f}</span></div>
          <div class="stat-row">
            <span class="stat-key" title="Si el precio sube hasta aquí, el bot vende automáticamente y registra la ganancia. Ganancia objetivo: 5%">Take Profit ℹ️</span>
            <span class="stat-val green">${t.take_profit:,.2f}</span></div>
          <div class="stat-row" style="border:none">
            <span class="stat-key" title="Cuánto estás ganando o perdiendo en esta operación ahora mismo. Cambia con el precio de BTC. Solo se hace real cuando el bot vende.">PnL no realizado ℹ️</span>
            <span class="stat-val" style="color:{u_c}">{unreal:+.2f}%</span></div>
        </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# SEGUNDA FILA: Estadísticas + Curva de Equity
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("<hr style='border-color:#30363d;margin:12px 0'>", unsafe_allow_html=True)
col_stats, col_eq = st.columns([1, 2])

with col_stats:
    st.markdown("<div class='section-hdr'>Estadísticas</div>", unsafe_allow_html=True)
    if closed:
        wins   = [t for t in closed if (t.pnl or 0) > 0]
        losses = [t for t in closed if (t.pnl or 0) <= 0]
        wr     = len(wins) / len(closed) * 100
        total_cl_pnl = sum(t.pnl or 0 for t in closed)
        avg_win  = sum(t.pnl or 0 for t in wins)  / len(wins)  if wins   else 0
        avg_loss = sum(t.pnl or 0 for t in losses) / len(losses) if losses else 0
        gross_win  = sum(t.pnl or 0 for t in wins)
        gross_loss = abs(sum(t.pnl or 0 for t in losses))
        pf = gross_win / gross_loss if gross_loss > 0 else 0
        total_commissions = sum(t.commission_paid or 0 for t in closed)
        avg_commission = total_commissions / len(closed) if closed else 0
        wr_c  = "#3fb950" if wr >= 50 else "#f85149"
        pf_c  = "#3fb950" if pf >= 1  else "#f85149"
        pnl_c = "#3fb950" if total_cl_pnl >= 0 else "#f85149"
        st.markdown(f"""
        <div style="font-size:0.84rem">
          <div class="stat-row"><span class="stat-key">Operaciones</span>
            <span class="stat-val">{len(closed)}</span></div>
          <div class="stat-row"><span class="stat-key">Win Rate</span>
            <span class="stat-val" style="color:{wr_c}">{wr:.1f}%</span></div>
          <div class="stat-row"><span class="stat-key">✅ Ganadoras</span>
            <span class="stat-val green">{len(wins)}</span></div>
          <div class="stat-row"><span class="stat-key">❌ Perdedoras</span>
            <span class="stat-val red">{len(losses)}</span></div>
          <div class="stat-row"><span class="stat-key">PnL neto total</span>
            <span class="stat-val" style="color:{pnl_c}">${total_cl_pnl:+.2f}</span></div>
          <div class="stat-row"><span class="stat-key">Ganancia media</span>
            <span class="stat-val green">${avg_win:.2f}</span></div>
          <div class="stat-row"><span class="stat-key">Pérdida media</span>
            <span class="stat-val red">${avg_loss:.2f}</span></div>
          <div class="stat-row"><span class="stat-key">Profit Factor</span>
            <span class="stat-val" style="color:{pf_c}">{pf:.2f}</span></div>
          <div class="stat-row"><span class="stat-key" title="Total pagado en comisiones a Binance (0.1% compra + 0.1% venta)">💸 Comisiones pagadas</span>
            <span class="stat-val" style="color:#d29922">${total_commissions:.4f}</span></div>
          <div class="stat-row" style="border:none"><span class="stat-key" title="Comisión promedio por operación">💸 Comisión media/op</span>
            <span class="stat-val" style="color:#d29922">${avg_commission:.4f}</span></div>
        </div>""", unsafe_allow_html=True)
    else:
        st.info("Sin operaciones cerradas aún.")

with col_eq:
    st.markdown("<div class='section-hdr'>Curva de Equity</div>", unsafe_allow_html=True)
    if snapshots:
        eq_df = pd.DataFrame([{"t": s.timestamp, "v": s.total_value} for s in snapshots])
        fig_eq = go.Figure()
        fig_eq.add_trace(go.Scatter(
            x=eq_df["t"], y=eq_df["v"],
            mode="lines", line=dict(color="#58a6ff", width=2),
            fill="tozeroy", fillcolor="rgba(88,166,255,0.06)",
        ))
        fig_eq.add_hline(y=initial_capital, line_dash="dash", line_color="#30363d",
                         annotation_text=f"${initial_capital:,.0f}",
                         annotation_font_color="#8b949e", annotation_font_size=10)
        fig_eq.update_layout(
            template="plotly_dark", height=200,
            paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
            margin=dict(l=0, r=0, t=0, b=0), showlegend=False,
            xaxis=dict(showgrid=True, gridcolor="#21262d"),
            yaxis=dict(showgrid=True, gridcolor="#21262d", side="right"),
        )
        st.plotly_chart(fig_eq, use_container_width=True,
                        config={"displayModeBar": False})
    else:
        st.info("Sin datos de equity aún.")

# ══════════════════════════════════════════════════════════════════════════════
# HISTORIAL DE OPERACIONES
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("<div class='section-hdr'>Historial de Operaciones</div>", unsafe_allow_html=True)
if trades:
    recent = sorted(trades, key=lambda t: t.timestamp or pd.Timestamp.min, reverse=True)[:25]
    rows = []
    for t in recent:
        if t.is_open:
            pnl_str = "🟢 Abierta"
            commission_str = "—"
        else:
            v = t.pnl or 0
            pnl_str = f"✅ +${v:.2f}" if v > 0 else f"❌ ${v:.2f}"
            commission_str = f"${(t.commission_paid or 0):.4f}"
        rows.append({
            "Fecha":      t.timestamp.strftime("%d/%m %H:%M") if t.timestamp else "—",
            "Par":        t.symbol,
            "Entrada":    f"${t.price:,.2f}",
            "Salida":     f"${t.close_price:,.2f}" if t.close_price else "—",
            "BTC":        f"{t.quantity:.5f}",
            "Comisión":   commission_str,
            "PnL neto":   pnl_str,
            "Cierre":     t.close_reason or "—",
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True,
                 hide_index=True, height=280)
else:
    st.info("Sin operaciones aún. El bot analiza el mercado cada hora.")

# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN INFERIOR
# ══════════════════════════════════════════════════════════════════════════════
c_bt, c_tg = st.columns(2)
with c_bt:
    with st.expander("📊 Resultado del Backtesting"):
        img = ROOT / "backtesting" / "results" / "backtest_result.png"
        if img.exists():
            st.image(str(img), use_container_width=True)
        else:
            st.info("Entrena el modelo para ver el backtest.")

with c_tg:
    with st.expander("📱 Configurar Telegram"):
        st.markdown("""
**Pasos para alertas en tu celular:**
1. Telegram → **@BotFather** → `/newbot` → guarda el token
2. Telegram → **@userinfobot** → guarda tu Chat ID
3. EasyPanel → Entorno → agrega:
```
TELEGRAM_TOKEN=tu_token
TELEGRAM_CHAT_ID=tu_id
TELEGRAM_ENABLED=true
```
4. Reinicia el servicio
""")

with st.expander("📖 ¿Qué significa cada número? — Glosario completo"):
    st.markdown("""
| Término | Qué significa en simple |
|---|---|
| **Valor Total** | Todo tu dinero ahora mismo: lo libre + lo que está invertido en BTC |
| **Capital inicial** | Con cuánto dinero arrancaste la simulación |
| **PnL Total** | Cuánto ganaste o perdiste desde el primer día. Verde = ganando ✅ |
| **PnL Hoy** | Cuánto ganaste o perdiste solo hoy. Se reinicia a medianoche |
| **PnL no realizado** | La ganancia/pérdida de una operación que todavía está abierta. Solo se hace real cuando el bot vende |
| **Drawdown** | Cuánto bajó tu dinero desde su punto más alto. Si llega a -20% el bot se detiene solo para protegerte |
| **EN POSICIÓN** 🟢 | El bot compró BTC y está esperando que suba para vender |
| **ANALIZANDO** 🔍 | El bot está mirando el mercado pero no ha encontrado una buena oportunidad aún |
| **ESPERANDO SEÑAL** ⏳ | La última operación terminó y el bot espera la próxima señal de compra |
| **Entrada** | Precio al que el bot compró Bitcoin |
| **Stop Loss** | Precio de emergencia: si BTC baja hasta ahí, el bot vende para no perder más (límite: -2%) |
| **Take Profit** | Precio objetivo: si BTC sube hasta ahí, el bot vende y toma la ganancia (+5%) |
| **Win Rate** | % de operaciones ganadoras. Ejemplo: 60% = de cada 10 operaciones, 6 terminaron en ganancia |
| **Profit Factor** | Relación entre lo que ganas y lo que pierdes. Mayor a 1.0 es rentable |
| **Sharpe Ratio** | Mide si las ganancias valen el riesgo. Mayor a 1.0 es bueno |
| **Buy & Hold** | Comparación: cuánto habrías ganado si solo comprabas BTC y lo dejabas sin hacer nada |
| **Max Drawdown** | La mayor caída que tuvo el portfolio en toda la historia del bot |
| **EMA 20 / 50 / 200** | Promedios del precio de BTC. El bot los usa para detectar tendencias |
| **RSI** | Indica si BTC está "sobrecomprado" (puede bajar) o "sobrevendido" (puede subir). Escala de 0 a 100 |
| **Volumen 24H** | Total de Bitcoin comprado y vendido en el mundo en las últimas 24 horas |
| **MÁX / MÍN 24H** | El precio más alto y más bajo que tuvo BTC en las últimas 24 horas |
| **Modo PAPER** 🟡 | Simulación con dinero ficticio. El bot opera como si fuera real pero sin arriesgar nada |
| **Modo LIVE** 🔴 | Dinero real. Solo activar cuando el bot lleve meses de simulación exitosa |
""")

st.markdown("<hr style='border-color:#30363d;margin:12px 0'>", unsafe_allow_html=True)
st.markdown(
    f"<div style='text-align:center;font-size:0.72rem;color:#8b949e'>"
    f"AsistenteTrading Pro &nbsp;·&nbsp; Modo {mode.upper()} &nbsp;·&nbsp; {symbol} &nbsp;·&nbsp; "
    f"Los resultados pasados no garantizan resultados futuros."
    f"</div>",
    unsafe_allow_html=True,
)
