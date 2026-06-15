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
    page_title="BTC/USDT — AsistenteTrading",
    page_icon="🟡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── CSS estilo Binance ──────────────────────────────────────────────────────────
st.markdown("""
<style>
  /* ── Reset global Binance ── */
  html, body, [class*="css"], .stApp {
    background-color: #0B0E11 !important;
    font-family: -apple-system, 'Helvetica Neue', Arial, sans-serif !important;
    color: #EAECEF !important;
  }
  .block-container { padding-top: 0 !important; padding-bottom: 0 !important; }
  #MainMenu, footer, header { visibility: hidden; }
  .stDeployButton { display: none; }

  /* ── Header Binance ── */
  .bn-header {
    background: #181A20;
    border-bottom: 1px solid #2B2F36;
    padding: 8px 0 6px 0;
    margin-bottom: 0;
  }
  .bn-logo {
    font-size: 1.15rem; font-weight: 900;
    color: #F0B90B; letter-spacing: 0.5px;
  }
  .bn-pair {
    font-size: 1.0rem; font-weight: 700; color: #EAECEF;
    display: inline-block; margin-left: 10px;
  }
  .bn-price {
    font-size: 1.75rem; font-weight: 700;
    color: #EAECEF; font-variant-numeric: tabular-nums;
    line-height: 1.1;
  }
  .bn-change-pos { color: #0ECB81; font-size: 0.88rem; font-weight: 600; }
  .bn-change-neg { color: #F6465D; font-size: 0.88rem; font-weight: 600; }

  /* ── Barra de stats 24h ── */
  .bn-stats-bar {
    background: #181A20;
    border-bottom: 1px solid #2B2F36;
    padding: 6px 0 8px 0;
    display: flex; gap: 32px; flex-wrap: wrap;
    font-size: 0.75rem;
  }
  .bn-stat-item {}
  .bn-stat-label { color: #848E9C; margin-bottom: 1px; }
  .bn-stat-value { color: #EAECEF; font-weight: 600; }
  .bn-stat-value.pos { color: #0ECB81; }
  .bn-stat-value.neg { color: #F6465D; }
  .bn-stat-value.gold { color: #F0B90B; }

  /* ── Panel derecho: estilo orden Binance ── */
  .bn-panel {
    background: #181A20;
    border: 1px solid #2B2F36;
    border-radius: 4px;
    padding: 0;
    overflow: hidden;
  }
  .bn-panel-tab {
    background: #181A20;
    border-bottom: 2px solid #2B2F36;
    padding: 10px 16px;
    font-size: 0.82rem; font-weight: 700;
    color: #F0B90B;
    letter-spacing: 0.5px;
  }
  .bn-panel-body { padding: 12px 14px; }

  /* ── Botón de estado (tipo BUY/SELL Binance) ── */
  .bn-btn-buy {
    width: 100%; padding: 11px 0;
    background: #0ECB81; border-radius: 4px;
    text-align: center; font-size: 0.88rem;
    font-weight: 700; color: #fff;
    letter-spacing: 0.5px; margin-bottom: 14px;
    cursor: default;
  }
  .bn-btn-sell {
    width: 100%; padding: 11px 0;
    background: #F6465D; border-radius: 4px;
    text-align: center; font-size: 0.88rem;
    font-weight: 700; color: #fff;
    letter-spacing: 0.5px; margin-bottom: 14px;
    cursor: default;
  }
  .bn-btn-wait {
    width: 100%; padding: 11px 0;
    background: #2B2F36; border-radius: 4px;
    text-align: center; font-size: 0.88rem;
    font-weight: 700; color: #848E9C;
    letter-spacing: 0.5px; margin-bottom: 14px;
    cursor: default;
  }

  /* ── Filas de datos ── */
  .bn-row {
    display: flex; justify-content: space-between; align-items: center;
    padding: 5px 0;
    border-bottom: 1px solid #2B2F36;
    font-size: 0.78rem;
  }
  .bn-row:last-child { border-bottom: none; }
  .bn-row-key { color: #848E9C; }
  .bn-row-val { color: #EAECEF; font-weight: 600; font-variant-numeric: tabular-nums; }
  .bn-row-val.pos { color: #0ECB81; }
  .bn-row-val.neg { color: #F6465D; }
  .bn-row-val.gold { color: #F0B90B; }

  /* ── Divider tipo Binance ── */
  .bn-divider {
    border: none; border-top: 1px solid #2B2F36;
    margin: 10px 0;
  }

  /* ── Sección header ── */
  .bn-section-title {
    font-size: 0.7rem; font-weight: 700;
    color: #848E9C; text-transform: uppercase;
    letter-spacing: 1.5px;
    padding: 8px 14px;
    background: #1E2026;
    border-bottom: 1px solid #2B2F36;
    border-top: 1px solid #2B2F36;
    margin: 8px 0 0 0;
  }

  /* ── Métrica cuadrada ── */
  .bn-metric {
    background: #181A20;
    border: 1px solid #2B2F36;
    border-radius: 4px;
    padding: 10px 12px;
    margin-bottom: 6px;
  }
  .bn-metric-label {
    font-size: 0.68rem; color: #848E9C;
    text-transform: uppercase; letter-spacing: 1px;
    margin-bottom: 2px;
  }
  .bn-metric-value {
    font-size: 1.3rem; font-weight: 700;
    color: #EAECEF; font-variant-numeric: tabular-nums;
    line-height: 1.2;
  }
  .bn-metric-sub { font-size: 0.7rem; color: #848E9C; margin-top: 1px; }

  /* ── Tabla historial ── */
  .bn-table { width: 100%; border-collapse: collapse; font-size: 0.76rem; }
  .bn-table th {
    color: #848E9C; text-transform: uppercase;
    font-size: 0.65rem; font-weight: 600; letter-spacing: 0.8px;
    padding: 6px 8px; border-bottom: 1px solid #2B2F36;
    text-align: left;
  }
  .bn-table td {
    padding: 6px 8px; border-bottom: 1px solid #2B2F36;
    color: #EAECEF;
  }
  .bn-table tr:last-child td { border-bottom: none; }
  .bn-table tr:hover td { background: #1E2026; }

  /* ── Badge modo ── */
  .bn-badge-paper {
    background: #2B2F36; color: #F0B90B;
    font-size: 0.65rem; font-weight: 700;
    padding: 2px 8px; border-radius: 3px;
    letter-spacing: 0.5px;
  }
  .bn-badge-live {
    background: rgba(246,70,93,0.15); color: #F6465D;
    font-size: 0.65rem; font-weight: 700;
    padding: 2px 8px; border-radius: 3px;
    letter-spacing: 0.5px;
  }

  /* ── Streamlit overrides ── */
  div[data-testid="stButton"] > button {
    background: #2B2F36 !important;
    border: 1px solid #474D57 !important;
    color: #EAECEF !important;
    border-radius: 4px !important;
    font-size: 0.78rem !important;
    font-weight: 600 !important;
    padding: 4px 12px !important;
  }
  div[data-testid="stButton"] > button:hover {
    background: #474D57 !important;
    border-color: #848E9C !important;
  }
  div[data-testid="stDataFrame"] {
    background: #181A20 !important;
    border: 1px solid #2B2F36 !important;
    border-radius: 4px !important;
  }
  .stDataFrame td, .stDataFrame th {
    background: #181A20 !important;
    color: #EAECEF !important;
    font-size: 0.76rem !important;
  }
  div[data-testid="stExpander"] {
    background: #181A20 !important;
    border: 1px solid #2B2F36 !important;
    border-radius: 4px !important;
  }
  div[data-testid="stExpander"] summary {
    color: #EAECEF !important;
    font-size: 0.82rem !important;
  }
  .streamlit-expanderHeader { color: #EAECEF !important; }
  [data-testid="stInfo"] {
    background: #1E2026 !important;
    border: 1px solid #2B2F36 !important;
    border-radius: 4px !important;
    color: #848E9C !important;
    font-size: 0.78rem !important;
  }
</style>
""", unsafe_allow_html=True)


# ── Helpers ────────────────────────────────────────────────────────────────────
def _squeeze(series):
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

        df["EMA20"]  = close.ewm(span=20).mean()
        df["EMA50"]  = close.ewm(span=50).mean()
        df["EMA200"] = close.ewm(span=200).mean()

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
        "#0ECB81" if float(c) >= float(o) else "#F6465D"
        for c, o in zip(close_, open_)
    ]

    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.018,
        row_heights=[0.60, 0.18, 0.22],
    )

    # Velas estilo Binance
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=open_, high=high_, low=low_, close=close_,
        name="BTC/USDT",
        increasing_line_color="#0ECB81", increasing_fillcolor="#0ECB81",
        decreasing_line_color="#F6465D", decreasing_fillcolor="#F6465D",
        line_width=1,
        whiskerwidth=0.8,
    ), row=1, col=1)

    # EMAs estilo Binance
    for col_name, color, label, width in [
        ("EMA20",  "#F0B90B", "EMA 20",  1.2),
        ("EMA50",  "#8B72FF", "EMA 50",  1.2),
        ("EMA200", "#FF6B35", "EMA 200", 1.2),
    ]:
        if col_name in df.columns:
            fig.add_trace(go.Scatter(
                x=df.index, y=_squeeze(df[col_name]),
                name=label, line=dict(color=color, width=width),
                hovertemplate=f"{label}: $%{{y:,.0f}}<extra></extra>",
            ), row=1, col=1)

    # Volumen
    fig.add_trace(go.Bar(
        x=df.index, y=vol_,
        name="Vol", marker_color=colors_vol, opacity=0.7, showlegend=False,
    ), row=2, col=1)

    # RSI
    if "RSI" in df.columns:
        rsi_ = _squeeze(df["RSI"])
        fig.add_trace(go.Scatter(
            x=df.index, y=rsi_,
            name="RSI 14", line=dict(color="#F0B90B", width=1.4),
            hovertemplate="RSI: %{y:.1f}<extra></extra>",
        ), row=3, col=1)
        fig.add_hrect(y0=70, y1=100, fillcolor="rgba(246,70,93,0.06)",
                      line_width=0, row=3, col=1)
        fig.add_hrect(y0=0, y1=30, fillcolor="rgba(14,203,129,0.06)",
                      line_width=0, row=3, col=1)
        for y_val, color in [(70, "#F6465D"), (30, "#0ECB81"), (50, "#474D57")]:
            fig.add_hline(y=y_val, line_dash="dot", line_color=color,
                          line_width=0.7, row=3, col=1)

    grid = "#2B2F36"
    fig.update_layout(
        template="plotly_dark",
        height=540,
        paper_bgcolor="#0B0E11", plot_bgcolor="#0B0E11",
        margin=dict(l=0, r=55, t=6, b=0),
        showlegend=True,
        legend=dict(
            orientation="h", x=0, y=1.03,
            font=dict(size=9, color="#848E9C"),
            bgcolor="rgba(0,0,0,0)",
        ),
        xaxis=dict(
            showgrid=True, gridcolor=grid,
            rangeslider_visible=False,
            showspikes=True, spikecolor="#848E9C", spikethickness=1,
            spikemode="across",
        ),
        xaxis2=dict(showgrid=True, gridcolor=grid),
        xaxis3=dict(showgrid=True, gridcolor=grid),
        yaxis=dict(
            showgrid=True, gridcolor=grid, side="right",
            title="USD", title_font_size=9,
            title_font_color="#848E9C",
        ),
        yaxis2=dict(showgrid=False, side="right",
                    title="Vol", title_font_size=8, title_font_color="#848E9C"),
        yaxis3=dict(
            showgrid=True, gridcolor=grid, side="right",
            title="RSI", title_font_size=8, title_font_color="#848E9C",
            range=[0, 100],
        ),
        hoverlabel=dict(
            bgcolor="#1E2026", font_color="#EAECEF",
            font_size=11, bordercolor="#2B2F36",
        ),
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
# HEADER — estilo Binance
# ══════════════════════════════════════════════════════════════════════════════
ch_class  = "pos" if ch >= 0 else "neg"
ch_arrow  = "▲" if ch >= 0 else "▼"
price_str = f"${price:,.2f}" if price > 0 else "—"
model_ok  = is_model_trained()
badge_cls = "bn-badge-paper" if mode == "paper" else "bn-badge-live"
badge_txt = "PAPER" if mode == "paper" else "LIVE"
model_color = "#0ECB81" if model_ok else "#F6465D"
model_text  = "IA Activa" if model_ok else "Sin modelo"

hA, hB, hC, hD = st.columns([2, 2, 5, 2])

with hA:
    st.markdown(f"""
    <div class="bn-header" style="padding-left:4px">
      <div class="bn-logo">&#9650; AsistenteTrading</div>
      <div style="margin-top:2px">
        <span class="bn-pair">{symbol}</span>
        &nbsp;<span class="{badge_cls}">{badge_txt}</span>
      </div>
    </div>""", unsafe_allow_html=True)

with hB:
    ch_color = "#0ECB81" if ch >= 0 else "#F6465D"
    st.markdown(f"""
    <div class="bn-header" style="padding-left:4px">
      <div class="bn-price" style="color:{ch_color}">{price_str}</div>
      <div style="margin-top:2px">
        <span class="bn-change-{'pos' if ch >= 0 else 'neg'}">
          {ch_arrow} ${abs(ch):,.2f} &nbsp; {ch_pct:+.2f}%
        </span>
      </div>
    </div>""", unsafe_allow_html=True)

with hC:
    st.markdown(f"""
    <div class="bn-header">
      <div class="bn-stats-bar">
        <div class="bn-stat-item">
          <div class="bn-stat-label">24H Change</div>
          <div class="bn-stat-value {ch_class}">{ch_pct:+.2f}%</div>
        </div>
        <div class="bn-stat-item">
          <div class="bn-stat-label">24H High</div>
          <div class="bn-stat-value pos">${high_24h:,.0f}</div>
        </div>
        <div class="bn-stat-item">
          <div class="bn-stat-label">24H Low</div>
          <div class="bn-stat-value neg">${low_24h:,.0f}</div>
        </div>
        <div class="bn-stat-item">
          <div class="bn-stat-label">24H Volume</div>
          <div class="bn-stat-value">${vol_24h/1e9:.2f}B</div>
        </div>
        <div class="bn-stat-item">
          <div class="bn-stat-label">Portfolio</div>
          <div class="bn-stat-value gold">${total_val:,.2f}</div>
        </div>
        <div class="bn-stat-item">
          <div class="bn-stat-label">PnL Total</div>
          <div class="bn-stat-value {'pos' if total_pnl >= 0 else 'neg'}">${total_pnl:+.2f}</div>
        </div>
        <div class="bn-stat-item">
          <div class="bn-stat-label">Modelo IA</div>
          <div class="bn-stat-value" style="color:{model_color}">{model_text}</div>
        </div>
      </div>
    </div>""", unsafe_allow_html=True)

with hD:
    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
    if st.button("⟳  Actualizar", use_container_width=True):
        st.cache_data.clear()
        st.rerun()
    st.markdown(
        f"<div style='font-size:0.65rem;color:#474D57;text-align:right;margin-top:3px'>"
        f"{datetime.now().strftime('%d %b %Y  %H:%M')}</div>",
        unsafe_allow_html=True,
    )

st.markdown("<div style='border-bottom:1px solid #2B2F36;margin:0 0 8px 0'></div>",
            unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# LAYOUT PRINCIPAL: Gráfico (izq 70%) + Panel (der 30%)
# ══════════════════════════════════════════════════════════════════════════════
col_chart, col_panel = st.columns([70, 30])

with col_chart:
    fig = build_chart(df_mkt)
    if fig:
        st.plotly_chart(fig, use_container_width=True, config={
            "displayModeBar": True,
            "displaylogo": False,
            "modeBarButtonsToRemove": ["lasso2d", "select2d", "autoScale2d"],
            "toImageButtonOptions": {"format": "png", "filename": "btc_chart"},
        })
    else:
        st.markdown(
            "<div style='padding:40px;text-align:center;color:#848E9C;font-size:0.85rem'>"
            "No se pudieron cargar los datos del mercado.</div>",
            unsafe_allow_html=True,
        )

with col_panel:
    # ── Estado del Bot (tipo panel de orden Binance) ──────────────────────────
    n_ops  = len(closed)
    n_wins = len([t for t in closed if (t.pnl or 0) > 0])
    wr     = n_wins / n_ops * 100 if n_ops > 0 else 0

    if open_trades:
        btn_cls  = "bn-btn-buy"
        btn_txt  = "▶ EN POSICIÓN — COMPRANDO"
        state_detail = "Posición BTC abierta"
    elif closed:
        last_cl = sorted(closed, key=lambda t: t.timestamp or pd.Timestamp.min)[-1]
        if last_cl.close_reason == "take_profit":
            btn_cls, btn_txt = "bn-btn-buy", "✓ TAKE PROFIT — GANANCIA"
        elif last_cl.close_reason == "stop_loss":
            btn_cls, btn_txt = "bn-btn-sell", "⬛ STOP LOSS — PROTEGIDO"
        else:
            btn_cls, btn_txt = "bn-btn-wait", "⏸ ESPERANDO SEÑAL"
        state_detail = "Analizando próxima oportunidad"
    else:
        btn_cls, btn_txt = "bn-btn-wait", "⌛ ANALIZANDO MERCADO"
        state_detail = "El bot analiza cada hora en punto"

    pnl_c  = "pos" if total_pnl >= 0 else "neg"
    hoy_c  = "pos" if today_pnl >= 0 else "neg"
    dd_c   = "pos" if drawdown > -10 else "neg"
    pct_c  = "pos" if total_pnl_pct >= 0 else "neg"

    _btn_color = {"bn-btn-buy": "#0ECB81", "bn-btn-sell": "#F6465D", "bn-btn-wait": "#474D57"}.get(btn_cls, "#474D57")
    _btn_text_c = "#fff" if btn_cls != "bn-btn-wait" else "#848E9C"

    st.markdown(
        f'<div style="background:#181A20;border:1px solid #2B2F36;border-radius:4px;overflow:hidden">'
        f'<div style="background:#1E2026;border-bottom:1px solid #2B2F36;padding:10px 14px;'
        f'font-size:0.75rem;font-weight:700;color:#F0B90B;letter-spacing:0.5px">⚡ PANEL DE CONTROL</div>'
        f'<div style="padding:12px 14px">'
        f'<div style="width:100%;padding:11px 0;background:{_btn_color};border-radius:4px;'
        f'text-align:center;font-size:0.85rem;font-weight:700;color:{_btn_text_c};'
        f'letter-spacing:0.5px;margin-bottom:6px">{btn_txt}</div>'
        f'<div style="font-size:0.68rem;color:#848E9C;text-align:center;margin-bottom:12px">'
        f'{state_detail} &nbsp;·&nbsp; Win Rate: <span style="color:#F0B90B;font-weight:700">{wr:.0f}%</span></div>'
        f'</div></div>',
        unsafe_allow_html=True,
    )

    def _row(label, value, color="#EAECEF", border=True, tooltip=""):
        tip = f' title="{tooltip}"' if tooltip else ""
        brd = "border-bottom:1px solid #2B2F36;" if border else ""
        return (
            f'<div style="display:flex;justify-content:space-between;align-items:center;'
            f'padding:6px 14px;{brd}font-size:0.78rem">'
            f'<span style="color:#848E9C"{tip}>{label}</span>'
            f'<span style="color:{color};font-weight:600;font-variant-numeric:tabular-nums">{value}</span>'
            f'</div>'
        )

    pnl_color = "#0ECB81" if total_pnl >= 0 else "#F6465D"
    hoy_color = "#0ECB81" if today_pnl >= 0 else "#F6465D"
    dd_color  = "#0ECB81" if drawdown > -10 else "#F6465D"

    rows_html = (
        _row("Valor Total",    f"${total_val:,.2f}",  "#F0B90B", tooltip="Capital libre + valor invertido en BTC")
        + _row("Capital inicial", f"${initial_capital:,.2f}")
        + _row("PnL Total",    f"${total_pnl:+.2f} ({total_pnl_pct:+.2f}%)", pnl_color, tooltip="Ganancia/pérdida desde el inicio")
        + _row("PnL Hoy",      f"${today_pnl:+.2f}", hoy_color, tooltip="Ganancia/pérdida solo hoy")
        + _row("Efectivo libre", f"${cash:,.2f}")
        + _row("Drawdown",     f"{drawdown:.1f}%", dd_color, tooltip="Límite -20%, el bot se protege solo")
        + _row("Operaciones",  f"{n_ops} cerradas · {len(open_trades)} abierta", border=False)
    )
    st.markdown(
        f'<div style="background:#181A20;border:1px solid #2B2F36;border-radius:4px;'
        f'margin-top:8px;overflow:hidden">{rows_html}</div>',
        unsafe_allow_html=True,
    )

    # ── Posición abierta (si existe) ──────────────────────────────────────────
    if open_trades:
        t = open_trades[0]
        unreal = (price / t.price - 1) * 100 if price > 0 and t.price > 0 else 0
        unreal_usd = (price - t.price) * t.quantity if price > 0 else 0

        price_now_color = "#0ECB81" if price >= t.price else "#F6465D"
        unreal_color    = "#0ECB81" if unreal >= 0 else "#F6465D"
        pos_rows = (
            _row("Precio entrada", f"${t.price:,.2f}", tooltip="Precio al que el bot compró BTC")
            + _row("Precio actual",  f"${price:,.2f}", price_now_color)
            + _row("Cantidad",       f"{t.quantity:.6f} BTC", tooltip="Cuántos Bitcoin compró el bot")
            + _row("Valor invertido", f"${t.value_usd:,.2f}")
            + _row("Stop Loss",      f"${t.stop_loss:,.2f}", "#F6465D", tooltip="Si BTC baja aquí el bot vende (-2%)")
            + _row("Take Profit",    f"${t.take_profit:,.2f}", "#0ECB81", tooltip="Si BTC sube aquí el bot vende (+5%)")
            + _row("PnL no realizado", f"{unreal:+.2f}% (${unreal_usd:+.2f})", unreal_color, border=False,
                   tooltip="Solo se hace real cuando el bot vende")
        )
        st.markdown(
            f'<div style="background:#181A20;border:1px solid #2B2F36;border-radius:4px;'
            f'margin-top:8px;overflow:hidden">'
            f'<div style="background:#1E2026;border-bottom:1px solid #2B2F36;padding:9px 14px;'
            f'font-size:0.72rem;font-weight:700;color:#F0B90B;letter-spacing:0.5px">'
            f'📊 POSICIÓN ABIERTA — BTC/USDT</div>'
            f'{pos_rows}</div>',
            unsafe_allow_html=True,
        )

# ══════════════════════════════════════════════════════════════════════════════
# SEGUNDA FILA: Estadísticas + Curva de Equity
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("<div style='border-top:1px solid #2B2F36;margin:10px 0 8px 0'></div>",
            unsafe_allow_html=True)

col_stats, col_eq = st.columns([1, 2])

with col_stats:
    st.markdown(
        "<div style='font-size:0.7rem;font-weight:700;color:#848E9C;text-transform:uppercase;"
        "letter-spacing:1.5px;padding:8px 14px;background:#1E2026;border-top:1px solid #2B2F36;"
        "border-bottom:1px solid #2B2F36;margin-bottom:0'>Estadísticas del Bot</div>",
        unsafe_allow_html=True,
    )
    if closed:
        wins   = [t for t in closed if (t.pnl or 0) > 0]
        losses = [t for t in closed if (t.pnl or 0) <= 0]
        wr_s   = len(wins) / len(closed) * 100
        total_cl_pnl = sum(t.pnl or 0 for t in closed)
        avg_win  = sum(t.pnl or 0 for t in wins)   / len(wins)   if wins   else 0
        avg_loss = sum(t.pnl or 0 for t in losses)  / len(losses) if losses else 0
        gross_win  = sum(t.pnl or 0 for t in wins)
        gross_loss = abs(sum(t.pnl or 0 for t in losses))
        pf = gross_win / gross_loss if gross_loss > 0 else 0
        total_commissions = sum(t.commission_paid or 0 for t in closed)
        avg_commission    = total_commissions / len(closed) if closed else 0
        wr_c  = "#0ECB81" if wr_s >= 50 else "#F6465D"
        pf_c  = "#0ECB81" if pf >= 1   else "#F6465D"
        pc    = "#0ECB81" if total_cl_pnl >= 0 else "#F6465D"

        def _srow(label, value, color="#EAECEF", border=True, tip=""):
            t2 = f' title="{tip}"' if tip else ""
            brd = "border-bottom:1px solid #2B2F36;" if border else ""
            return (
                f'<div style="display:flex;justify-content:space-between;padding:5px 14px;'
                f'{brd}font-size:0.78rem">'
                f'<span style="color:#848E9C"{t2}>{label}</span>'
                f'<span style="color:{color};font-weight:600">{value}</span></div>'
            )

        stats_html = (
            _srow("Total operaciones", str(len(closed)))
            + _srow("Win Rate",         f"{wr_s:.1f}%",       wr_c)
            + _srow("✅ Ganadoras",      str(len(wins)),        "#0ECB81")
            + _srow("❌ Perdedoras",     str(len(losses)),      "#F6465D")
            + _srow("PnL neto total",    f"${total_cl_pnl:+.2f}", pc)
            + _srow("Ganancia media/op", f"${avg_win:.3f}",     "#0ECB81")
            + _srow("Pérdida media/op",  f"${avg_loss:.3f}",    "#F6465D")
            + _srow("Profit Factor",     f"{pf:.2f}",           pf_c)
            + _srow("Comisiones totales", f"${total_commissions:.4f}", "#F0B90B",
                    tip="Total pagado a Binance (0.1% compra + 0.1% venta)")
            + _srow("Comisión media/op", f"${avg_commission:.4f}", "#F0B90B", border=False)
        )
        st.markdown(
            f'<div style="background:#181A20;border:1px solid #2B2F36;border-radius:4px;'
            f'margin-top:4px">{stats_html}</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            "<div style='padding:16px;color:#848E9C;font-size:0.8rem'>"
            "Sin operaciones cerradas aún.</div>",
            unsafe_allow_html=True,
        )

with col_eq:
    st.markdown(
        "<div style='font-size:0.7rem;font-weight:700;color:#848E9C;text-transform:uppercase;"
        "letter-spacing:1.5px;padding:8px 14px;background:#1E2026;border-top:1px solid #2B2F36;"
        "border-bottom:1px solid #2B2F36;margin-bottom:0'>Curva de Equity</div>",
        unsafe_allow_html=True,
    )
    if snapshots:
        eq_df = pd.DataFrame([{"t": s.timestamp, "v": s.total_value} for s in snapshots])
        fig_eq = go.Figure()
        fig_eq.add_trace(go.Scatter(
            x=eq_df["t"], y=eq_df["v"],
            mode="lines", line=dict(color="#F0B90B", width=2),
            fill="tozeroy", fillcolor="rgba(240,185,11,0.07)",
            hovertemplate="$%{y:,.2f}<extra></extra>",
        ))
        fig_eq.add_hline(
            y=initial_capital,
            line_dash="dash", line_color="#2B2F36", line_width=1,
            annotation_text=f"  Inicial: ${initial_capital:,.0f}",
            annotation_font_color="#848E9C", annotation_font_size=9,
        )
        fig_eq.update_layout(
            template="plotly_dark", height=195,
            paper_bgcolor="#0B0E11", plot_bgcolor="#0B0E11",
            margin=dict(l=0, r=0, t=4, b=0), showlegend=False,
            xaxis=dict(showgrid=True, gridcolor="#2B2F36", showticklabels=True,
                       tickfont=dict(size=9, color="#848E9C")),
            yaxis=dict(showgrid=True, gridcolor="#2B2F36", side="right",
                       tickfont=dict(size=9, color="#848E9C")),
            hoverlabel=dict(bgcolor="#1E2026", font_color="#EAECEF",
                            bordercolor="#2B2F36"),
        )
        st.plotly_chart(fig_eq, use_container_width=True,
                        config={"displayModeBar": False})
    else:
        st.markdown(
            "<div style='padding:16px;color:#848E9C;font-size:0.8rem'>"
            "Sin datos de equity aún.</div>",
            unsafe_allow_html=True,
        )

# ══════════════════════════════════════════════════════════════════════════════
# HISTORIAL DE OPERACIONES — tabla estilo Binance
# ══════════════════════════════════════════════════════════════════════════════
st.markdown(
    "<div style='border-top:1px solid #2B2F36;margin:10px 0 0 0'></div>",
    unsafe_allow_html=True,
)
st.markdown(
    "<div style='font-size:0.7rem;font-weight:700;color:#848E9C;text-transform:uppercase;"
    "letter-spacing:1.5px;padding:8px 14px;background:#1E2026;border-top:1px solid #2B2F36;"
    "border-bottom:1px solid #2B2F36;margin-bottom:4px'>Historial de Operaciones</div>",
    unsafe_allow_html=True,
)

if trades:
    recent = sorted(trades, key=lambda t: t.timestamp or pd.Timestamp.min, reverse=True)[:25]
    rows = []
    for t in recent:
        if t.is_open:
            pnl_str = "🟢 Abierta"
            comm_str = "—"
        else:
            v = t.pnl or 0
            pnl_str = f"+${v:.3f}" if v > 0 else f"${v:.3f}"
            comm_str = f"${(t.commission_paid or 0):.4f}"
        rows.append({
            "Fecha":     t.timestamp.strftime("%d/%m %H:%M") if t.timestamp else "—",
            "Par":       t.symbol,
            "Entrada":   f"${t.price:,.2f}",
            "Salida":    f"${t.close_price:,.2f}" if t.close_price else "—",
            "BTC":       f"{t.quantity:.6f}",
            "Comisión":  comm_str,
            "PnL neto":  pnl_str,
            "Cierre":    (t.close_reason or "—").replace("_", " ").upper(),
        })

    st.dataframe(
        pd.DataFrame(rows),
        use_container_width=True,
        hide_index=True,
        height=260,
        column_config={
            "PnL neto": st.column_config.TextColumn("PnL neto", width="small"),
            "Comisión": st.column_config.TextColumn("Comisión", width="small"),
        },
    )
else:
    st.markdown(
        "<div style='padding:20px;text-align:center;color:#848E9C;font-size:0.82rem'>"
        "Sin operaciones aún. El bot analiza el mercado cada hora.</div>",
        unsafe_allow_html=True,
    )

# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN INFERIOR — Expanders
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("<div style='border-top:1px solid #2B2F36;margin:10px 0 6px 0'></div>",
            unsafe_allow_html=True)

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
| **ANALIZANDO** ⌛ | El bot está mirando el mercado pero no ha encontrado una buena oportunidad aún |
| **ESPERANDO SEÑAL** ⏸ | La última operación terminó y el bot espera la próxima señal de compra |
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
| **PAPER** 🟡 | Simulación con dinero ficticio. El bot opera como si fuera real pero sin arriesgar nada |
| **LIVE** 🔴 | Dinero real. Solo activar cuando el bot lleve meses de simulación exitosa |
| **Comisión** | 0.1% que cobra Binance por cada compra + 0.1% por cada venta (0.2% total por operación) |
""")

# ── Footer estilo Binance ──────────────────────────────────────────────────────
st.markdown(
    f"<div style='border-top:1px solid #2B2F36;margin-top:12px;padding:8px 0;"
    f"text-align:center;font-size:0.65rem;color:#474D57'>"
    f"AsistenteTrading Pro &nbsp;·&nbsp; {symbol} &nbsp;·&nbsp; Modo {mode.upper()} "
    f"&nbsp;·&nbsp; Los resultados pasados no garantizan resultados futuros."
    f"</div>",
    unsafe_allow_html=True,
)
