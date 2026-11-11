"""댄 장거 돌파 스캐너 — 실행: streamlit run app.py"""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

import scanner as sc

st.set_page_config(page_title="장거 돌파 스캐너", page_icon="📈", layout="wide")
st.title("📈 댄 장거 돌파 스캐너")
st.caption("2개월 이상 횡보 → 상단 저항선 돌파 → 거래량 2배 확인 → 진입 (-3% 손절)")

MARKETS = ["미국주식", "한국주식", "바이낸스", "업비트"]

with st.sidebar:
    st.header("시장")
    markets = st.multiselect("스캔할 시장", MARKETS, default=MARKETS)
    us_which = st.selectbox("미국 유니버스", ["S&P 500 + 나스닥 100", "S&P 500", "나스닥 100"])
    kr_n = st.slider("한국: 시총 상위 N", 50, 1000, 300, 50)
    bn_n = st.slider("바이낸스: 거래대금 상위 N (USDT)", 20, 300, 100, 10)
    up_n = st.slider("업비트: 거래대금 상위 N (KRW)", 20, 200, 80, 10)

    st.header("조건")
    base_stock = st.slider("횡보 기간 – 주식 (거래일)", 20, 120, 42, 1, help="42거래일 ≈ 2개월")
    base_coin = st.slider("횡보 기간 – 코인 (일)", 20, 180, 60, 1, help="코인은 365일 거래, 60일 = 2개월")
    max_width = st.slider("박스 폭 최대 – 주식 (%)", 5, 60, 25, 1,
                          help="횡보 구간의 (최고가÷최저가−1). 작을수록 타이트한 횡보만")
    max_width_coin = st.slider("박스 폭 최대 – 코인 (%)", 10, 100, 40, 5)
    vol_mult = st.slider("거래량 배수 (평균 대비)", 1.0, 5.0, 2.0, 0.1)
    vol_len = st.slider("평균 거래량 기간 (봉)", 10, 60, 50, 5)
    recent = st.slider("최근 며칠 이내 돌파", 1, 10, 3, 1)
    stop_pct = st.slider("손절 폭 (%)", 1.0, 10.0, 3.0, 0.5)
    res_close = st.checkbox("저항선을 종가 기준으로 (꼬리 무시)", False)
    show_near = st.checkbox("돌파 임박 종목도 표시", True)
    near_pct = st.slider("임박 기준: 저항선 아래 (%)", 1.0, 10.0, 3.0, 0.5)
    run = st.button("🔍 스캔 실행", type="primary", use_container_width=True)


@st.cache_data(ttl=900, show_spinner=False)
def load(market: str, us_which: str, kr_n: int, bn_n: int, up_n: int):
    if market == "미국주식":
        t, names = sc.us_universe(us_which)
        return sc.yf_history(t), names
    if market == "한국주식":
        t, names = sc.kr_universe(kr_n)
        return sc.yf_history(t), names
    if market == "바이낸스":
        t, names = sc.binance_universe(bn_n)
        return sc.binance_history(t), names
    t, names = sc.upbit_universe(up_n)
    return sc.upbit_history(t), names


def params_for(market: str) -> sc.Params:
    coin = market in ("바이낸스", "업비트")
    return sc.Params(
        base_len=base_coin if coin else base_stock,
        max_width=(max_width_coin if coin else max_width) / 100,
        vol_len=vol_len, vol_mult=vol_mult, recent=recent,
        stop_pct=stop_pct / 100, res_by_close=res_close, near_pct=near_pct / 100,
    )


if run:
    rows, hist, errors = [], {}, []
    prog = st.progress(0.0, "데이터 불러오는 중…")
    for n, m in enumerate(markets):
        prog.progress(n / max(len(markets), 1), f"{m} 불러오는 중…")
        try:
            data, names = load(m, us_which, kr_n, bn_n, up_n)
        except Exception as e:
            errors.append(f"{m}: {e}")
            continue
        if not data:
            errors.append(f"{m}: 데이터를 받지 못했습니다")
            continue
        p = params_for(m)
        for sym, df in data.items():
            r = sc.scan_one(df, p) or (sc.scan_near(df, p) if show_near else None)
            if r:
                r.update(market=m, symbol=sym, name=names.get(sym, sym))
                rows.append(r)
                hist[(m, sym)] = df
    prog.empty()
    st.session_state["res"] = (rows, hist, errors, {m: len(load(m, us_which, kr_n, bn_n, up_n)[0])
                                                    for m in markets if not any(e.startswith(m) for e in errors)})

if "res" not in st.session_state:
    st.info("왼쪽에서 시장과 조건을 고른 뒤 **스캔 실행**을 누르세요. (모바일은 좌상단 › 버튼)")
    st.stop()

rows, hist, errors, counts = st.session_state["res"]
for e in errors:
    st.warning(e)
st.caption("스캔 종목 수 — " + " · ".join(f"{m} {c}" for m, c in counts.items()))

if not rows:
    st.warning("조건에 맞는 종목이 없습니다. 박스 폭을 넓히거나 거래량 배수·최근 일수를 완화해 보세요.")
    st.stop()

res = pd.DataFrame(rows)


def table(df: pd.DataFrame, near: bool):
    v = pd.DataFrame({
        "시장": df["market"], "종목": df["name"], "코드": df["symbol"],
        "신호일": pd.to_datetime(df["date"]).dt.strftime("%m-%d"),
        "거래량배수": df["vol_ratio"].round(2),
        "박스폭%": (df["width"] * 100).round(1),
        "저항선": df["resistance"], "현재가": df["last"],
        "저항선대비%": (df["ext"] * 100).round(1),
    })
    if not near:
        v["진입가"] = df["entry"]
        v["손절가"] = df["stop"]
        v["진입후%"] = (df["chg"] * 100).round(1)
        v["상태"] = df["status"]
    st.dataframe(v, hide_index=True, use_container_width=True)


brk = res[res["signal"] == "돌파"].sort_values(["bars_ago", "vol_ratio"], ascending=[True, False])
near = res[res["signal"] == "임박"].sort_values("ext", ascending=False)

c1, c2, c3 = st.columns(3)
c1.metric("돌파 신호", len(brk))
c2.metric("그중 유효 (손절 미이탈)", int((brk["status"] == "유효").sum()) if len(brk) else 0)
c3.metric("돌파 임박", len(near))

st.subheader("🚀 거래량 동반 돌파")
if len(brk):
    table(brk, False)
else:
    st.write("해당 없음")
if show_near:
    st.subheader("⏳ 돌파 임박 (저항선 바로 아래 횡보 중)")
    if len(near):
        table(near, True)
    else:
        st.write("해당 없음")

# ── 차트 ──
st.subheader("차트")
allr = pd.concat([brk, near])
labels = [f"[{r.market}] {r.name} ({r.symbol}) · {r.signal}" for r in allr.itertuples()]
pick = st.selectbox("종목 선택", range(len(labels)), format_func=lambda i: labels[i])
r = allr.iloc[pick]
df = hist[(r["market"], r["symbol"])].iloc[-180:]

fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.72, 0.28], vertical_spacing=0.03)
fig.add_trace(go.Candlestick(x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
                             increasing_line_color="#e03131", decreasing_line_color="#1971c2",
                             name="가격"), row=1, col=1)
fig.add_shape(type="rect", x0=r["base_start"], x1=r["date"], y0=r["support"], y1=r["resistance"],
              fillcolor="rgba(255,193,7,0.12)", line=dict(color="rgba(255,193,7,0.6)", width=1), row=1, col=1)
fig.add_hline(y=r["resistance"], line=dict(color="#f08c00", dash="dash"), annotation_text="저항선", row=1, col=1)
if r["signal"] == "돌파":
    fig.add_hline(y=r["stop"], line=dict(color="#868e96", dash="dot"), annotation_text="손절", row=1, col=1)
    fig.add_trace(go.Scatter(x=[r["date"]], y=[r["entry"]], mode="markers", name="돌파",
                             marker=dict(symbol="triangle-up", size=14, color="#2f9e44")), row=1, col=1)
p = params_for(r["market"])
avg = df["Volume"].rolling(p.vol_len).mean().shift(1)
colors = ["#2f9e44" if d == r["date"] and r["signal"] == "돌파" else "#adb5bd" for d in df.index]
fig.add_trace(go.Bar(x=df.index, y=df["Volume"], marker_color=colors, name="거래량"), row=2, col=1)
fig.add_trace(go.Scatter(x=df.index, y=avg * p.vol_mult, line=dict(color="#f08c00", width=1),
                         name=f"평균×{p.vol_mult:g}"), row=2, col=1)
fig.update_layout(height=560, margin=dict(l=10, r=10, t=10, b=10), showlegend=False,
                  xaxis_rangeslider_visible=False)
if r["market"] in ("미국주식", "한국주식"):
    fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])
st.plotly_chart(fig, use_container_width=True)
st.link_button("TradingView에서 보기", sc.tv_link(r["market"], r["symbol"]))

st.caption("일봉 기준 · 코인의 거래량은 거래대금(USDT/KRW) · 데이터는 15분 캐시 · "
           "조건 검색 도구일 뿐 투자 권유가 아닙니다.")
