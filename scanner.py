"""댄 장거식 돌파 스캔 로직 + 시장별 데이터 수집."""
from __future__ import annotations

import io
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np
import pandas as pd
import requests

UA = {"User-Agent": "Mozilla/5.0 (zanger-scanner)"}


# ─────────────────────────── 스캔 로직 ───────────────────────────
@dataclass
class Params:
    base_len: int = 42          # 횡보(박스) 기간 – 봉 수 (주식 42봉 ≈ 2개월)
    max_width: float = 0.25     # 박스 상단/하단 폭 최대치 (25%)
    vol_len: int = 50           # 평균 거래량 계산 기간
    vol_mult: float = 2.0       # 돌파일 거래량 ≥ 평균의 N배
    recent: int = 3             # 최근 N봉 이내 돌파만
    buffer: float = 0.0         # 저항선 위로 몇 % 넘어야 돌파로 볼지
    stop_pct: float = 0.03      # 손절 폭 (-3%)
    res_by_close: bool = False  # 저항선을 종가 최고치로 (꼬리 무시)
    near_pct: float = 0.03      # '돌파 임박' 판정: 저항선 아래 N% 이내


def _base(df: pd.DataFrame, i: int, p: Params):
    """i번째 봉 직전 base_len 봉의 박스 정보."""
    b = df.iloc[i - p.base_len:i]
    res = (b["Close"] if p.res_by_close else b["High"]).max()
    sup = b["Low"].min()
    width = res / sup - 1 if sup > 0 else np.inf
    avgv = df["Volume"].iloc[max(0, i - p.vol_len):i].mean()
    return res, sup, width, avgv


def scan_one(df: pd.DataFrame, p: Params) -> dict | None:
    """돌파 신호(최근 p.recent봉 이내) 찾기. 없으면 None."""
    df = df.dropna(subset=["Close", "High", "Low", "Volume"])
    n = len(df)
    need = max(p.base_len, p.vol_len) + p.recent + 1
    if n < need:
        return None
    last_close = float(df["Close"].iloc[-1])
    for k in range(p.recent):
        i = n - 1 - k
        res, sup, width, avgv = _base(df, i, p)
        if width > p.max_width or not avgv or avgv <= 0:
            continue
        bar = df.iloc[i]
        prev_close = df["Close"].iloc[i - 1]
        vr = bar["Volume"] / avgv
        if bar["Close"] > res * (1 + p.buffer) and prev_close <= res and vr >= p.vol_mult:
            entry = float(bar["Close"])
            stop = entry * (1 - p.stop_pct)
            after = df["Low"].iloc[i + 1:] if i + 1 < n else pd.Series(dtype=float)
            stopped = bool((after <= stop).any())
            return {
                "signal": "돌파",
                "date": df.index[i],
                "bars_ago": k,
                "resistance": float(res),
                "support": float(sup),
                "width": float(width),
                "vol_ratio": float(vr),
                "entry": entry,
                "stop": stop,
                "last": last_close,
                "ext": last_close / res - 1,
                "chg": last_close / entry - 1,
                "status": "손절 이탈" if stopped or last_close <= stop else "유효",
                "base_start": df.index[i - p.base_len],
                "bar_index": i,
            }
    return None


def scan_near(df: pd.DataFrame, p: Params) -> dict | None:
    """아직 돌파 전이지만 저항선 바로 아래에서 박스를 다지는 종목."""
    df = df.dropna(subset=["Close", "High", "Low", "Volume"])
    n = len(df)
    if n < max(p.base_len, p.vol_len) + 2:
        return None
    i = n - 1
    res, sup, width, avgv = _base(df, i, p)
    c = float(df["Close"].iloc[-1])
    if width > p.max_width or c > res or c < res * (1 - p.near_pct):
        return None
    return {
        "signal": "임박",
        "date": df.index[i],
        "bars_ago": 0,
        "resistance": float(res),
        "support": float(sup),
        "width": float(width),
        "vol_ratio": float(df["Volume"].iloc[-1] / avgv) if avgv else np.nan,
        "entry": np.nan,
        "stop": np.nan,
        "last": c,
        "ext": c / res - 1,
        "chg": np.nan,
        "status": "대기",
        "base_start": df.index[i - p.base_len],
        "bar_index": i,
    }


# ─────────────────────────── 유니버스 ───────────────────────────
US_FALLBACK = """AAPL MSFT NVDA AMZN GOOGL META TSLA AVGO AMD NFLX COST ADBE CRM ORCL INTC QCOM TXN MU
AMAT LRCX KLAC MRVL SNPS CDNS PANW CRWD FTNT ZS DDOG SNOW NET PLTR SHOP UBER ABNB DASH COIN HOOD
SOFI MSTR SMCI ARM ANET DELL HPE IBM NOW INTU ADP PYPL V MA JPM BAC WFC GS MS C SCHW BLK AXP UNH
LLY JNJ PFE MRK ABBV TMO ISRG VRTX REGN AMGN GILD XOM CVX COP OXY SLB CAT DE GE GEV HON BA LMT RTX
NOC UPS FDX WMT TGT HD LOW NKE SBUX MCD KO PEP PG DIS CMCSA T VZ TMUS LULU CMG BKNG MAR RCL CCL DAL
UAL VST CEG NRG NEE SO DUK APP RBLX TTD ROKU AFRM RIVN LCID NIO ONON DECK CELH ELF AXON TSM ASML""".split()

KR_FALLBACK = [
    ("005930", "삼성전자", "KS"), ("000660", "SK하이닉스", "KS"), ("373220", "LG에너지솔루션", "KS"),
    ("207940", "삼성바이오로직스", "KS"), ("005380", "현대차", "KS"), ("000270", "기아", "KS"),
    ("068270", "셀트리온", "KS"), ("005490", "POSCO홀딩스", "KS"), ("035420", "NAVER", "KS"),
    ("035720", "카카오", "KS"), ("051910", "LG화학", "KS"), ("006400", "삼성SDI", "KS"),
    ("105560", "KB금융", "KS"), ("055550", "신한지주", "KS"), ("012330", "현대모비스", "KS"),
    ("028260", "삼성물산", "KS"), ("066570", "LG전자", "KS"), ("003550", "LG", "KS"),
    ("096770", "SK이노베이션", "KS"), ("034730", "SK", "KS"), ("017670", "SK텔레콤", "KS"),
    ("030200", "KT", "KS"), ("015760", "한국전력", "KS"), ("032830", "삼성생명", "KS"),
    ("086790", "하나금융지주", "KS"), ("316140", "우리금융지주", "KS"), ("009150", "삼성전기", "KS"),
    ("018260", "삼성에스디에스", "KS"), ("011200", "HMM", "KS"), ("010130", "고려아연", "KS"),
    ("012450", "한화에어로스페이스", "KS"), ("329180", "HD현대중공업", "KS"), ("009540", "HD한국조선해양", "KS"),
    ("042660", "한화오션", "KS"), ("064350", "현대로템", "KS"), ("047810", "한국항공우주", "KS"),
    ("034020", "두산에너빌리티", "KS"), ("010140", "삼성중공업", "KS"), ("267260", "HD현대일렉트릭", "KS"),
    ("003670", "포스코퓨처엠", "KS"), ("259960", "크래프톤", "KS"), ("036570", "엔씨소프트", "KS"),
    ("352820", "하이브", "KS"), ("323410", "카카오뱅크", "KS"), ("000810", "삼성화재", "KS"),
    ("138040", "메리츠금융지주", "KS"), ("402340", "SK스퀘어", "KS"), ("000100", "유한양행", "KS"),
    ("128940", "한미약품", "KS"), ("042700", "한미반도체", "KS"), ("011170", "롯데케미칼", "KS"),
    ("247540", "에코프로비엠", "KQ"), ("086520", "에코프로", "KQ"), ("196170", "알테오젠", "KQ"),
    ("028300", "HLB", "KQ"), ("403870", "HPSP", "KQ"), ("058470", "리노공업", "KQ"),
    ("263750", "펄어비스", "KQ"), ("041510", "에스엠", "KQ"), ("035900", "JYP Ent.", "KQ"),
    ("145020", "휴젤", "KQ"), ("214150", "클래시스", "KQ"), ("039030", "이오테크닉스", "KQ"),
    ("357780", "솔브레인", "KQ"), ("240810", "원익IPS", "KQ"), ("277810", "레인보우로보틱스", "KQ"),
    ("087010", "펩트론", "KQ"), ("141080", "리가켐바이오", "KQ"), ("298380", "에이비엘바이오", "KQ"),
    ("214450", "파마리서치", "KQ"), ("000250", "삼천당제약", "KQ"),
]

STABLES = {"USDC", "FDUSD", "TUSD", "BUSD", "DAI", "USDP", "EUR", "AEUR", "USDE", "PYUSD",
           "USD1", "XUSD", "BFUSD", "EURI", "PAXG", "WBTC", "WBETH", "BNSOL"}


def us_universe(which: str) -> tuple[list[str], dict]:
    """S&P500 / 나스닥100 티커. 실패 시 내장 목록."""
    urls = {
        "S&P 500": "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
        "나스닥 100": "https://en.wikipedia.org/wiki/Nasdaq-100",
    }
    tickers, names = [], {}
    targets = list(urls) if which == "S&P 500 + 나스닥 100" else [which]
    try:
        for t in targets:
            html = requests.get(urls[t], headers=UA, timeout=15).text
            for tb in pd.read_html(io.StringIO(html)):
                col = next((c for c in tb.columns if str(c) in ("Symbol", "Ticker")), None)
                if col is not None and len(tb) > 80:
                    namecol = next((c for c in tb.columns if str(c) in ("Security", "Company")), None)
                    for _, r in tb.iterrows():
                        s = str(r[col]).strip().replace(".", "-")
                        if s and s not in names:
                            tickers.append(s)
                            names[s] = str(r[namecol]) if namecol is not None else s
                    break
    except Exception:
        pass
    if len(tickers) < 50:
        tickers = US_FALLBACK
        names = {t: t for t in tickers}
    return tickers, names


def kr_universe(top_n: int) -> tuple[list[str], dict]:
    """시가총액 상위 N개 (KOSPI+KOSDAQ). yfinance용 6자리.KS/.KQ."""
    try:
        import FinanceDataReader as fdr
        df = fdr.StockListing("KRX")
        df = df[df["Market"].isin(["KOSPI", "KOSDAQ", "KOSDAQ GLOBAL"])]
        df = df.sort_values("Marcap", ascending=False).head(top_n)
        tickers, names = [], {}
        for _, r in df.iterrows():
            sfx = "KS" if r["Market"] == "KOSPI" else "KQ"
            t = f"{r['Code']}.{sfx}"
            tickers.append(t)
            names[t] = r["Name"]
        if len(tickers) >= 20:
            return tickers, names
    except Exception:
        pass
    tickers = [f"{c}.{s}" for c, _, s in KR_FALLBACK][:top_n]
    names = {f"{c}.{s}": n for c, n, s in KR_FALLBACK}
    return tickers, names


# ─────────────────────────── 시세 데이터 ───────────────────────────
def yf_history(tickers: list[str], days: int = 400) -> dict[str, pd.DataFrame]:
    import yfinance as yf
    out: dict[str, pd.DataFrame] = {}
    for s in range(0, len(tickers), 100):
        chunk = tickers[s:s + 100]
        try:
            data = yf.download(chunk, period=f"{days}d", interval="1d", group_by="ticker",
                               auto_adjust=True, threads=True, progress=False)
        except Exception:
            continue
        for t in chunk:
            try:
                d = data[t] if isinstance(data.columns, pd.MultiIndex) else data
                d = d[["Open", "High", "Low", "Close", "Volume"]].dropna()
                if len(d) > 30:
                    out[t] = d
            except Exception:
                pass
    return out


BINANCE_HOSTS = ["https://data-api.binance.vision", "https://api.binance.com"]


def _binance_get(path: str, params=None):
    last = None
    for h in BINANCE_HOSTS:
        try:
            r = requests.get(h + path, params=params, headers=UA, timeout=15)
            if r.status_code == 200:
                return r.json()
            last = r.status_code
        except Exception as e:
            last = e
    raise RuntimeError(f"Binance 요청 실패: {last}")


def binance_universe(top_n: int) -> tuple[list[str], dict]:
    data = _binance_get("/api/v3/ticker/24hr")
    rows = []
    for d in data:
        s = d["symbol"]
        if not s.endswith("USDT"):
            continue
        base = s[:-4]
        if base in STABLES or base.endswith(("UP", "DOWN", "BULL", "BEAR")):
            continue
        rows.append((s, float(d["quoteVolume"])))
    rows.sort(key=lambda x: -x[1])
    syms = [s for s, _ in rows[:top_n]]
    return syms, {s: s[:-4] for s in syms}


def binance_history(symbols: list[str], limit: int = 300) -> dict[str, pd.DataFrame]:
    def one(sym):
        try:
            k = _binance_get("/api/v3/klines", {"symbol": sym, "interval": "1d", "limit": limit})
            df = pd.DataFrame(k).iloc[:, :8]
            df.columns = ["t", "Open", "High", "Low", "Close", "BaseVol", "ct", "Volume"]  # Volume=USDT 거래대금
            df.index = pd.to_datetime(df["t"], unit="ms")
            return sym, df[["Open", "High", "Low", "Close", "Volume"]].astype(float)
        except Exception:
            return sym, None
    with ThreadPoolExecutor(8) as ex:
        return {s: d for s, d in ex.map(one, symbols) if d is not None and len(d) > 30}


def upbit_universe(top_n: int) -> tuple[list[str], dict]:
    mk = requests.get("https://api.upbit.com/v1/market/all", params={"isDetails": "false"},
                      headers=UA, timeout=15).json()
    krw = [m for m in mk if m["market"].startswith("KRW-")]
    names = {m["market"]: m.get("korean_name", m["market"]) for m in krw}
    codes = [m["market"] for m in krw]
    vol = {}
    for s in range(0, len(codes), 100):
        tk = requests.get("https://api.upbit.com/v1/ticker", params={"markets": ",".join(codes[s:s + 100])},
                          headers=UA, timeout=15).json()
        for t in tk:
            vol[t["market"]] = t.get("acc_trade_price_24h", 0)
    codes.sort(key=lambda c: -vol.get(c, 0))
    codes = codes[:top_n]
    return codes, {c: names[c] for c in codes}


def upbit_history(markets: list[str], count: int = 200) -> dict[str, pd.DataFrame]:
    def one(m):
        for attempt in range(4):
            try:
                r = requests.get("https://api.upbit.com/v1/candles/days",
                                 params={"market": m, "count": count}, headers=UA, timeout=15)
                if r.status_code == 429:
                    time.sleep(0.5 * (attempt + 1))
                    continue
                df = pd.DataFrame(r.json())
                df.index = pd.to_datetime(df["candle_date_time_kst"])
                df = df.rename(columns={"opening_price": "Open", "high_price": "High",
                                        "low_price": "Low", "trade_price": "Close",
                                        "candle_acc_trade_price": "Volume"})  # Volume=원화 거래대금
                return m, df[["Open", "High", "Low", "Close", "Volume"]].astype(float).sort_index()
            except Exception:
                time.sleep(0.3)
        return m, None
    with ThreadPoolExecutor(4) as ex:
        return {s: d for s, d in ex.map(one, markets) if d is not None and len(d) > 30}


def tv_link(market: str, sym: str) -> str:
    if market == "미국주식":
        tv = sym.replace("-", ".")
    elif market == "한국주식":
        tv = "KRX:" + sym.split(".")[0]
    elif market == "바이낸스":
        tv = "BINANCE:" + sym
    else:
        tv = "UPBIT:" + sym.split("-")[1] + "KRW"
    return f"https://www.tradingview.com/chart/?symbol={tv}"
