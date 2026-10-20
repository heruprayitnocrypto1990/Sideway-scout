import os
import time
import logging
import requests
import numpy as np
import pandas as pd

# ==================================================
# CONFIGURATION
# ==================================================
COINGECKO_API_KEY = os.getenv("COINGECKO_API_KEY", "")
DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL", "")

COIN_LIMIT = int(os.getenv("COIN_LIMIT", "500"))
MAX_CANDLE_PAIRS = int(os.getenv("MAX_CANDLE_PAIRS", "30"))
DRY_RUN = os.getenv("DRY_RUN", "true").lower() == "true"

MAX_SPREAD_PCT = 0.05
MIN_QUOTE_VOLUME = 500_000

STOP_LOSS_PCT = 0.02
TAKE_PROFIT_1_PCT = 0.02
TAKE_PROFIT_2_PCT = 0.03

CG_BASE = "https://api.coingecko.com/api/v3"
PIONEX_BASE = "https://api.pionex.com"

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "crypto-scouting-bot/1.0"})

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

# ==================================================
# HTTP
# ==================================================
def get_json(url, params=None, headers=None, retries=3):
    for attempt in range(retries):
        try:
            response = SESSION.get(url, params=params, headers=headers, timeout=20)
            response.raise_for_status()
            data = response.json()

            if isinstance(data, dict) and data.get("code") not in (None, "0", 0, "200", 200):
                raise RuntimeError(f"API error {data.get('code')}: {data.get('message', data.get('msg', 'unknown'))}")
            return data
        except Exception as exc:
            logging.warning("Request gagal (%s/%s): %s", attempt + 1, retries, exc)
            if attempt + 1 == retries:
                raise
            time.sleep(2 ** attempt)

# ==================================================
# COINGECKO
# ==================================================
def fetch_coins():
    coins = []
    per_page = 250
    headers = {}
    if COINGECKO_API_KEY:
        headers["x-cg-demo-api-key"] = COINGECKO_API_KEY

    pages = (COIN_LIMIT + per_page - 1) // per_page
    for page in range(1, pages + 1):
        data = get_json(
            f"{CG_BASE}/coins/markets",
            params={"vs_currency": "usd", "order": "market_cap_desc", "per_page": per_page, "page": page, "sparkline": "false"},
            headers=headers,
        )
        if not isinstance(data, list):
            raise RuntimeError("Format respons CoinGecko tidak sesuai.")
        coins.extend(data)
        time.sleep(1)

    coins = coins[:COIN_LIMIT]
    symbols = {}
    for coin in coins:
        symbol = str(coin.get("symbol", "")).upper()
        if symbol:
            symbols.setdefault(symbol, []).append(coin)

    unique = {symbol: items[0] for symbol, items in symbols.items() if len(items) == 1}
    logging.info("CoinGecko: %s koin; %s simbol unik", len(coins), len(unique))
    return unique

# ==================================================
# PIONEX MARKET DATA
# ==================================================
def extract_rows(data):
    """
    Secara rekursif mencari list pertama di dalam struktur JSON.
    Diperbaiki untuk menangani respons bersarang seperti {"data": {"symbols": [...]}}.
    """
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ("symbols", "tickers", "data", "result", "records", "items"):
            value = data.get(key)
            if isinstance(value, list):
                return value
        for value in data.values():
            result = extract_rows(value)
            if result is not None:
                return result
    return None

def get_pionex_symbols():
    return extract_rows(get_json(f"{PIONEX_BASE}/api/v1/common/symbols", params={"type": "SPOT"}))

def get_pionex_tickers():
    return extract_rows(get_json(f"{PIONEX_BASE}/api/v1/market/tickers", params={"type": "SPOT"}))

def get_pionex_book_tickers():
    return extract_rows(get_json(f"{PIONEX_BASE}/api/v1/market/bookTickers", params={"type": "SPOT"}))

def as_float(value, default=0.0):
    try: return float(value)
    except (TypeError, ValueError): return default

def get_field(row, *names, default=None):
    for name in names:
        if row.get(name) is not None: return row[name]
    return default

def normalize_symbol(row):
    return str(get_field(row, "symbol", "symbolName", default="")).upper()

def build_market_list(coins):
    symbols = get_pionex_symbols()
    tickers = get_pionex_tickers()
    books = get_pionex_book_tickers()

    ticker_map = {normalize_symbol(x): x for x in tickers}
    book_map = {normalize_symbol(x): x for x in books}
    candidates = []

    for market in symbols:
        pair = normalize_symbol(market)
        base = str(get_field(market, "baseCurrency", "baseAsset", default="")).upper()
        quote = str(get_field(market, "quoteCurrency", "quoteAsset", default="")).upper()

        if not pair or quote != "USDT" or not base or base not in coins:
            continue

        enabled = get_field(market, "enable", "enabled", "isTrading", default=True)
        if enabled is False or str(enabled).lower() in ("false", "0", "disabled"):
            continue

        ticker, book = ticker_map.get(pair, {}), book_map.get(pair, {})
        bid = as_float(get_field(book, "bidPrice", "bid", default=0))
        ask = as_float(get_field(book, "askPrice", "ask", default=0))

        if bid <= 0 or ask <= 0 or ask < bid: continue
        mid = (bid + ask) / 2
        spread_pct = (ask - bid) / mid * 100
        if spread_pct > MAX_SPREAD_PCT: continue

        volume = as_float(get_field(ticker, "amount", "quoteVolume", "quoteVol", "turnover", default=0))
        if volume < MIN_QUOTE_VOLUME: continue

        candidates.append({"pair": pair, "base": base, "coin": coins[base], "bid": bid, "ask": ask, "spread_pct": spread_pct, "volume": volume})

    candidates.sort(key=lambda x: x["volume"], reverse=True)
    logging.info("Pasangan lolos filter awal: %s", len(candidates))
    return candidates[:MAX_CANDLE_PAIRS]

# ==================================================
# CANDLES
# ==================================================
def fetch_candles(pair, interval, limit=100):
    data = get_json(f"{PIONEX_BASE}/api/v1/market/klines", params={"symbol": pair, "interval": interval, "limit": limit})
    rows = extract_rows(data)
    if not rows: return pd.DataFrame()

    if isinstance(rows[0], dict):
        df = pd.DataFrame(rows)
        aliases = {"time": ("time", "timestamp", "startTime", "openTime"), "open": ("open", "openPrice"), "high": ("high", "highPrice"), "low": ("low", "lowPrice"), "close": ("close", "closePrice"), "volume": ("volume", "baseVolume", "vol")}
        normalized = {}
        for target, choices in aliases.items():
            for choice in choices:
                if choice in df.columns:
                    normalized[target] = df[choice]
                    break
        df = pd.DataFrame(normalized)
    else:
        df = pd.DataFrame(rows)
        if df.shape[1] < 6: return pd.DataFrame()
        df = df.iloc[:, :6]
        df.columns = ["time", "open", "close", "high", "low", "volume"]

    required = {"time", "open", "high", "low", "close", "volume"}
    if not required.issubset(df.columns): return pd.DataFrame()

    for col in ("time", "open", "high", "low", "close", "volume"):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna().sort_values("time").drop_duplicates("time")

    if not df.empty and df["time"].median() > 10_000_000_000:
        now_ms = int(time.time() * 1000)
        interval_ms = 5 * 60 * 1000 if interval == "5M" else 15 * 60 * 1000
        df = df[df["time"] + interval_ms <= now_ms]
    else:
        now_s = int(time.time())
        interval_s = 300 if interval == "5M" else 900
        df = df[df["time"] + interval_s <= now_s]

    return df.reset_index(drop=True)

# ==================================================
# INDICATORS
# ==================================================
def calculate_indicators(df):
    df = df.copy()
    high, low, close, previous_close = df["high"], df["low"], df["close"], df["close"].shift(1)

    tr = pd.concat([high - low, (high - previous_close).abs(), (low - previous_close).abs()], axis=1).max(axis=1)
    df["atr"] = tr.ewm(alpha=1/14, adjust=False).mean()

    up, down = high.diff(), -low.diff()
    plus_dm = up.where((up > down) & (up > 0), 0.0)
    minus_dm = down.where((down > up) & (down > 0), 0.0)
    plus_di = 100 * (plus_dm.ewm(alpha=1/14, adjust=False).mean() / df["atr"].replace(0, np.nan))
    minus_di = 100 * (minus_dm.ewm(alpha=1/14, adjust=False).mean() / df["atr"].replace(0, np.nan))
    dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan))
    df["adx"] = dx.ewm(alpha=1/14, adjust=False).mean()

    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/14, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    df["rsi"] = 100 - (100 / (1 + rs))

    df["bb_mid"] = close.rolling(20).mean()
    bb_std = close.rolling(20).std(ddof=0)
    df["bb_upper"] = df["bb_mid"] + 2 * bb_std
    df["bb_lower"] = df["bb_mid"] - 2 * bb_std
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_mid"].replace(0, np.nan)

    df["volume_sma"] = df["volume"].rolling(20).mean()
    df["atr_avg"] = df["atr"].rolling(20).mean()
    return df

# ==================================================
# SCOUTING RULES
# ==================================================
def analyze_pair(market):
    pair = market["pair"]
    candles5, candles15 = fetch_candles(pair, "5M", 100), fetch_candles(pair, "15M", 100)
    if len(candles5) < 60 or len(candles15) < 40: return None

    m5, m15 = calculate_indicators(candles5), calculate_indicators(candles15)
    current, previous, trend = m5.iloc[-1], m5.iloc[-2], m15.iloc[-1]

    values = [current["atr"], current["rsi"], current["bb_mid"], current["volume_sma"], current["atr_avg"], trend["adx"]]
    if not np.all(np.isfinite(values)): return None

    if trend["adx"] > 20: return None

    window = candles5.tail(48)
    support, resistance = float(window["low"].min()), float(window["high"].max())
    if support <= 0 or resistance <= support: return None

    range_pct = (resistance - support) / support * 100
    if not 0.8 <= range_pct <= 6.0: return None

    atr_ratio = current["atr"] / current["atr_avg"]
    if not 0.7 <= atr_ratio <= 1.3: return None

    if current["volume_sma"] <= 0: return None
    volume_ratio = current["volume"] / current["volume_sma"]
    if not 0.5 <= volume_ratio <= 1.8: return None

    if len(m5) < 25: return None
    bb_mid_change_pct = abs((m5.iloc[-1]["bb_mid"] - m5.iloc[-5]["bb_mid"]) / m5.iloc[-5]["bb_mid"]) * 100
    if bb_mid_change_pct > 0.25: return None

    if current["bb_width"] > previous["bb_width"] * 1.25: return None

    entry = market["ask"]
    if entry <= 0: return None

    range_position = (entry - support) / (resistance - support)
    rsi_cross_up = previous["rsi"] <= 35 and current["rsi"] > 35
    bullish_candle = current["close"] > current["open"]

    if range_position > 0.25 or not rsi_cross_up or not bullish_candle: return None

    stop_loss = entry * (1 - STOP_LOSS_PCT)
    take_profit_1 = entry * (1 + TAKE_PROFIT_1_PCT)
    take_profit_2 = entry * (1 + TAKE_PROFIT_2_PCT)

    if resistance < take_profit_1: return None

    return {
        "pair": pair, "name": market["coin"].get("name", market["base"]), "volume": market["volume"], "spread": market["spread_pct"],
        "adx": float(trend["adx"]), "rsi": float(current["rsi"]), "atr_ratio": float(atr_ratio), "range_pct": float(range_pct),
        "support": support, "resistance": resistance, "entry": entry, "sl": stop_loss, "tp1": take_profit_1, "tp2": take_profit_2,
        "rr1": TAKE_PROFIT_1_PCT / STOP_LOSS_PCT, "rr2": TAKE_PROFIT_2_PCT / STOP_LOSS_PCT, "candle_time": int(current["time"]),
    }

# ==================================================
# DISCORD ALERT
# ==================================================
def format_price(value):
    if value >= 100: return f"{value:.2f}"
    if value >= 1: return f"{value:.5f}"
    return f"{value:.8f}".rstrip("0").rstrip(".")

def send_discord(message):
    if DRY_RUN or not DISCORD_WEBHOOK_URL:
        logging.info("DRY RUN / Discord belum dikonfigurasi:\n%s", message)
        return

    response = SESSION.post(DISCORD_WEBHOOK_URL, json={"content": message[:1900]}, timeout=20)
    if response.status_code == 429:
        try: delay = float(response.json().get("retry_after", 2))
        except (ValueError, TypeError): delay = 2
        time.sleep(min(delay, 10))
        response = SESSION.post(DISCORD_WEBHOOK_URL, json={"content": message[:1900]}, timeout=20)
    response.raise_for_status()

def make_alert(signal):
    return (
        f"**CRYPTO SCOUT — LONG**\nPair: `{signal['pair']}` ({signal['name']})\nVolume proxy: `{signal['volume']:,.0f}`\n"
        f"Spread: `{signal['spread']:.4f}%`\nADX 15m: `{signal['adx']:.2f}` | RSI 5m: `{signal['rsi']:.2f}`\n"
        f"ATR ratio: `{signal['atr_ratio']:.2f}` | Range: `{signal['range_pct']:.2f}%`\n\n"
        f"Entry (ask): `{format_price(signal['entry'])}`\nStop Loss (-2%): `{format_price(signal['sl'])}`\n"
        f"Take Profit 1 (+2%): `{format_price(signal['tp1'])}`\nTake Profit 2 (+3%): `{format_price(signal['tp2'])}`\n"
        f"RR TP1: `1:1` | RR TP2: `1:1.5`\nSupport: `{format_price(signal['support'])}`\nResistance: `{format_price(signal['resistance'])}`\n"
        f"Closed candle: `{signal['candle_time']}`\n_Sinyal teknikal; bukan jaminan profit. Belum memperhitungkan fee dan slippage._"
    )

# ==================================================
# MAIN
# ==================================================
def main():
    logging.info("Mulai crypto scouting")
    coins = fetch_coins()
    markets = build_market_list(coins)

    signals = []
    for market in markets:
        try:
            signal = analyze_pair(market)
            if signal:
                signals.append(signal)
                logging.info("Sinyal ditemukan: %s", signal["pair"])
        except Exception as exc:
            logging.warning("%s gagal dianalisis: %s", market["pair"], exc)
        time.sleep(1)

    signals.sort(key=lambda x: (x["rr2"], x["volume"], -x["spread"]), reverse=True)

    if not signals:
        send_discord("**Crypto Scout:** Tidak ada sinyal yang lolos filter.")
    else:
        for signal in signals[:10]:
            send_discord(make_alert(signal))
            time.sleep(1)

    logging.info("Selesai. Pasangan dianalisis: %s | Sinyal: %s", len(markets), len(signals))

if __name__ == "__main__":
    main()
