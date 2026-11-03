import os
import time
import sqlite3
import logging
import requests
import pandas as pd

from datetime import datetime, timezone
from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
)

# ---------------- CONFIG ----------------

TOKEN = os.getenv("BOT_TOKEN")
DB_FILE = os.getenv("DB_PATH", "signals.db")

SCAN_INTERVAL = 900  # 15 minutes
MIN_SCORE = 4

BINANCE = "https://fapi.binance.com"
COINGECKO = "https://api.coingecko.com/api/v3"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)

# ---------------- DATABASE ----------------

def db():
    return sqlite3.connect(DB_FILE)


def init_db():
    con = db()
    cur = con.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            chat_id INTEGER PRIMARY KEY
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS signals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT,
            side TEXT,
            entry REAL,
            sl REAL,
            tp1 REAL,
            tp2 REAL,
            tp3 REAL,
            score INTEGER,
            created TEXT,
            tp1_hit INTEGER DEFAULT 0,
            tp2_hit INTEGER DEFAULT 0,
            status TEXT DEFAULT 'OPEN'
        )
    """)

    con.commit()
    con.close()


# ---------------- MARKET DATA ----------------

def get_top_coins():
    url = f"{COINGECKO}/coins/markets"

    response = requests.get(
        url,
        params={
            "vs_currency": "usd",
            "order": "market_cap_desc",
            "per_page": 30,
            "page": 1
        },
        timeout=20
    )

    response.raise_for_status()

    return [
        coin["symbol"].upper() + "USDT"
        for coin in response.json()
    ]


def get_futures_symbols():
    response = requests.get(
        f"{BINANCE}/fapi/v1/exchangeInfo",
        timeout=20
    )
    response.raise_for_status()

    return {
        item["symbol"]
        for item in response.json()["symbols"]
        if item["contractType"] == "PERPETUAL"
        and item["quoteAsset"] == "USDT"
        and item["status"] == "TRADING"
    }


def get_candles(symbol):
    response = requests.get(
        f"{BINANCE}/fapi/v1/klines",
        params={
            "symbol": symbol,
            "interval": "15m",
            "limit": 150
        },
        timeout=20
    )

    response.raise_for_status()

    data = response.json()

    df = pd.DataFrame(data, columns=[
        "time", "open", "high", "low", "close",
        "volume", "close_time", "quote_volume",
        "trades", "taker_buy", "taker_quote", "ignore"
    ])

    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col])

    return df


# ---------------- INDICATORS ----------------

def analyze(df):
    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]

    ema20 = close.ewm(span=20).mean()
    ema50 = close.ewm(span=50).mean()

    delta = close.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1/14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, 1e-10)
    rsi = 100 - (100 / (1 + rs))

    ema12 = close.ewm(span=12).mean()
    ema26 = close.ewm(span=26).mean()

    macd = ema12 - ema26
    signal_line = macd.ewm(span=9).mean()
    histogram = macd - signal_line

    previous_close = close.shift(1)

    tr = pd.concat([
        high - low,
        (high - previous_close).abs(),
        (low - previous_close).abs()
    ], axis=1).max(axis=1)

    atr = tr.rolling(14).mean()

    volume_avg = volume.rolling(20).mean()

    return {
        "price": float(close.iloc[-1]),
        "ema20": float(ema20.iloc[-1]),
        "ema50": float(ema50.iloc[-1]),
        "rsi": float(rsi.iloc[-1]),
        "macd": float(histogram.iloc[-1]),
        "atr": float(atr.iloc[-1]),
        "volume_ratio": float(
            volume.iloc[-1] / volume_avg.iloc[-1]
        )
    }


# ---------------- SIGNAL GENERATION ----------------

def make_signal(symbol, data):
    price = data["price"]
    atr = data["atr"]

    if not atr or atr <= 0:
        return None

    long_conditions = [
        data["ema20"] > data["ema50"],
        price > data["ema20"],
        52 <= data["rsi"] <= 70,
        data["macd"] > 0,
        data["volume_ratio"] > 1.1
    ]

    short_conditions = [
        data["ema20"] < data["ema50"],
        price < data["ema20"],
        30 <= data["rsi"] <= 48,
        data["macd"] < 0,
        data["volume_ratio"] > 1.1
    ]

    long_score = sum(long_conditions)
    short_score = sum(short_conditions)

    if long_score >= MIN_SCORE and long_score > short_score:
        side = "LONG"
        score = long_score
        sl = price - 1.5 * atr
        risk = price - sl
        tp1 = price + risk
        tp2 = price + 1.5 * risk
        tp3 = price + 2 * risk

    elif short_score >= MIN_SCORE and short_score > long_score:
        side = "SHORT"
        score = short_score
        sl = price + 1.5 * atr
        risk = sl - price
        tp1 = price - risk
        tp2 = price - 1.5 * risk
        tp3 = price - 2 * risk

    else:
        return None

    return {
        "symbol": symbol,
        "side": side,
        "entry": price,
        "sl": sl,
        "tp1": tp1,
        "tp2": tp2,
        "tp3": tp3,
        "score": score,
        "rsi": data["rsi"],
        "volume": data["volume_ratio"]
    }


# ---------------- SIGNAL STORAGE ----------------

def save_signal(s):
    con = db()
    cur = con.cursor()

    cur.execute("""
        SELECT id FROM signals
        WHERE symbol = ?
        AND status = 'OPEN'
        AND created > ?
    """, (
        s["symbol"],
        datetime.now(timezone.utc).strftime("%Y-%m-%d")
    ))

    if cur.fetchone():
        con.close()
        return False

    cur.execute("""
        INSERT INTO signals (
            symbol, side, entry, sl, tp1, tp2, tp3,
            score, created
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        s["symbol"], s["side"], s["entry"], s["sl"],
        s["tp1"], s["tp2"], s["tp3"], s["score"],
        datetime.now(timezone.utc).isoformat()
    ))

    con.commit()
    con.close()
    return True


# ---------------- TELEGRAM ----------------

def format_signal(s):
    return (
        f"🚨 سیگنال جدید فیوچرز\n\n"
        f"ارز: {s['symbol']}\n"
        f"جهت: {s['side']}\n"
        f"ورود: {s['entry']:.6g}\n"
        f"حد ضرر: {s['sl']:.6g}\n\n"
        f"🎯 هدف اول: {s['tp1']:.6g}\n"
        f"🎯 هدف دوم: {s['tp2']:.6g}\n"
        f"🎯 هدف سوم: {s['tp3']:.6g}\n\n"
        f"RSI: {s['rsi']:.1f}\n"
        f"نسبت حجم به میانگین: {s['volume']:.2f}\n"
        f"قدرت شرایط تکنیکال: {s['score']}/5\n\n"
        f"⚠️ امتیاز، احتمال موفقیت نیست.\n"
        f"این پیام توصیه مالی نیست."
    )


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    con = db()
    con.execute(
        "INSERT OR IGNORE INTO users(chat_id) VALUES (?)",
        (update.effective_chat.id,)
    )
    con.commit()
    con.close()

    await update.message.reply_text(
        "سلام! ربات سیگنال فیوچرز فعال است. 📊\n\n"
        "سیگنال‌ها به‌صورت خودکار ارسال می‌شوند.\n"
        "برای گزارش عملکرد بنویس /report"
    )


async def report(update: Update, context: ContextTypes.DEFAULT_TYPE):
    con = db()
    cur = con.cursor()

    cur.execute("SELECT COUNT(*) FROM signals")
    total = cur.fetchone()[0]

    cur.execute("""
        SELECT COUNT(*) FROM signals WHERE status = 'OPEN'
    """)
    opened = cur.fetchone()[0]

    cur.execute("""
        SELECT COUNT(*) FROM signals WHERE status = 'TP3'
    """)
    wins = cur.fetchone()[0]

    cur.execute("""
        SELECT COUNT(*) FROM signals WHERE status = 'SL'
    """)
    losses = cur.fetchone()[0]

    con.close()

    await update.message.reply_text(
        "📊 گزارش عملکرد سیگنال‌ها\n\n"
        f"کل سیگنال‌ها: {total}\n"
        f"باز: {opened}\n"
        f"رسیده به هدف سوم: {wins}\n"
        f"رسیده به حد ضرر: {losses}\n\n"
        "این آمار فقط بر اساس داده‌های ثبت‌شده است."
    )


# ---------------- SCANNER ----------------

async def scan_market(app):
    logging.info("Starting market scan")

    try:
        coins = get_top_coins()
        futures = get_futures_symbols()

        for symbol in coins:
            if symbol not in futures:
                continue

            try:
                df = get_candles(symbol)
                data = analyze(df)
                signal = make_signal(symbol, data)

                if signal and save_signal(signal):
                    con = db()
                    users = con.execute(
                        "SELECT chat_id FROM users"
                    ).fetchall()
                    con.close()

                    for (chat_id,) in users:
                        try:
                            await app.bot.send_message(
                                chat_id=chat_id,
                                text=format_signal(signal)
                            )
                        except Exception:
                            logging.exception("Telegram send failed")

                time.sleep(0.2)

            except Exception:
                logging.exception("Symbol scan failed: %s", symbol)

    except Exception:
        logging.exception("Market scan failed")


async def scanner_loop(app):
    while True:
        await scan_market(app)
        await __import__("asyncio").sleep(SCAN_INTERVAL)


# ---------------- START BOT ----------------

def main():
    if not TOKEN:
        raise RuntimeError("BOT_TOKEN is missing")

    init_db()

    app = Application.builder().token(TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("report", report))

    async def post_init(application):
        application.create_task(scanner_loop(application))

    app.post_init = post_init

    app.run_polling()


if __name__ == "__main__":
    main()
