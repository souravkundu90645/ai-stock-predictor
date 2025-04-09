import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ta.momentum import RSIIndicator
from ta.trend import MACD, SMAIndicator
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import feedparser
import datetime

# Title
st.set_page_config(layout="wide")
st.title("📈 AI Stock Predictor with Sentiment Analysis")

# Sidebar: select tickers
tickers = st.sidebar.multiselect("Choose stock tickers", ["^NSEI", "^NSEBANK", "RELIANCE.NS", "TCS.NS", "INFY.NS"], default=["^NSEI"])

# Sidebar: select time period
timeframe = st.sidebar.selectbox("Timeframe", ["1d", "5d", "1mo", "3mo", "6mo", "1y"])

# Sentiment analysis
def fetch_sentiment():
    feeds = [
        "https://news.google.com/rss/search?q=nifty+OR+banknifty&hl=en-IN&gl=IN&ceid=IN:en"
    ]
    analyzer = SentimentIntensityAnalyzer()
    sentiment_scores = []

    for feed_url in feeds:
        feed = feedparser.parse(feed_url)
        for entry in feed.entries[:20]:
            score = analyzer.polarity_scores(entry.title)["compound"]
            sentiment_scores.append(score)

    return np.mean(sentiment_scores) if sentiment_scores else 0

sentiment_score = fetch_sentiment()
sentiment_label = "🟢 Positive" if sentiment_score > 0.05 else "🔴 Negative" if sentiment_score < -0.05 else "🟡 Neutral"
st.markdown(f"**📰 Market Sentiment Score:** `{sentiment_score:.2f}` → {sentiment_label}")

# Loop over selected tickers
for ticker in tickers:
    st.subheader(f"📊 {ticker} Stock Analysis")
    data = yf.download(ticker, period=timeframe, interval="1d")
    data.dropna(inplace=True)

    if data.empty:
        st.warning(f"No data found for {ticker}.")
        continue

    # Technical Indicators
    data["RSI"] = RSIIndicator(data["Close"]).rsi()
    data["MACD"] = MACD(data["Close"]).macd()
    data["SMA"] = SMAIndicator(data["Close"]).sma_indicator()

    # Visualization
    fig, axs = plt.subplots(4, 1, figsize=(12, 10), sharex=True)

    axs[0].plot(data["Close"], label="Close Price")
    axs[0].set_title("Close Price")
    axs[0].legend()

    axs[1].plot(data["RSI"], color="orange", label="RSI")
    axs[1].axhline(70, color="red", linestyle="--")
    axs[1].axhline(30, color="green", linestyle="--")
    axs[1].set_title("Relative Strength Index (RSI)")
    axs[1].legend()

    axs[2].plot(data["MACD"], color="purple", label="MACD")
    axs[2].set_title("MACD")
    axs[2].legend()

    axs[3].plot(data["SMA"], color="blue", label="Simple Moving Average")
    axs[3].plot(data["Close"], alpha=0.4, label="Close Price")
    axs[3].set_title("SMA vs Close Price")
    axs[3].legend()

    st.pyplot(fig)

    # Download CSV
    csv = data.to_csv(index=True)
    st.download_button("📥 Download Prediction Report (CSV)", csv, f"{ticker}_report.csv", "text/csv")

