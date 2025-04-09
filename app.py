import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import ta
from io import BytesIO
import feedparser
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Function to fetch and prepare data
def fetch_data(ticker, interval, period):
    df = yf.download(tickers=ticker, interval=interval, period=period)
    df = df.dropna()
    return df

# Function to create dataset for time series prediction
def create_dataset(data, time_step=60):
    X, y = [], []
    for i in range(len(data) - time_step):
        X.append(data[i:(i + time_step), 0])
        y.append(data[i + time_step, 0])
    return np.array(X), np.array(y)

# Function to build LSTM model
def build_model(input_shape):
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Function to fetch news and analyze sentiment
def get_market_sentiment(keywords=["nifty", "banknifty"]):
    st.header("📰 Market Sentiment from Google News")
    feed_url = "https://news.google.com/rss/search?q=stock+market+india&hl=en-IN&gl=IN&ceid=IN:en"
    news_feed = feedparser.parse(feed_url)
    analyzer = SentimentIntensityAnalyzer()

    sentiments = []
    headlines = []
    scores = []

    for entry in news_feed.entries[:25]:
        title = entry.title
        if any(keyword.lower() in title.lower() for keyword in keywords):
            score = analyzer.polarity_scores(title)['compound']
            sentiment = 'Positive' if score > 0.05 else 'Negative' if score < -0.05 else 'Neutral'
            headlines.append(title)
            sentiments.append(sentiment)
            scores.append(score)

    if not headlines:
        st.warning("No relevant news found for Nifty/BankNifty.")
        return 0

    df_sentiment = pd.DataFrame({"Headline": headlines, "Sentiment": sentiments, "Score": scores})
    st.table(df_sentiment)

    sentiment_count = df_sentiment['Sentiment'].value_counts()
    st.bar_chart(sentiment_count)

    avg_score = np.mean(scores)
    if avg_score > 0.05:
        st.success("📈 Overall Sentiment: Bullish")
    elif avg_score < -0.05:
        st.error("📉 Overall Sentiment: Bearish")
    else:
        st.info("➖ Overall Sentiment: Neutral")

    return avg_score

# Predict and plot with sentiment influence
def predict_and_plot(ticker, interval, period, sentiment_score, time_step=60):
    st.subheader(f"\nAnalyzing {ticker} at {interval} interval")
    df = fetch_data(ticker, interval, period)

    # Technical Indicators
    df['RSI'] = ta.momentum.RSIIndicator(df['Close']).rsi()
    df['MACD'] = ta.trend.MACD(df['Close']).macd()
    df['SMA_20'] = ta.trend.SMAIndicator(df['Close'], window=20).sma_indicator()
    df['EMA_20'] = ta.trend.EMAIndicator(df['Close'], window=20).ema_indicator()

    # Prepare data for model
    close_prices = df[['Close']].dropna()
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(close_prices.values)

    X, y = create_dataset(data_scaled, time_step)
    X = X.reshape(X.shape[0], X.shape[1], 1)

    model = build_model((X.shape[1], 1))
    model.fit(X, y, epochs=5, batch_size=32, verbose=0)

    # Predict next point
    last_60_days = data_scaled[-time_step:]
    last_60_days = last_60_days.reshape(1, time_step, 1)
    prediction = model.predict(last_60_days)
    predicted_price = scaler.inverse_transform(prediction)[0][0]

    # Modify prediction using sentiment
    if sentiment_score > 0.05:
        predicted_price *= 1.01
    elif sentiment_score < -0.05:
        predicted_price *= 0.99

    st.write(f"Predicted Next Price (Adjusted for Sentiment): **{predicted_price:.2f}**")

    # Plot closing prices with MA
    fig, ax = plt.subplots(figsize=(10, 4))
    df[['Close', 'SMA_20', 'EMA_20']].plot(ax=ax, title=f"{ticker} Closing Prices with MA ({interval})")
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    st.pyplot(fig)

    # Plot RSI and MACD
    st.line_chart(df[['RSI']].dropna(), use_container_width=True)
    st.line_chart(df[['MACD']].dropna(), use_container_width=True)

    # Download report as CSV
    report = df[['Close', 'RSI', 'MACD', 'SMA_20', 'EMA_20']].copy()
    report['Predicted Next Price'] = predicted_price
    csv = report.to_csv().encode('utf-8')
    st.download_button(
        label="📥 Download Prediction Report (CSV)",
        data=csv,
        file_name=f"{ticker}_{interval}_report.csv",
        mime='text/csv'
    )

# Streamlit app
st.title("📈 AI Stock Predictor: Multi-Ticker with Technicals + Sentiment")

# Get market sentiment score
sentiment_score = get_market_sentiment()

# Ticker input
user_input = st.text_input("Enter comma-separated tickers (e.g. ^NSEI,^NSEBANK,RELIANCE.NS):", "^NSEBANK")
ticker_list = [ticker.strip() for ticker in user_input.split(",") if ticker.strip()]

# Time intervals and periods
intervals_periods = [
    ("1m", "1d"), ("5m", "5d"), ("10m", "5d"), ("15m", "5d"),
    ("30m", "1mo"), ("60m", "1mo"), ("1d", "3mo"), ("1wk", "6mo"),
    ("1mo", "1y"), ("3mo", "2y"), ("6mo", "5y"), ("1y", "10y")
]

# User selects timeframe
interval = st.selectbox("Select Interval:", [ip[0] for ip in intervals_periods])
period = next((ip[1] for ip in intervals_periods if ip[0] == interval), "1mo")

# Predict for each ticker
for ticker in ticker_list:
    try:
        predict_and_plot(ticker, interval, period, sentiment_score)
    except Exception as e:
        st.error(f"Error processing {ticker} at interval {interval}: {e}")
