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
import snscrape.modules.twitter as sntwitter
import plotly.graph_objects as go

# Function to fetch and prepare data
def fetch_data(ticker, interval, period):
    df = yf.download(tickers=ticker, interval=interval, period=period)
    if df.empty or df.isnull().all().all():
        raise ValueError(f"No data returned for {ticker} with interval {interval} and period {period}")
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

# Function to fetch Google News sentiment
def get_market_sentiment(keywords=["nifty", "banknifty"]):
    st.header("📰 Market Sentiment from Google News")
    feed_url = "https://news.google.com/rss/search?q=stock+market+india&hl=en-IN&gl=IN&ceid=IN:en"
    news_feed = feedparser.parse(feed_url)
    analyzer = SentimentIntensityAnalyzer()

    sentiments, headlines, scores = [], [], []

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
    st.bar_chart(df_sentiment['Sentiment'].value_counts())

    return np.mean(scores)

# Twitter sentiment using snscrape
def twitter_sentiment_analysis(query="Nifty OR BankNifty", limit=50):
    analyzer = SentimentIntensityAnalyzer()
    tweets = []
    for tweet in sntwitter.TwitterSearchScraper(query).get_items():
        if len(tweets) >= limit:
            break
        score = analyzer.polarity_scores(tweet.content)['compound']
        sentiment = 'Positive' if score > 0.05 else 'Negative' if score < -0.05 else 'Neutral'
        tweets.append((tweet.content, sentiment, score))

    if not tweets:
        return 0

    df_twitter = pd.DataFrame(tweets, columns=["Tweet", "Sentiment", "Score"])
    st.subheader("🐦 Twitter Sentiment on Nifty/BankNifty")
    st.table(df_twitter.head(10))
    st.bar_chart(df_twitter['Sentiment'].value_counts())

    return np.mean(df_twitter['Score'])

# Predict and plot with sentiment influence
def predict_and_plot(ticker, interval, period, sentiment_score, time_step=60):
    st.subheader(f"\nAnalyzing {ticker} at {interval} interval")

    df = fetch_data(ticker, interval, period)

    if df.shape[0] <= time_step:
        st.warning(f"Not enough data rows ({df.shape[0]}) for {ticker} with time_step {time_step}. Skipping.")
        return

    df['RSI'] = ta.momentum.RSIIndicator(df['Close']).rsi()
    df['MACD'] = ta.trend.MACD(df['Close']).macd()
    df['SMA_20'] = ta.trend.SMAIndicator(df['Close'], window=20).sma_indicator()
    df['EMA_20'] = ta.trend.EMAIndicator(df['Close'], window=20).ema_indicator()

    close_prices = df[['Close']].dropna()
    if len(close_prices) <= time_step:
        st.warning(f"Not enough data points to make prediction for {ticker}. Needed >{time_step}, got {len(close_prices)}.")
        return

    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(close_prices.values)

    X, y = create_dataset(data_scaled, time_step)
    if X.size == 0 or y.size == 0:
        st.warning(f"Insufficient data after processing for {ticker}. Skipping.")
        return

    X = X.reshape(X.shape[0], X.shape[1], 1)
    model = build_model((X.shape[1], 1))
    model.fit(X, y, epochs=5, batch_size=32, verbose=0)

    last_60_days = data_scaled[-time_step:]
    last_60_days = last_60_days.reshape(1, time_step, 1)
    prediction = model.predict(last_60_days)
    predicted_price = scaler.inverse_transform(prediction)[0][0]

    if sentiment_score > 0.05:
        predicted_price *= 1.01
    elif sentiment_score < -0.05:
        predicted_price *= 0.99

    st.write(f"Predicted Next Price (Adjusted for Sentiment): **{predicted_price:.2f}**")

    fig = go.Figure(data=[go.Candlestick(
        x=df.index,
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        name='Candlestick'
    )])
    fig.add_trace(go.Scatter(x=df.index, y=df['SMA_20'], mode='lines', name='SMA 20'))
    fig.add_trace(go.Scatter(x=df.index, y=df['EMA_20'], mode='lines', name='EMA 20'))
    fig.update_layout(title=f"{ticker} Candlestick with SMA & EMA", xaxis_title='Date', yaxis_title='Price')
    st.plotly_chart(fig, use_container_width=True)

    st.line_chart(df[['RSI']].dropna(), use_container_width=True)
    st.line_chart(df[['MACD']].dropna(), use_container_width=True)

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

news_score = get_market_sentiment()
twitter_score = twitter_sentiment_analysis()
sentiment_score = np.mean([news_score, twitter_score])

user_input = st.text_input("Enter comma-separated tickers (e.g. ^NSEI,^NSEBANK,RELIANCE.NS):", "^NSEBANK")
ticker_list = [ticker.strip() for ticker in user_input.split(",") if ticker.strip()]

intervals_periods = [
    ("1m", "1d"), ("5m", "5d"), ("10m", "5d"), ("15m", "5d"),
    ("30m", "1mo"), ("60m", "1mo"), ("1d", "3mo"), ("1wk", "6mo"),
    ("1mo", "1y"), ("3mo", "2y"), ("6mo", "5y"), ("1y", "10y")
]

interval = st.selectbox("Select Interval:", [ip[0] for ip in intervals_periods])
period = next((ip[1] for ip in intervals_periods if ip[0] == interval), "1mo")

for ticker in ticker_list:
    try:
        predict_and_plot(ticker, interval, period, sentiment_score)
    except Exception as e:
        st.error(f"❌ Error processing {ticker} at interval {interval}: {e}")
