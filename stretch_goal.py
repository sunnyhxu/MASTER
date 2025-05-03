from polygon import RESTClient
import pandas as pd
import matplotlib.pyplot as plt
import time
import matplotlib.dates as mdates

# This code was written for the stretch goal of the project, but due to time constraints posed by how utterly long it takes to train a model,
# we were not able to incorporate it. It is in the repo to show what kind of thing we were planning to do
# Citation: Modified from documentation/tutorial from https://polygon.io/blog/sentiment-analysis-with-ticker-news-api-insights
# Code is based off of tutorial examples. This is because we felt it was important to understand how this particular API was meant to be used
# The code requiress an API key as an arg in RESTClient()

show_plot = True # IMPORTANT
starting_date = "2025-04-01"
ending_date = "2025-04-02" # extend date range but note API key needed
tickers = ["AAPL", "GOOGL"] # list of stock tickers

client = RESTClient()
stock_sentiments = []
master_sentiment_dataframe = None

for ticker in tickers:
    stock_sentiment = []
    for current_day in pd.date_range(start=starting_date, end=ending_date):

        time.sleep(1.1)
        
        current_day_news = list(client.list_ticker_news(ticker, published_utc=current_day.strftime("%Y-%m-%d"), limit=100))
        sentiment_today = {
            'date': current_day.strftime("%Y-%m-%d"),
            'pos': 0,
            'neg': 0,
            'neutral': 0
        }

        for item in current_day_news:
            if hasattr(item, 'insights') and item.insights:
                for insight in item.insights:
                    if insight.sentiment == 'neg':
                        sentiment_today['neg'] += 1
                    elif insight.sentiment == 'pos':
                        sentiment_today['pos'] += 1
                    elif insight.sentiment == 'neutral':
                        sentiment_today['neutral'] += 1
        stock_sentiment.append(sentiment_today)

    df_sentiment = pd.DataFrame(stock_sentiment)
    df_sentiment['date'] = pd.to_datetime(df_sentiment['date'])
    df_sentiment.set_index('date', inplace=True)

    if master_sentiment_dataframe:
        master_sentiment_dataframe = pd.concat([master_sentiment_dataframe, df_sentiment], axis=1)
    else:
        master_sentiment_dataframe = df_sentiment

# plot out the very last one for a sanity check

# plt.figure(figsize=(20, 10))
# plt.plot(df_sentiment['positive'], label='Positive', color='green')
# plt.plot(df_sentiment['negative'], label='Negative', color='red')
# plt.plot(df_sentiment['neutral'], label='Neutral', color='grey', linestyle='--')

# plt.title(f'Sentiment for {tickers[-1]}')
# plt.xlabel('Date')
# plt.ylabel('Cnt')
# plt.grid(True)
# plt.legend()

if show_plot:
    plt.show()