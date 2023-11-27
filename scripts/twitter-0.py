from os import getenv
from pathlib import Path
from pickle import dump
from sys import path, setrecursionlimit

from dotenv import load_dotenv
from tweepy import Client
import pandas as pd

path.append(str(Path(__file__).parent.parent))

from twitter import pull_tweets


def main():
    setrecursionlimit(1000000)
    load_dotenv()

    bearer_token = getenv('X_BEARER_TOKEN')
    client = Client(bearer_token, wait_on_rate_limit=True)
    tweets = []

    try:
        for tweet in pull_tweets(
                client,
                '"@" is:verified -is:retweet has:mentions lang:en',
                100,
        ):
            tweets.append(tweet)
    except:
        pass

    try:
        data = {'id': [], 'text': [], 'author_id': []}

        for tweet in tweets:
            for key, value in data.items():
                if key in tweet:
                    value.append(tweet[key])

        df = pd.DataFrame(data=data)

        df.to_csv(
            Path(__file__).parent.parent / 'data' / 'twitter' / 'general-0.csv',
        )
    except:
        pass

    with open(
            (
                Path(__file__).parent.parent
                / 'data'
                / 'twitter'
                / 'general.pickle'
            ),
            'wb',
    ) as file:
        dump(tweets, file)


if __name__ == '__main__':
    main()
