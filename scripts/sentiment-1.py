from os import getenv
from pathlib import Path
from sys import path

from dotenv import load_dotenv
from openai import OpenAI
import pandas as pd

path.append(str(Path(__file__).parent.parent))

from sentiment import (
    BertweetBaseSentimentAnalyzer,
    DistilbertBaseMultilingualCasedStudentSentimentAnalyzer,
    FineTunedGPTAnalyzer,
    Sentiment,
    TwitterRobertaBaseLatestSentimentAnalyzer,
    VaderSentimentAnalyzer,
)


def main():
    load_dotenv()

    client = OpenAI()
    sentiment_analyzers = {
        'vader': VaderSentimentAnalyzer(),
        'bertweet': BertweetBaseSentimentAnalyzer(0),
        'distilbert': (
            DistilbertBaseMultilingualCasedStudentSentimentAnalyzer(0)
        ),
        'roberta': TwitterRobertaBaseLatestSentimentAnalyzer(0),
        'gpt': FineTunedGPTAnalyzer(
            client,
            getenv('FINE_TUNED_GPT_SENTIMENT_ANALYZER_MODEL'),
        ),
    }
    df = pd.read_csv(
        (
            Path(__file__).parent.parent
            / 'data'
            / 'sentiment-analysis-dataset'
            / 'test.csv'
        ),
        encoding='ISO-8859-1',
    )
    inputs = []
    labels = []
    mapping = {
        'negative': Sentiment.NEGATIVE,
        'neutral': Sentiment.NEUTRAL,
        'positive': Sentiment.POSITIVE,
    }
    indices = df['text'].notnull()

    inputs.extend(df['text'][indices])
    labels.extend(map(mapping.get, df['sentiment'][indices]))

    for key, value in sentiment_analyzers.items():
        outputs = value.analyze(inputs)
        correct = 0
        total = 0
        fcorrect = 0
        ftotal = 0

        for output, label in zip(outputs, labels):
            correct += output == label
            total += 1

            if output == Sentiment.NEUTRAL or label == Sentiment.NEUTRAL:
                continue

            fcorrect += output == label
            ftotal += 1

        print(
            f'{key}:',
            f'{correct}/{total} {correct / total * 100:.3f}%',
            f'{fcorrect}/{ftotal} {fcorrect / ftotal * 100:.3f}%',
        )


if __name__ == '__main__':
    main()
