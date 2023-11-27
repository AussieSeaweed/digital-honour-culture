from abc import ABC, abstractmethod
from enum import Enum, auto
from operator import itemgetter

from transformers import pipeline
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


class Sentiment(Enum):
    NEGATIVE = auto()
    NEUTRAL = auto()
    POSITIVE = auto()


class SentimentAnalyzer(ABC):
    @abstractmethod
    def analyze(self, texts):
        pass


class VaderSentimentAnalyzer(SentimentAnalyzer):
    def __init__(self):
        self.intensity_analyzer = SentimentIntensityAnalyzer()

    def analyze(self, texts):
        return list(map(self.sub_analyze, texts))

    def sub_analyze(self, text):
        scores = self.intensity_analyzer.polarity_scores(text)

        if scores['compound'] >= 0.05:
            sentiment = Sentiment.POSITIVE
        elif scores['compound'] <= -0.05:
            sentiment = Sentiment.NEGATIVE
        else:
            sentiment = Sentiment.NEUTRAL

        return sentiment


class MappedSentimentAnalyzer(ABC):
    mapping = None

    def analyze(self, texts):
        return list(map(self.mapping.get, self._analyze(texts)))

    @abstractmethod
    def _analyze(self, texts):
        pass


class SentimentPipelineAnalyzer(MappedSentimentAnalyzer, ABC):
    task = 'sentiment-analysis'
    model = None

    def __init__(self, device=None):
        self.pipeline = pipeline(self.task, self.model, device=device)

    def _analyze(self, texts):
        return map(itemgetter('label'), self.pipeline(texts))


class BertweetBaseSentimentAnalyzer(SentimentPipelineAnalyzer):
    mapping = {
        'NEG': Sentiment.NEGATIVE,
        'NEU': Sentiment.NEUTRAL,
        'POS': Sentiment.POSITIVE,
    }
    model = 'finiteautomata/bertweet-base-sentiment-analysis'


class DistilbertBaseMultilingualCasedStudentSentimentAnalyzer(
        SentimentPipelineAnalyzer,
):
    mapping = {
        'negative': Sentiment.NEGATIVE,
        'neutral': Sentiment.NEUTRAL,
        'positive': Sentiment.POSITIVE,
    }
    model = 'lxyuan/distilbert-base-multilingual-cased-sentiments-student'


class TwitterRobertaBaseLatestSentimentAnalyzer(SentimentPipelineAnalyzer):
    mapping = {
        'negative': Sentiment.NEGATIVE,
        'neutral': Sentiment.NEUTRAL,
        'positive': Sentiment.POSITIVE,
    }
    model = 'cardiffnlp/twitter-roberta-base-sentiment-latest'


class FineTunedGPTAnalyzer(MappedSentimentAnalyzer):
    mapping = {
        'negative': Sentiment.NEGATIVE,
        'neutral': Sentiment.NEUTRAL,
        'positive': Sentiment.POSITIVE,
    }
    system_prompt = (
        'Classify the following text as negative, neutral, or positive.'
    )

    def __init__(self, client, model):
        self.client = client
        self.model = model

    def _analyze(self, texts):
        return map(self._sub_analyze, texts)

    def _sub_analyze(self, text):
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {'role': 'system', 'content': self.system_prompt},
                {'role': 'user', 'content': text},
            ],
        )

        return completion.choices[0].message.content
