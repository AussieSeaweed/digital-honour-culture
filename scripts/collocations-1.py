from json import dump
from pathlib import Path

from nltk import wordpunct_tokenize, corpus
from nltk.collocations import (
    BigramAssocMeasures,
    TrigramAssocMeasures,
    QuadgramAssocMeasures,
    BigramCollocationFinder,
    TrigramCollocationFinder,
    QuadgramCollocationFinder,
)
from nltk.probability import FreqDist

import pandas as pd


def main():
    df = pd.read_csv(
        (
            Path(__file__).parent.parent
            / 'data'
            / 'jibes-and-delights'
            / 'filtered'
            / 'RoastMe-filtered.csv'
        ),
        sep='\t',
        names=(
            'subreddit',
            'user',
            'hash',
            'id',
            'upvotes',
            'text',
            'keyword0',
            'keyword1',
        ),
    )
    documents = list(map(wordpunct_tokenize, df['text']))
    ignored_words = corpus.stopwords.words('english')
    metrics = {
        'bigram': (BigramAssocMeasures, BigramCollocationFinder, 25),
        'trigram': (TrigramAssocMeasures, TrigramCollocationFinder, 10),
        'quadgram': (QuadgramAssocMeasures, QuadgramCollocationFinder, 3),
    }
    min_freqs = [1, 3, 5, 10, 25, 100, 250]
    count = 100
    phrases = []

    for (
            key,
            (measures_cls, finder_cls, best_min_freq),
    ) in metrics.items():
        measures = measures_cls()

        for min_freq in min_freqs:
            finder = finder_cls.from_documents(documents)

            finder.apply_freq_filter(min_freq)
            finder.apply_word_filter(lambda w: len(w) < 3)
            finder.apply_word_filter(lambda w: w.lower() in ignored_words)

            ngrams = finder.nbest(measures.pmi, count)

            print(f'{key} (>={min_freq}):', ngrams)

            if min_freq == best_min_freq:
                phrases.extend(map(' '.join, ngrams))

    with open(
            (
                Path(__file__).parent.parent
                / 'data'
                / 'collocations'
                / 'phrases.json'
            ),
            'w',
    ) as file:
        dump(phrases, file)


if __name__ == '__main__':
    main()
