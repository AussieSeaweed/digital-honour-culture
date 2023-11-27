from itertools import islice
from json import dumps
from pathlib import Path
from sys import path

from dotenv import load_dotenv
from openai import File, OpenAI
import pandas as pd

path.append(str(Path(__file__).parent.parent))

from sentiment import FineTunedGPTAnalyzer


def main():
    load_dotenv()

    client = OpenAI()
    df = pd.read_csv(
        (
            Path(__file__).parent.parent
            / 'data'
            / 'sentiment-analysis-dataset'
            / 'train.csv'
        ),
        encoding='ISO-8859-1',
    )
    inputs = []
    labels = []
    indices = df['text'].notnull()

    inputs.extend(df['text'][indices])
    labels.extend(df['sentiment'][indices])

    content = '\n'.join(
        dumps(
            {
                'messages': [
                    {
                        'role': 'system',
                        'content': FineTunedGPTAnalyzer.system_prompt,
                    },
                    {'role': 'user', 'content': input_},
                    {'role': 'assistant', 'content': label},
                ],
            },
        ) for input_, label in islice(zip(inputs, labels), 100)
    )

    with open(
            (
                Path(__file__).parent.parent
                / 'data'
                / 'ftjob'
                / 'sentiment.jsonl'
            ),
            'w',
    ) as file:
        file.write(content)

    with open(
            (
                Path(__file__).parent.parent
                / 'data'
                / 'ftjob'
                / 'sentiment.jsonl'
            ),
            'rb',
    ) as file:
        training_file = client.files.create(
            file=file,
            purpose='fine-tune',
        )

    client.fine_tuning.jobs.create(
        training_file=training_file.id,
        model='gpt-3.5-turbo-1106',
    )


if __name__ == '__main__':
    main()
