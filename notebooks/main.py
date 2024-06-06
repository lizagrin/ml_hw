import os
import pickle
import re
import sys
import warnings

import click
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score

warnings.filterwarnings(action='ignore')


def preprocess_text(text):
    text = str(text)
    text = text.lower()
    text = re.sub(r"[^\w\s]+", '', text)
    text = text.replace('"', '')
    return text


def load_data(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File '{file_path}' does not exist")
    return pd.read_csv(file_path)


def preprocess_data(df):
    df['title_and_text'] = df['title'] + ' ' + df['text']
    df['title_and_text'] = df['title_and_text'].apply(preprocess_text)
    return df


def build_model():
    return Pipeline([
        ('vectorizer', CountVectorizer()),
        ('model', LogisticRegression(random_state=42, max_iter=100))
    ])


@click.group()
def main():
    pass


@main.command()
@click.option('--data', required=True)
@click.option('--test', required=False, default=None)
@click.option('--split', type=float, required=False, default=None)
@click.option('--model', required=True)
def train(data, test, split, model):
    df = pd.read_csv(data)
    df = preprocess_data(df)

    x = df['title_and_text']
    y = df['rating']

    if test and split:
        raise ValueError("Test and split cannot provided together")

    elif test:
        test_df = load_data(test)
        test_df = preprocess_data(test_df)
        x_train, y_train = x, y
        x_test, y_test = test_df['title_and_text'], test_df['rating']
    elif split:
        if not 0 <= split <= 1:
            raise ValueError("Split must be between 0 and 1")
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=split)
    else:
        click.echo("Invalid argument")
        sys.exit(1)

    pipeline = build_model()
    pipeline.fit(x_train, y_train)

    with open(model, 'wb') as f:
        pickle.dump(pipeline, f)

    if test or split:
        y_pred = pipeline.predict(x_test)
        f1 = f1_score(y_pred, y_test, average="weighted")
        click.echo(f"f1 result is {f1}")
        click.echo(f"{classification_report(y_test, y_pred)}")


@main.command()
@click.option('--model', type=click.Path(exists=True), help='Path to the trained model.')
@click.option('--data', help='Data for prediction.')
def predict(model, data):
    if not os.path.exists(model):
        raise FileNotFoundError(f"'{model}' does not exist")

    with open(model, 'rb') as model:
        model = pickle.load(model)

    if os.path.exists(data):
        df = load_data(data)
        df = preprocess_data(df)
        x = df['title_and_text']
        predictions = model.predict(x)
        click.echo(f"predicted rating is")
        for p in predictions:
            click.echo(f"{p}")
    else:
        p = model.predict([data])
        click.echo(f"predicted rating is {p[0]}")


if __name__ == '__main__':
    main()
