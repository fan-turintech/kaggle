import pandas as pd
import numpy as np
import json
from typing import List, Dict, Optional, Tuple, Set

from model import RecommendModel, NaiveGraph, SimplePopularity, EnsembleKnowledge, SimilarityModel


def create_target(data: pd.DataFrame):
    """
    the data should include the header of card0..card29.
    this function create samples by turning one of the cards to the target.
    """
    samples = []
    card_cols = [f'card{i}' for i in range(30)]
    other_cols = [col for col in data.columns if not col.startswith('card')]
    for _, row in data.iterrows():
        deck = row[card_cols].to_list()
        other_fields = row[other_cols].to_list()
        for i in range(_ % 30, 30, 30):
            samples.append(other_fields + deck[:i] + deck[i + 1:] + [deck[i]])

    ret = pd.DataFrame(samples, columns=other_cols + card_cols[:-1] + ['target'])
    return ret


def evaluate(model: RecommendModel, data: pd.DataFrame, test_size: int, final_test: Optional[pd.DataFrame] = None):
    train_data = data.head(data.shape[0] - test_size)
    test_data = data.tail(test_size)
    train_data = create_target(train_data)
    test_data = create_target(test_data)
    print("fitting starts.")
    model.fit(train_data)
    print("fitting finishes. predicting starts.")
    pred = model.predict(test_data)
    if final_test is not None:
        final_pred = model.predict(final_test)
        output = pd.DataFrame(final_test['deckid'], columns=['deckid'])
        output['recommendations'] = [' '.join([str(c) for c in l]) for l in final_pred]
        output.to_csv("submission.csv", index=None)
    if test_data is None or test_data.shape[0] == 0:
        return 0
    score = 0
    for i in range(len(pred)):
        target = test_data.loc[i, 'target']
        p = pred[i]
        if target in p:
            score += 1
    return score / len(pred)


if __name__ == '__main__':
    validation_size = 200
    submission = False
    data = pd.read_csv("data/data_2018.csv")
    if submission:
        test = pd.read_csv("data/test.csv")
    else:
        test = None
    model = EnsembleKnowledge()
    print(evaluate(model, data, validation_size, test))
