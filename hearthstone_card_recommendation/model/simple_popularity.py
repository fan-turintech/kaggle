from . import RecommendModel
import pandas as pd
from typing import List, Dict, Tuple, Optional
from datetime import datetime
from .utility import check_validity


class SimplePopularity(RecommendModel):
    def __init__(self):
        self.popularity: Dict[int, float] = {}

    def _update_popularity(self, card: int, w: float):
        if card not in self.popularity:
            self.popularity[card] = w
        else:
            self.popularity[card] += w

    def fit(self, data: pd.DataFrame):
        """
        columns: deckid,update_date,hero,card0..28, target
        """
        date_begin = datetime.strptime(data['update_date'].min(), '%Y-%m-%d')
        date_end = datetime.strptime(data['update_date'].max(), '%Y-%m-%d')
        max_diff_days = (date_end - date_begin).days
        for i, row in data.iterrows():
            date = datetime.strptime(row['update_date'], '%Y-%m-%d')
            diff_days = (date - date_begin).days
            weight = diff_days / max_diff_days
            deck = row[[f'card{j}' for j in range(29)] + ['target']].to_list()
            for card in list(set(deck)):
                self._update_popularity(card, weight)

    def predict(self, data: pd.DataFrame) -> List[List[int]]:
        ret = []
        sorted_vocabulary = sorted(self.popularity.items(), key=lambda x: x[1], reverse=True)
        for i, row in data.iterrows():
            recommendation = []
            incomplete_deck = row[[f'card{j}' for j in range(29)]]
            hero = row['hero']
            counts = incomplete_deck.value_counts()
            single_cards = sorted([(c, self.popularity[c]) for c in counts[counts == 1].index], key=lambda x: x[1], reverse=True)
            incomplete_deck = incomplete_deck.to_list()
            for c, _ in single_cards:
                if check_validity(incomplete_deck, c, hero):
                    recommendation.append(c)
                    if len(recommendation) == 3:
                        break
            if len(recommendation) < 3:
                for c, _ in sorted_vocabulary:
                    if check_validity(incomplete_deck, c, hero):
                        recommendation.append(c)
                        if len(recommendation) == 3:
                            break
            ret.append(recommendation)

        return ret


if __name__ == '__main__':
    data = pd.read_csv("../data/data.csv")
    data = data.tail(2000)