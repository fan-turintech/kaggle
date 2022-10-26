from typing import List, Dict, Optional, Set
import pandas as pd
from .abstract_model import RecommendModel
from .utility import check_validity


def jaccard_similarity_score(a: Set, b: Set) -> float:
    return len(a.intersection(b)) / len(a.union(b))


class SimilarityModel(RecommendModel):
    def __init__(self):
        self.decks: Dict[str, Set] = {}
        self.cards_profile: Dict[int, Set] = {}

    def fit(self, data: pd.DataFrame):
        """
        columns: deckid,update_date,hero,card0..28, target
        """
        for i, row in data.iterrows():
            cards = row[[f'card{j}' for j in range(29)] + ['target']]
            deck_id = row['deckid']
            self.decks[deck_id] = set(cards)
            for card in cards:
                if card in self.cards_profile:
                    self.cards_profile[card].add(deck_id)
                else:
                    self.cards_profile[card] = {deck_id}

    def predict(self, data: pd.DataFrame) -> List[List[int]]:
        """
        columns: deckid,update_date,hero,card0..28
        """
        pred = []
        for _, row in data.iterrows():
            pred.append(self._predict_one([row[f'card{i}'] for i in range(29)], row['hero']))
            if _ % 10 == 9:
                print(f'prediction progress {_ + 1} / {data.shape[0]}')
        return pred

    def _predict_one(self, incomplete_deck: List[int], hero: Optional[str] = None) -> List[int]:
        rankings = {}
        for card, profile in self.cards_profile.items():
            tmp = [jaccard_similarity_score(self.cards_profile[t], profile) for t in list(set(incomplete_deck)) if t != card and t in self.cards_profile]
            rankings[card] = sum(tmp) / len(tmp)

        top_cards = sorted([(k, v) for k, v in rankings.items()], key=lambda x: x[1],
                           reverse=True)
        recommendation = []
        for card, _ in top_cards:
            if check_validity(incomplete_deck, card, hero):
                recommendation.append(card)
                if len(recommendation) == 3:
                    break
        return recommendation
