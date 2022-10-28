import json
import statistics
import os

from . import RecommendModel
import pandas as pd
from typing import List, Dict, Tuple, Optional, Set
from datetime import datetime
from .utility import check_validity, jaccard_similarity_score


class EnsembleKnowledge(RecommendModel):
    """
    Aggregate the scores from the following metrics:
    Synergy: how much synergy this card has with the deck
    Mana cost: how well does the cost of the card fit in the deck
    Popularity: how popular this card is in the recent months
    Combo cluster: whether there are other cards in a combo cluster in the deck
    """
    def __init__(self):
        # common variables
        self.weights = {'synergy': 10.0, 'mana': 0.1, 'popularity': 0.5, 'combo': 0.0}
        # synergy-related variables
        self.cards_profile: Dict[int, Set] = {}
        # mana-related variables
        self.mana: Dict[int, int] = {}
        # popularity-related variables
        self.popularity: Dict[int, float] = {}
        # combo-cluster-related variables

    def _update_popularity(self, card: int, w: float):
        if card not in self.popularity:
            self.popularity[card] = w
        else:
            self.popularity[card] += w

    def fit(self, data: pd.DataFrame):
        """
        columns: deckid,update_date,hero,card0..28, target
        """
        self._synergy_fit(data)
        self._mana_fit(data)
        self._popularity_fit(data)
        self._combo_cluster_fit(data)

    def _synergy_fit(self, data: pd.DataFrame):
        filename = "data/saved_card_profiles.json"
        if os.path.exists(filename):
            with open(filename, "r") as fp:
                self.cards_profile = json.load(fp)
                self.cards_profile = {int(k): set(v) for k, v in self.cards_profile.items()}
                return
        for i, row in data.iterrows():
            cards = row[[f'card{j}' for j in range(29)] + ['target']]
            deck_id = row['deckid']
            for card in cards:
                if card in self.cards_profile:
                    self.cards_profile[card].add(deck_id)
                else:
                    self.cards_profile[card] = {deck_id}
        test_data = pd.read_csv("data/test.csv")
        for i, row in test_data.iterrows():
            cards = row[[f'card{j}' for j in range(29)]]
            deck_id = row['deckid']
            for card in cards:
                if card in self.cards_profile:
                    self.cards_profile[card].add(deck_id)
                else:
                    self.cards_profile[card] = {deck_id}
        with open(filename, "w") as fp:
            tmp = {k: list(v) for k, v in self.cards_profile.items()}
            json.dump(tmp, fp)

    def _synergy_predict(self, incomplete_deck: List[int], hero: str, date: str) -> Dict[int, float]:
        """
        look for cards that have similar profiles to the ones in the deck.
        A card profile is a set of deck-ids that include this card.
        """
        rankings: Dict[int, float] = {}
        for card, profile in self.cards_profile.items():
            tmp = [jaccard_similarity_score(self.cards_profile[t], profile) for t in
                   list(set(incomplete_deck)) if t != card and t in self.cards_profile]
            rankings[card] = sum(tmp) / len(tmp)
        return rankings

    def _mana_fit(self, data: pd.DataFrame):
        for i, row in pd.read_csv("data/cards_2018.csv", index_col='id').iterrows():
            self.mana[row.name] = row['cost']

    def _mana_predict(self, incomplete_deck: List[int], hero: str, date: str) -> Dict[int, float]:
        """
        check all-even or all-odd deck.
        Otherwise it favors the cards that are within the range of the card costs from the deck.
        """
        deck_costs = [self.mana[c] for c in incomplete_deck if c in self.mana]
        min_cost = min(deck_costs)
        max_cost = max(deck_costs)
        median_cost = statistics.median(deck_costs)
        all_odd = all([c % 2 == 1 for c in deck_costs])
        all_even = all([c % 2 == 0 for c in deck_costs])
        # if all cards are odd-cost but Baku is not in the deck
        if all_odd and 89335 not in incomplete_deck:
            ret = {89335: 10}
            return ret
        # if all cards are even-cost but Genn is not in the deck
        if all_even and 89336 not in incomplete_deck:
            ret = {89336: 10}
            return ret

        def mana2score(mana: int) -> float:
            if all_odd and mana % 2 == 0:
                return -5.0
            if all_even and mana % 2 == 1:
                return -5.0
            if max_cost >= mana >= min_cost:
                return 1.0
            else:
                return 0.0

        return {k: mana2score(v) for k, v in self.mana.items()}

    def _popularity_fit(self, data: pd.DataFrame):
        fname = "data/saved_popularity.json"
        if os.path.exists(fname):
            with open(fname, "r") as fp:
                self.popularity = json.load(fp)
                self.popularity = {int(k): v for k, v in self.popularity.items()}
                return
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
        max_popularity = max(self.popularity.values())
        self.popularity = {k: v / max_popularity for k, v in self.popularity.items()}
        with open(fname, "w") as fp:
            json.dump(self.popularity, fp)

    def _popularity_predict(self, incomplete_deck: List[int], hero: str, date: str) -> Dict[int, float]:
        counts = pd.Series(incomplete_deck).value_counts()
        scores: Dict[int, float] = {}
        if all(counts == 1):
            # no-duplicate deck, recommend Reno and Kazakus
            if 1024971 not in incomplete_deck:
                scores[1024971] = 10
            if 49622 not in incomplete_deck:
                scores[49622] = 10
            if 49744 not in incomplete_deck and hero == 'warlock':
                scores[49744] = 5
            if 49693 not in incomplete_deck and hero == 'mage':
                scores[49693] = 5
            if 49702 not in incomplete_deck and hero == 'priest':
                scores[49702] = 5
            scores.update({k: v for k, v in self.popularity.items() if k not in scores and k not in incomplete_deck})
        else:
            scores = {c: self.popularity[c] + 2 for c in counts[counts == 1].index}
            scores.update({k: v for k, v in self.popularity.items() if k not in scores})
        return scores

    def _combo_cluster_fit(self, data: pd.DataFrame):
        pass

    def _combo_cluster_predict(self, incomplete_deck: List[int], hero: str, date: str) -> Dict[int, float]:
        return {}

    def predict(self, data: pd.DataFrame) -> List[List[int]]:
        """
        columns: deckid,update_date,hero,card0..28
        """
        ret = []
        for i, row in data.iterrows():
            incomplete_deck = row[[f'card{j}' for j in range(29)]].to_list()
            hero = row['hero']
            date = row['update_date']
            synergy_scores = pd.Series(self._synergy_predict(incomplete_deck, hero, date), name='synergy')
            mana_scores = pd.Series(self._mana_predict(incomplete_deck, hero, date), name='mana')
            popularity_scores = pd.Series(self._popularity_predict(incomplete_deck, hero, date), name='popularity')
            # combo_scores = pd.Series(self._combo_cluster_predict(incomplete_deck, hero, date), name='combo')
            scores = synergy_scores.to_frame()\
                .join(mana_scores, how='outer')\
                .join(popularity_scores, how='outer')#\
                #.join(combo_scores, how='outer')
            scores.fillna(0, inplace=True)
            scores['aggregated'] = scores['synergy'] * self.weights['synergy'] \
                                   + scores['mana'] * self.weights['mana'] \
                                   + scores['popularity'] * self.weights['popularity']# \
                                   #+ scores['combo'] * self.weights['combo']
            scores.sort_values(by='aggregated', ascending=False, inplace=True)
            recommendation = []
            for card in scores.index:
                if check_validity(incomplete_deck, card, hero):
                    recommendation.append(card)
                    if len(recommendation) == 3:
                        break
            ret.append(recommendation)
            if i % 10 == 9:
                print(f"Prediction progress {i + 1} / {data.shape[0]}")
        return ret
