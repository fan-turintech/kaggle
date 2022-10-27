from .abstract_model import RecommendModel
import pandas as pd
from typing import List, Dict, Tuple, Optional
from .utility import check_validity
from cornac.models.bpr.recom_bpr import BPR
import cornac
from sklearn.preprocessing import LabelEncoder
import torch


class BPRRecommender(RecommendModel):
    def __init__(self):
        self.model = cornac.models.BiVAECF(
            k=50,
            encoder_structure=[200],
            act_fn="tanh",
            likelihood="pois",
            n_epochs=500,
            batch_size=128,
            learning_rate=0.001,
            seed=42,
            use_gpu=torch.cuda.is_available(),
            verbose=True
        )
        self.train: List[List] = []
        self.all_cards = []
        self.encoder = LabelEncoder()

    def _update_train_set(self, deck_id: str, deck: List[int]):
        for card in deck:
            self.train.append([deck_id, card, 1])

    def fit(self, data: pd.DataFrame):
        for i, row in data.iterrows():
            deck_id = row['deckid']
            deck = row[[f'card{j}' for j in range(29)] + ['target']]
            self._update_train_set(deck_id, deck.to_list())

    def predict(self, data: pd.DataFrame) -> List[List[int]]:
        for i, row in data.iterrows():
            deck_id = row['deckid']
            deck = row[[f'card{j}' for j in range(29)]]
            self._update_train_set(deck_id, deck.to_list())
        train_df = pd.DataFrame(self.train, columns=['userID', 'itemID', 'rating'])
        train_df.drop_duplicates(inplace=True)
        train_df['userID'] = self.encoder.fit_transform(train_df['userID'])
        train_set = cornac.data.Dataset.from_uir(train_df.itertuples(index=False), seed=0)
        self.all_cards = sorted(list(train_set.item_ids))
        self.model.fit(train_set)
        pred = []
        for _, row in data.iterrows():
            pred.append(
                self._predict_one([row[f'card{i}'] for i in range(29)], row['deckid'], row['hero']))
        return pred

    def _predict_one(self, incomplete_deck: List[int], deck_id: str, hero: Optional[str] = None) -> \
    List[int]:
        user_id = self.encoder.transform([deck_id])[0]
        scores = sorted(zip(self.all_cards, list(self.model.score(user_id))), key=lambda x: x[1],
                        reverse=True)
        recommendations = []
        for c, _ in scores:
            if check_validity(incomplete_deck, c, hero):
                recommendations.append(c)
                if len(recommendations) == 3:
                    break
        return recommendations
