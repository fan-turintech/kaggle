from .abstract_model import RecommendModel
import pandas as pd
from typing import List, Dict, Tuple, Optional
from .utility import check_validity
import json


class NaiveGraph(RecommendModel):
    def __init__(self):
        self.graph: Dict[Tuple[int, int], float] = {}
        self.vocabulary = set()

    def _update_graph(self, c1: int, c2: int, weight: float):
        self.vocabulary.add(c1)
        self.vocabulary.add(c2)
        if c1 < c2:
            key = (c1, c2)
        else:
            key = (c2, c1)
        if key in self.graph:
            self.graph[key] += weight
        else:
            self.graph[key] = weight

    def _get_weight(self, c1: int, c2: int) -> float:
        if c1 < c2:
            key = (c1, c2)
        else:
            key = (c2, c1)
        if key in self.graph:
            return self.graph[key]
        return 0.0

    def fit(self, data: pd.DataFrame):
        for _, row in data.iterrows():
            for i in range(29):
                self._update_graph(row[f'card{i}'], row['target'], 1.0)
        with open("data/naive_graph.json", "w") as fp:
            graph = {str(key): v for key, v in self.graph.items()}
            json.dump(graph, fp)

    def predict(self, data: pd.DataFrame) -> List[List[int]]:
        pred = []
        for _, row in data.iterrows():
            pred.append(self._predict_one([row[f'card{i}'] for i in range(29)], row['hero']))
        return pred

    def _predict_one(self, incomplete_deck: List[int], hero: Optional[str] = None) -> List[int]:
        rankings = {}
        for candidate in list(self.vocabulary):
            rankings[candidate] = 0
            for card in incomplete_deck:
                rankings[candidate] += self._get_weight(card, candidate)
        rankings = sorted([(k, v) for k, v in rankings.items()], key=lambda r: r[1], reverse=True)
        valid_rankings = []
        for r in rankings:
            if check_validity(incomplete_deck, r[0], hero):
                valid_rankings.append(r[0])
                if len(valid_rankings) == 3:
                    break
        return valid_rankings
