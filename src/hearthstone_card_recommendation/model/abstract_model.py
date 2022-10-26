from abc import ABC, abstractmethod
from typing import List
import pandas as pd


class RecommendModel(ABC):
    @abstractmethod
    def fit(self, data: pd.DataFrame):
        """
        columns: deckid,update_date,hero,card0..28, target
        """

    @abstractmethod
    def predict(self, data: pd.DataFrame) -> List[List[int]]:
        """
        columns: deckid,update_date,hero,card0..28
        """
