import pandas as pd
from typing import List, Optional, Set

REFERENCE = pd.read_csv("data/cards.csv", index_col='id')


def check_validity(incomplete_deck: List[int], card: int, hero: Optional[str] = None) -> bool:
    global REFERENCE
    card_info = REFERENCE.loc[card, :]
    if not pd.isna(card_info['class']) and hero is not None and card_info['class'] != hero:
        return False
    if card not in incomplete_deck:
        return True
    limit = 1 if card_info['rarity'] == 'legendary' else 2
    return incomplete_deck.count(card) < limit


def jaccard_similarity_score(a: Set, b: Set) -> float:
    return len(a.intersection(b)) / len(a.union(b))
