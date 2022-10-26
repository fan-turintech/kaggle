import pandas as pd
import numpy as np
import json
from typing import List, Dict, Optional, Tuple, Set


if __name__ == '__main__':
    data = pd.read_csv("data/test.csv")
    cabin = data['Cabin'].apply(lambda s: [] if pd.isna(s) else s.split('/'))
    data['Cabin_0'] = cabin.apply(lambda s: s[0] if s else None)
    data['Cabin_1'] = cabin.apply(lambda s: s[1] if s else None)
    data['Cabin_2'] = cabin.apply(lambda s: s[2] if s else None)
    data.drop('Cabin', inplace=True, axis=1)
    data.to_csv("data/test_.csv", index=None)

