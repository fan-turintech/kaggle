import pandas as pd
import json
from typing import List, Dict


def find_clusters(links: pd.Series):
    adj_graph: Dict[int, List[int]] = {}
    all_nodes = set()
    for key, value in links.items():
        comma = str(key).find(',')
        c1 = int(key[1:comma])
        c2 = int(key[comma + 1:-1])
        all_nodes.add(c1)
        all_nodes.add(c2)
        if c1 in adj_graph:
            adj_graph[c1].append(c2)
        else:
            adj_graph[c1] = [c2]
        if c2 in adj_graph:
            adj_graph[c2].append(c1)
        else:
            adj_graph[c2] = [c1]
    # find spanning trees
    all_trees = []
    while all_nodes:
        tree = set()
        q = {list(all_nodes)[0]}
        r = set()
        tree = tree.union(q)
        while q:
            node = list(q)[0]
            linked_nodes = set(adj_graph[node])
            linked_nodes = linked_nodes.difference(r)
            tree = tree.union(linked_nodes)
            q = q.union(linked_nodes)
            q.remove(node)
            r.add(node)
        all_nodes = all_nodes.difference(tree)
        all_trees.append(tree)
        print(tree)
    return all_trees


if __name__ == '__main__':
    with open("src/hearthstone_card_recommendation/data/naive_graph.json") as fp:
        naive_graph = json.load(fp)
    links = pd.Series(naive_graph, name='link')
    threshold = 650
    links = links[links > threshold]
    print(links.shape)
    print(len(find_clusters(links)))
