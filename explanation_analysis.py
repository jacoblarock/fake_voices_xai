from ast import literal_eval
import pandas as pd
import numpy as np
from classification import get_labels

def load_explanations(path: str) -> dict[str,dict[str,float]]:
    out = {}
    name = None
    with open(path, "r") as file:
        lines = file.readlines()
    for i in range(len(lines)):
        if i % 2 == 0:
            name = lines[i][:-1]
        elif name != None:
            out[name] = literal_eval(lines[i][:-1])
        else:
            raise RuntimeError("Expected name before result")
    return out

def importance(explanations: dict[str,dict[str,float]]) -> dict[str,float]:
    features = explanations[list(explanations.keys())[0]].keys()
    out = {}
    # add features to out map
    for feature in features:
        out[feature] = []
    for key in explanations:
        for feature in features:
            out[feature].append(np.abs(explanations[key][feature]))
    # summarize values in out map
    for feature in features:
        out[feature] = np.average(out[feature])
    return out

def relevance(explanations: dict[str,dict[str,float]],
              labels: pd.DataFrame
              ) -> dict[str,float]:
    features = explanations[list(explanations.keys())[0]].keys()
    out = {}
    # add features to out map
    for feature in features:
        out[feature] = []
    for key in explanations:
        label = labels.loc[labels["name"]==key,"label"].to_numpy()[0]
        for feature in features:
            out[feature].append(explanations[key][feature] * (2 * label - 1))
    # summarize values in out map
    for feature in features:
        out[feature] = np.average(out[feature])
    return out

if __name__ == "__main__":
    exp = load_explanations("./cache/exp_log.txt")
    imp = importance(exp)
    labels = get_labels("./datasets/release_in_the_wild/meta.csv",
                        "file",
                        "label",
                        "spoof",
                        "bona-fide")
    rel = relevance(exp, labels)
    print(rel)
