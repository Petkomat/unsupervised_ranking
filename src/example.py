from ensemble_ranking import EnsembleRanking
from relief import URelief

import scipy.io
import numpy as np


def load_data(data_file):
    return scipy.io.loadmat(data_file)["X"].astype(np.float32)


x = load_data("../data/lung.mat")

compute_ensemble = True
compute_relief = True

if compute_ensemble:
    # Ensemble Ranking
    e_ranking_1 = EnsembleRanking()
    e_ranking_2 = EnsembleRanking(
        score=["Genie3", "RForest"], ensemble="ExtraTrees", ensemble_size=[2, 3]
    )
    scores_1 = e_ranking_1.fit(x)  # or e_ranking_1.fit(x); scores = e_ranking.feature_importance_
    scores_2 = e_ranking_2.fit(x)
    print("Showing feature rankings in scores_1:")
    for pair in scores_1.items():
        print(pair)
    print("and scores_2:")
    for pair in scores_2.items():
        print(pair)

if compute_relief:
    # URelief
    relief_1 = URelief()
    relief_1.fit(x)
    # 2000 will be ignored:
    relief_2 = URelief(iterations=[0.25, 0.9, 1.0], neighbours=[10, 15, 2000])
    relief_2.fit(x)
    print("URelief scores:")
    for pair in relief_1.feature_importance_.items():
        print(pair)
    print("")
    for pair in relief_2.feature_importance_.items():
        print(pair)
