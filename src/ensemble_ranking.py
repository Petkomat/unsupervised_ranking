import numpy as np
import shutil
import os

from typing import List, Union
from helpers import clus_experiment, find_temp_dir, read_ensemble_scores


class EnsembleRanking:
    """
    Implements unsupervised Genie3 (aka Gini aka variance reduction) feature ranking
    and unsupervised Random Forest (aka permutation) feature ranking.

    Internally, it uses Clus (written in Java) for inducing unsupervised
    (predictive clustering) trees.
    """

    ENSEMBLES = ["ExtraTrees", "RForest"]
    SCORES = ["Genie3", "RForest"]
    SUBSETS_SIZES = ["LOG", "SQRT"]

    def __init__(self,
                 score: Union[str, List[str]] = "Genie3",
                 ensemble: str = "ExtraTrees",
                 subspace: Union[str, float, int] = "LOG",
                 ensemble_size: Union[int, List[int]] = 100):
        """
        Constructor for the class.

        :param score: An element or a sublist of ["Genie3", "RForest"]
        :param ensemble: An element of ["ExtraTrees", "RForest"]
        :param subspace: An element of ["LOG", "SQRT"], a float such that 0 < subspace <= 1,
                         or an int, such that 1 <= subspace. Depending on the type,
                         if n_features are present in the data, the following number of
                         features is used in tree splits:
                         - str: LOG(n_features) or SQUARE ROOT of n_features,
                         - float: round(subspace * n_features),
                         - int: subspace.
        :param ensemble_size: A positive integer or a list thereof. The number of trees in the
                              ensemble. If list, max(ensemble_size) trees are grown and the first
                              ensemble_size[i] are used for computing the ranking for an ensemble
                              of ensemble_size[i] trees.
        """
        self.feature_importance_ = None

        # score
        if not isinstance(score, list):
            score = [score]
        for s in score:
            if s not in EnsembleRanking.SCORES:
                raise ValueError(
                    f"Wrong score ({s}). Choose an element or sublist of {EnsembleRanking.SCORES}."
                )
        self.score = score
        # ensemble
        if ensemble not in EnsembleRanking.ENSEMBLES:
            raise ValueError(
                f"Wrong ensemble ({ensemble}). Choose an element of {EnsembleRanking.ENSEMBLES}."
            )
        self.ensemble = ensemble
        # subspace
        if isinstance(subspace, str) and subspace not in EnsembleRanking.SUBSETS_SIZES:
            raise ValueError(
                f"Wrong subspace string ({subspace}). "
                f"If string, subspace must be an element of {EnsembleRanking.SUBSETS_SIZES}."
            )
        elif isinstance(subspace, float) and not(0.0 < subspace <= 1.0):
            raise ValueError(
                f"If float, subspace must be between 0 (exclusive) and 1.0 (inclusive), "
                f"but was {subspace}."
            )
        elif isinstance(subspace, int) and not 1 <= subspace:
            raise ValueError(f"If integer, subspace must be positive.")
        self.subspace = subspace
        # ensemble size
        if not isinstance(ensemble_size, list):
            ensemble_size = [ensemble_size]
        for e in ensemble_size:
            if not isinstance(e, int) and 0 < e:
                raise ValueError(f"Every ensemble size must be a positive integer.")
        self.ensemble_size = ensemble_size

    def fit(self, x: np.ndarray, bag_selection="-1", java=""):
        """
        Computes the ranking.
        :param x: A 2D numpy array that represents the data set, x[i, j] is the value of the j-th
                  feature of the i-th example.
        :param bag_selection: "-1" if the whole ensemble (all the trees) is grown. For parallel
                              ensemble induction, set this value to [start, stop] where start and
                              stop are the (1-based) indices of trees. In this case, we grow the
                              trees (and compute feature ranking scores out of them) with indices
                              start <= tree <= stop. For Example, bag_selection = [5, 5] if we want
                              to grow only the fifth tree, and [1, 5] if we want to grow the first
                              five trees.
        :param java: String with the optional java parameters, such as "-Xmx20G" or
                     "-Xms10G -Xmx20G" that are passed to Clus.
        :return: A dictionary {ranking_name: feature_importance_scores, ...} where every
                 ranking_name is a string, and feature_importance_scores is a list whose
                 j-th element is the importance score of the j-th feature. The number of feature
                 rankings computed equals len(self.score) * len(self.ensemble_size).
        """
        n_examples, n_features = x.shape
        parameters = [
            ("Attributes", "Descriptive", f"1-{n_features}"),
            ("Attributes", "Clustering", f"1-{n_features}"),
            ("Attributes", "Target", f"1-{n_features}"),
            ("kNN", "SearchMethod", "Oracle"),
            ("kNN", "K", "[1, 3, 5, 10, 15, 20, 40]"),
            ("Ensemble", "SelectRandomSubspaces", self.subspace),
            ("Ensemble", "EnsembleMethod", self.ensemble),
            ("Ensemble", "Iterations", self.ensemble_size),
            ("Ensemble", "FeatureRanking", f"[{','.join(self.score)}]"),
            ("Ensemble", "EnsembleBootstrapping", "Yes"),
            ("Ensemble", "SymbolicWeight", "Dynamic"),
            ("Ensemble", "Optimize", "Yes"),
            ("Ensemble", "BagSelection", bag_selection),
            ("Model", "MinimalWeight", "1.0"),
            ("Output", "WritePerBagModelFile", "No"),
            ("Output", "TrainErrors", "No")
        ]

        # create clus ranking experiments
        temp_dir = find_temp_dir()
        os.makedirs(temp_dir)

        experiment_name = "experiment"
        clus_experiment(x, None, experiment_name, ["forest"], parameters, temp_dir, java=java)
        # read
        feature_importance_scores = read_ensemble_scores(
            f"{temp_dir}/{experiment_name}", self.score, self.ensemble_size
        )
        shutil.rmtree(temp_dir)
        self.feature_importance_ = feature_importance_scores
        return self.feature_importance_
