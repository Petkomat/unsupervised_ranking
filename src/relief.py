import numpy as np
import shutil
import os
import re

from typing import List, Union
from helpers import clus_experiment, update_neighbours, min_max_factors, find_temp_dir,\
    create_n_file, read_relief_scores


class URelief:
    """
    Implements unsupervised Relief. Internally, Clus (written in Java)
    is partially used (and it's implementation
    of predictive clustering Relief). Neighbours are computed directly in Python.
    """
    def __init__(self,
                 iterations: Union[float, List[float]] = 1.0,
                 neighbours: Union[int, List[int]] = 20):
        """
        Constructor for this class. Its two arguments are the parameters of the Relief algorithm.

        :param iterations: a proportion of instances that are selected in the algorithm, or a list
                           thereof. If the data contains n_examples, the algorithm selects
                           round(proportion * n_examples) examples.
                           If more than one value is given, the algorithm makes
                           round(max(iterations) * n_examples) iterations, and computes the rankings
                           that correspond to smaller values, when the number of iterations in the
                           algorithm reaches that value.
        :param neighbours: number of neighbours in the algorithm, or a list thereof. Value(s) that
                           are equal or greater than n_examples in the data will be ignored. If
                           no other values exist, ValueError is raised.
                           Similarly as many values of iterations parameter are handled,
                           we compute max(neighbours) neighbours and compute feature rankings for
                           smaller values on the fly.
        """
        self.feature_importance_ = None

        # iterations
        if isinstance(iterations, float):
            iterations = [iterations]
        for i in iterations:
            if not (0.0 < i <= 1.0):
                raise ValueError(
                    f"The proportion of iterations {i} is not between 0.0 (exclusive) "
                    f"and 1.0 (inclusive)."
                )
        self.iterations = iterations
        # neighbours
        if isinstance(neighbours, int):
            neighbours = [neighbours]
        for n in neighbours:
            if not 1 <= n:
                raise ValueError(f"The number of neighbours {n} is not positive!")
        self.neighbors = neighbours

    def fit(self, x: np.ndarray, java=""):
        """
        Computes feature ranking(s). Ignores the neighbour values that are greater or equal to the
        number of examples in the data.

        :param x: A 2D numpy array that represents the data set, x[i, j] is the value of the j-th
                  feature of the i-th example.
        :param java: String with the optional java parameters, such as "-Xmx20G" or
                     "-Xms10G -Xmx20G" that are passed to Clus.
        :return: A dictionary {ranking_name: feature_importance_scores, ...} where every
                 ranking_name is a string, and feature_importance_scores is a list whose
                 j-th element is the importance score of the j-th feature. The number of feature
                 rankings computed equals len(self.iterations) * len(self.neighbours).
        """
        n_examples, n_features = x.shape
        n_neighbors, neighbours = update_neighbours(n_examples, self.neighbors)

        parameters = [
            ("Relief", "Iterations", self.iterations),
            ("Relief", "Neighbours", neighbours),
            ("Relief", "UseIndicesInRankingNames", "Yes"),
            ("Attributes", "Descriptive", f"1-{n_features}"),
            ("Attributes", "Clustering", f"1-{n_features}"),
            ("Attributes", "Target", f"1-{n_features}"),
            ("kNN", "SearchMethod", "Oracle"),
            ("kNN", "K", "[1, 3, 5, 10, 15, 20, 40]")
        ]

        factors = min_max_factors(x)
        x_train = x / factors

        # create Clus ranking experiments
        temp_dir = find_temp_dir()
        os.makedirs(temp_dir)

        neighbors_file = f"{temp_dir}/neighbors.n"
        create_n_file(x_train, x_train, n_neighbors, True, neighbors_file, 'cityblock')
        experiment_name = "experiment"
        clus_experiment(
            x_train, None,
            experiment_name,
            ["relief"],
            parameters + [("kNN", "LoadNeighboursFiles", f"{{{neighbors_file}}}")],
            temp_dir,
            java=java
        )
        # read
        feature_importance_scores = read_relief_scores(f"{temp_dir}/{experiment_name}")
        shutil.rmtree(temp_dir)
        raw = feature_importance_scores
        self.feature_importance_ = {}
        for name, scores in raw.items():
            match = re.match("Iter([0-9]+)Neigh([0-9]+)", name)
            i_iterations = int(match.group(1))
            i_neighbours = int(match.group(2))
            iterations = self.iterations[i_iterations]
            neighbours = self.neighbors[i_neighbours]
            new_name = f"Iter{iterations}Neigh{neighbours}"
            self.feature_importance_[new_name] = scores
        return feature_importance_scores
