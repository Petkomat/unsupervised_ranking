import random
import numpy as np
import scipy.io
from scipy.spatial.distance import cdist
import re
import os
import subprocess


def create_settings(s_file, parameters):
    """

    Parameters
    ----------
    s_file str
        path to the s file
    parameters list
        list of triplets (section, setting, value)
    Returns
    -------
        None
    """
    parameters.sort()  # make sections appear together
    stupid_section = "no section"
    last_section = stupid_section
    with open(s_file, "w", newline="") as f:
        for section, setting, value in parameters:
            if section != last_section:
                if last_section != stupid_section:
                    print("", file=f)
                print(f"[{section}]", file=f)
                last_section = section
            print(f"{setting} = {value}", file=f)


def find_temp_dir():
    temp_dir_pattern = "temp{}"
    i = 1
    while os.path.exists(temp_dir_pattern.format(i)):
        i += 1
    return temp_dir_pattern.format(i)


def create_partition(n, k, seed=123):
    random.seed(seed)
    q = n // k
    r = n % k
    sizes = [0] + [q + int(i < r) for i in range(k)]
    starts = np.cumsum(sizes)
    indices = list(range(n))
    random.shuffle(indices)
    partition = [indices[starts[i]: starts[i + 1]] for i in range(k)]
    return partition


class Fimp:
    def __init__(self, f_name=None, num_feat=float("inf"), attr_dict=None, header=None):
        self.f_name = f_name
        self.header = []  # list of lines from the start to the --------- line
        self.table = []  # [[dataset index, name, ranks, relevances], ...]
        self.attrs = {}  # {name: [dataset index, ranks, relevances], ...}
        self.ranking_names = []
        if f_name is None:
            assert attr_dict is not None
            self.attrs = attr_dict
            self.header = header
            for attr in attr_dict:
                row = [attr_dict[attr][0], attr, attr_dict[attr][1], attr_dict[attr][2]]
                self.table.append(row)
                # self.sort_by_relevance()
        else:
            Infinity = float("inf")
            NaN = -1.0  # so that it is ignored
            with open(self.f_name) as f:
                for x in f:
                    self.header.append(x.strip())
                    if x.startswith("---------"):
                        break
                # ranking names: header of the table, last list, remove [ and ], split
                try:
                    self.ranking_names = self.header[-2].strip().split('\t')[-1][1:-1].split(", ")
                except:
                    print("Cannot compute ranking names")
                for feat_ind, x in enumerate(f):
                    if feat_ind == num_feat:
                        break
                    ind, name, rnks, rels = x.strip().split("\t")
                    ind = int(ind)
                    rnks = eval(rnks)
                    rels = eval(rels)
                    self.attrs[name] = [ind, rnks, rels]
                    self.table.append([ind, name, rnks, rels])

    def get_ranking_name(self, ranking_index=0):
        if ranking_index >= len(self.ranking_names):
            print("Cannot give you the name")
            return None
        else:
            return self.ranking_names[ranking_index]

    def get_relevances(self, ranking_index=None):
        return [row[-1] if ranking_index is None else row[-1][ranking_index] for row in self.table]

    def get_attr_indices(self):
        return [row[0] for row in self.table]

    def get_nb_rankings(self):
        return 0 if not self.table else len(self.table[0][-1])

    def sort_by_feature_index(self):
        self.table.sort(key=lambda row: row[0])

    def create_extended_feature_weights(self, ranking_index=0, normalise=False):
        return Fimp._create_extended_feature_weights_static(self.get_attr_indices(),
                                                            self.get_relevances(ranking_index),
                                                            self.f_name,
                                                            normalise)

    @staticmethod
    def _create_extended_feature_weights_static(
            attribute_indices, attribute_relevances, ranking_file, normalise
    ):
        n = max(attribute_indices)
        ws = [0] * n
        for attribute_index, w in zip(attribute_indices, attribute_relevances):
            ws[attribute_index - 1] = max(0, w)
        if normalise:
            w_max = max(ws)
            if w_max == 0.0:
                ws = [1.0] * len(ws)
                print("Only default models in", ranking_file)
            elif w_max == float("inf"):
                ws = [0.0 if w < w_max else 1.0 for w in ws]
                print("Maximal weight = inf, for", ranking_file)
            else:
                ws = [w / w_max for w in ws]
        return ws

    @staticmethod
    def create_extended_feature_weights_dict(feature_ranking, normalise=True):
        pairs = sorted(feature_ranking.items())
        ranking_file = "no file"
        return Fimp._create_extended_feature_weights_static(
            [p[0] for p in pairs], [p[1] for p in pairs], ranking_file, normalise
        )


def simple_arff_convert(x_data, arff_name):
    n_features = x_data.shape[1]
    with open(arff_name, "w", newline="") as f:
        print("@relation love-hate", file=f)
        for i in range(n_features):
            print(f"@attribute x{i} numeric", file=f)
        print("@data", file=f)
        for xs in x_data:
            print(','.join([str(x) for x in xs]), file=f)


def load_data(data_file):
    return scipy.io.loadmat(data_file)["X"].astype(np.float32)


def number_of_clusters(data_file):
    try:
        ys = scipy.io.loadmat(data_file)["Y"]
        ys = ys.flatten()
        n = len(set(ys.tolist()))
        return max(2, n)
    except KeyError:
        return 2  # this happens for dlbcl which is binary (Yes, I know ...)


def number_of_features(data_file):
    return load_data(data_file).shape[1]


def test_number_of_clusters():
    for f in os.listdir("../data"):
        if f.endswith(".mat"):
            print(f, number_of_clusters(f"../data/{f}"))


def compute_neighbours(is_train, x_needs_neighbors, x_train, metric, n_neighbors):
    n_examples_train, n_features = x_train.shape
    n_examples_test = x_needs_neighbors.shape[0]
    # no more than 1M distances at once
    max_matrix_size = 10 ** 6
    portion = max(1, min(max_matrix_size // n_examples_train, n_examples_test))
    ranges = [(i, min(i + portion, n_examples_test)) for i in range(0, n_examples_test, portion)]
    neighbour_range = list(range(int(is_train), int(is_train) + n_neighbors))
    for start, end in ranges:
        chosen = list(range(start, end))
        distances = cdist(x_needs_neighbors[chosen], x_train, metric=metric) / n_features
        top_neighbour_indices = np.argsort(distances)[:, neighbour_range]
        for i_i, i in enumerate(chosen):
            top_indices = top_neighbour_indices[i_i]
            line = [f"NN({top};{distances[i_i, top]})" for top in top_indices]
            yield i, top_indices, line, distances[i_i, top_indices]


def create_n_file(
        x_train, x_needs_neighbors, n_neighbors, is_train, neighbors_file, metric='euclidean'
):
    """
    Finds the nearest neighbours of the second argument in the first argument
    Parameters
    ----------
    x_train
    x_needs_neighbors
    n_neighbors
    is_train
    neighbors_file
    metric

    Returns
    -------

    """
    n_examples_train, n_features = x_train.shape
    i_offset = 0 if is_train else n_examples_train
    with open(neighbors_file, "w", newline="") as f:
        print("START_TARGET;-1", file=f)
        for i, _, line, _ in compute_neighbours(
                is_train, x_needs_neighbors, x_train, metric, n_neighbors):
            print(f"START_TUPLE;{i + i_offset}", file=f)
            print("NB_TARGET_VALUES;1", file=f)
            print("&".join(line), file=f)
            print("END_TUPLE", file=f)
        print("END_TARGET", file=f)


def min_max_factors(x):
    factors = np.max(x, axis=0) - np.min(x, axis=0)
    factors[np.abs(factors) == 0] = 1.0  # make them effectively constant  < 10 ** -12
    return factors


def replace_weird_symbols(line):
    fake_result = "-1"
    new_line = []
    for c in line:
        if re.search("[0-9A-Za-z.,:\\[\\]() +-]", c) is not None:
            new_line.append(c)
        elif new_line and new_line[-1] != fake_result:
            new_line.append(fake_result)
    return "".join(new_line)


def parse_regression_out(out_file):
    """

    Parameters
    ----------
    out_file

    Returns
    -------
    Tuple of models, [training examples, testing examples], errors,
    where errors have the following structure:
    errors[train/test][error measure name] = [(per target first model, average first model), ...]

    """
    error_strings = ["Training error", "Testing error"]
    model_info = "Model information:"
    models = []
    errors = [{}, {}]
    n_examples = [-1, -1]
    pattern = "[^[]+(\\[[^]]+\\])((, Avg r\\^2)?\\: (.+))?"
    with open(out_file, encoding="utf-8") as f:
        NaN = None
        for line in f:
            line = line.strip()
            if line.startswith(model_info):
                y = f.readline().strip()
                while y:
                    models.append(y)
                    y = f.readline().strip()
            else:
                i0 = -1
                for i in range(2):
                    if line.startswith(error_strings[i]):
                        i0 = i
                        break
                if i0 >= 0:
                    for _ in range(2):
                        f.readline()
                    y = f.readline().strip()
                    n_examples[i0] = int(y[y.rfind(": ") + 1:])
                    es = errors[i0]
                    y = f.readline().strip()
                    while y:
                        # error name
                        if "(" in y:
                            y = y[:y.find("(")]
                        e_name = y.strip()
                        es[e_name] = []
                        for _ in range(len(models)):
                            e_line = re.search(pattern, replace_weird_symbols(f.readline().strip()))
                            components = eval(e_line.group(1))
                            if e_line.group(4) is not None:
                                average = float(e_line.group(4))
                            else:
                                average = components[0]
                            es[e_name].append((components, average))
                        y = f.readline().strip()
                    if i0 == 1:
                        break
    if not errors[1]:
        _ = 21
    return models, n_examples, errors


def update_neighbours(n_examples, neighbours):
    n_neighbors = max(neighbours)
    if n_neighbors > n_examples - 1:
        print(f"Too many neighbors ({n_neighbors}) > {n_examples} - 1! Will ignore some.")
        neighbours = [n for n in neighbours if n <= n_examples - 1]
        n_neighbors = max(neighbours)
    return n_neighbors, neighbours


def read_feature_ranking(fimp_names, normalize):
    scores = {}
    for fimp_name in fimp_names:
        base_name = os.path.basename(fimp_name)[:-4]  # remove fimp (leave .)
        fimp = Fimp(fimp_name)
        fimp.sort_by_feature_index()
        n_rankings = fimp.get_nb_rankings()
        for i in range(n_rankings):
            ranking_name = fimp.get_ranking_name(i)
            relevance_vector = fimp.create_extended_feature_weights(i, normalise=normalize)
            scores[base_name + ranking_name] = relevance_vector
    return scores


def read_relief_scores(partial_path):
    def nicer_name(name):
        return name[name.find("Iter"):]
    raw = read_feature_ranking([partial_path + ".fimp"], True)
    return {nicer_name(name): ranking for name, ranking in raw.items()}


def read_ensemble_scores(partial_path, rankings, iterations):
    def nicer_name(name):
        return name[name.find("Tree"): name.find(".")]

    raw = read_feature_ranking(
        [f"{partial_path}Trees{i}{ranking}.fimp" for ranking in rankings for i in iterations],
        False
    )
    return {nicer_name(name): ranking for name, ranking in raw.items()}


def data_iterator(use_only_eval=False):
    forbidden_eval = [
        x.lower() for x in
        ["gasdrift", "lung_small", "orlraws10P", "OVA_Breast", "pgp", "arrhythmia", "warpAR10P"]
    ]
    for data_set in os.listdir("../data"):
        if not data_set.endswith(".mat"):
            continue
        data = data_set[:data_set.find(".mat")]
        if (not use_only_eval) or data.lower() not in forbidden_eval:
            yield data  # , f"data/{data_set}"


def write_ranking_result(result):
    with open("results.txt", "w", newline="") as f:
        print("This is a place-holder, so nothing gets lost (except for, maybe, T)", file=f)
        for component in result:
            print(component, file=f)


def read_scores_file(file):
    inf = float("inf")
    scores = []
    with open(file) as f:
        for line in f:
            scores.append(eval(line))
    return scores


def generate_feature_sizes(features):
    sizes = []
    n = 1
    while n < features:
        sizes.append(n)
        n *= 2
    sizes.append(features)
    return sizes


def clus_experiment(x_train, x_test, experiment_name, command_line_parameters, settings_parameters,
                    temp_dir, java=""):
    # convert data to arff
    a_train = f"{temp_dir}/data_train.arff"
    simple_arff_convert(x_train, a_train)
    if x_test is not None:
        a_test = f"{temp_dir}/data_test.arff"
        simple_arff_convert(x_test, a_test)
    else:
        a_test = None
    # create settings file
    extended_parameters = settings_parameters[:]
    extended_parameters += [
        ("Data", "File", a_train)
    ]
    if a_test is not None:
        extended_parameters.append(("Data", "TestSet", a_test))
    s_file = f"{temp_dir}/{experiment_name}.s"
    create_settings(s_file, extended_parameters)
    # create command line
    switches = ' '.join([f"-{c}" for c in command_line_parameters])
    if java:
        java += " "
    command = f"java {java}-jar clus.jar {switches} {s_file}"
    print(command)
    if "relief" in command_line_parameters:
        exit_code = subprocess.call(command,
                                    shell=True)
    else:
        exit_code = subprocess.call(command, shell=True)
    if exit_code != 0:
        print(f"Clus exited with code {exit_code}")
    return exit_code
