# Ensemble- and Distance-based Feature Ranking and Selection for Unsupervised Learning

This repository contains the code for unsupervised feature ranking. We implement two approaches:

- Ensemble-based feature rankings (computed from ensemlbes of predictive clustering trees),
- Distance-based feature rankings (defined by the unsupervised Relief algorithm).

Both approaches follow the paradigm of predictive clustering, as implemented in Clus (written in java).

## Examples

The code is easy to use! For example, once we have our data stored in a numpy array `x`, we simply call

```
e_ranking_1 = EnsembleRanking()
scores_1 = e_ranking_1.fit(x)
```

or

```
e_ranking_2 = EnsembleRanking(score=["Genie3", "RForest"], ensemble="ExtraTrees", ensemble_size=[2, 3])
scores_2 = e_ranking_2.fit(x)
```

if we want to explicitly set the parameters. Similarly for URelief rankings:

```
relief_1 = URelief()
scores_1 = relief_1.fit(x)

relief_2 = URelief(iterations=[0.25, 0.9, 1.0], neighbours=[10, 15, 2000])
scores_2 = relief_2.fit(x)
```

For more examples, see `src/example.py`


## Requirements

The code requires

- Python 3.6 or higher, together with `numpy` and `scipy`,
- Java 1.8 (because internally, Clus.jar is used).

## License

The code is under the [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/deed.en) licence.

