import cudf
import dask.array as da
from cuml.linear_model import LogisticRegression
from cuml.preprocessing.model_selection import train_test_split
from sklearn.datasets import load_iris

import pandas as pd
import optuna


# FYI: Objective functions can take additional arguments
# (https://optuna.readthedocs.io/en/stable/faq.html#objective-func-additional-args).
def objective(trial):
    iris = load_iris()
    X, y = cudf.DataFrame(pd.DataFrame(iris.data.astype('float32'))), cudf.DataFrame(pd.DataFrame(iris.target.astype('float32')))
    solver = trial.suggest_categorical("solver", ["qn"])
    C = trial.suggest_uniform("C", 0.0, 1.0)

    if solver == "qn":
        penalty = trial.suggest_categorical("penalty", ["l1", "l2"])
    else:
        # 'penalty' parameter isn't relevant for this solver,
        # so we always specify 'l2' as the dummy value.
        penalty = "l2"

    classifier = LogisticRegression(max_iter=200, solver=solver, C=C, penalty=penalty)

    X_train, X_valid, y_train, y_valid = train_test_split(X, y)
    classifier.fit(X_train, y_train)

    score = classifier.score(X_valid, y_valid)
    return score


if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=100)

    print("Number of finished trials: ", len(study.trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
