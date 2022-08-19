from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import GridSearchCV, StratifiedKFold
import pandas as pd 
from typing import List, Dict 

RANDOM_SEED = 42

def train_model(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    n_splits: int = 5,
    parameters: Dict[str, List] = {
        "bootstrap": [True],
        "max_depth": [10, 20, 50, None],
        "max_features": ["auto", "sqrt"],
        "min_samples_leaf": [1, 2],
        "min_samples_split": [2, 5, 10],
        "n_estimators": [100, 200, 600],
    },
    n_estimators: int = 100,
    max_depth: int = None,
    max_features: str = "auto",
    min_samples_leaf: int = 1,
    min_samples_split: int = 2,
    bootstrap: bool = True,
    save_model: bool = True,
    model_path: str = None,
    load_model: bool = False,
    method: str = "train_test"
):
    """
    Train a random forest classifier.
        Args:
            n_splits: Number of folds for stratified k fold
            parameters: Optional selection of parameters to test during
                        grid search cv
            n_estimators: The number of trees in the forest.
            max_depth: The maximum depth of the tree.
            max_features: The number of features to consider when looking for
                            the best split
            min_samples_leaf: The minimum number of samples required to be at a
                                leaf node.
            min_samples_split: The minimum number of samples required to split
                                an internal node
            bootstrap: Whether bootstrap samples are used when building trees.
                        If False, the whole dataset is used to build each tree.
            save_model: save trained model
            model_path: path containing save filename for model
        Returns:
            model: trained model
    """
   
    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        max_features=max_features,
        min_samples_leaf=min_samples_leaf,
        min_samples_split=min_samples_split,
        bootstrap=bootstrap,
        random_state=RANDOM_SEED,
    )

    X_train = train_data.loc[:, train_data.columns!='label']
    y_train = train_data['label']
    X_test = test_data.loc[:, test_data.columns!='label']
    y_test = test_data['label']

    if method == "default":
        clf.fit(X_train, y_train)

    elif method == "grid_search":
        clf = grid_search(
            clf,
            parameters,
            X_train,
            y_train,
            n_splits,
        )
    
    return clf

def train_svm_model(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    n_splits: int = 5,
    parameters: Dict[str, List] = {
        "C": [0.1, 1, 10, 20, 100],
        "kernel": ["rbf", "linear", "poly", "sigmoid"],
    },
    C: float = 1.0,
    kernel: str = "rbf",
    gamma: str = "scale",
    average: str = None,
    save_model: bool = True,
    probability: bool = True,
    model_savepath: str = None,
    method: str = "train_test"
):
    """
    Train a support vector machine of type SVC.
    Args:
        n_splits: Number of folds for stratified k fold
        data_test: Optional dataframe with test/val data and target labels
                    if data has been pre-split.
        parameters: selection of parameters to test during grid search cv
        C: Regularization parameter
        kernel:Specifies the kernel type to be used in the algorithm.
                It must be one of "linear", "poly", "rbf", "sigmoid",
                "precomputed" or a callable.
        gamma: Kernel coefficient for "rbf", "poly" and "sigmoid".
        average: Type of averaging performed when calculating scores such
                    as precision, recall,f1 when the
                    labels are multiclass.
        save_model: save trained model
        model_savepath: path containing save filename for model
    Returns:
        model: trained model
    Note: ensure label/target column is unique and does not appear in the
    vocabulary of the trained model (avoid common words)
    """

    clf = SVC(
        class_weight="balanced",
        random_state=RANDOM_SEED,
        kernel=kernel,
        C=C,
        probability=probability,
    )

    X_train = train_data.loc[:, train_data.columns!='label']
    y_train = train_data['label']
    X_test = test_data.loc[:, test_data.columns!='label']
    y_test = test_data['label']

    if method == "train_test":
        clf.fit(X_train, y_train)

    elif method == "grid_search":
        clf = grid_search(
            clf,
            parameters,
            X_train,
            y_train,
            n_splits,
        )

    return clf

def grid_search(model, parameters, X_train, y_train, n_splits=5):
    """
    Perform a grid search using the model parameters and fit the data
    using the best model outputted by grid search.
    Args:
        model: Classifier model
        parameters: model parameters to use in grid search
        X_train: Training data vectorized
        y_train: Training data labels
        n_splits: number of splits in stratified k fold
    
    Returns:
        Best classifier fit on training data
    """
    skf = StratifiedKFold(n_splits=n_splits)
    clf = GridSearchCV(model, parameters, n_jobs=-1, cv=skf)
    clf.fit(X_train, y_train)

    return clf 