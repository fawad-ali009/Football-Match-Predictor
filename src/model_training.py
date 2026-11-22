from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import HistGradientBoostingClassifier

def train_logistic_regression(X_train, y_train):

    model = LogisticRegression(max_iter=1000)

    model.fit(X_train, y_train)

    return model

def train_random_forest(X_train, y_train):

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=12,
        min_samples_leaf=3,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    return model

def train_gradient_boosting(X_train, y_train):

    model = HistGradientBoostingClassifier(
        max_iter=100,
        learning_rate=0.1,
        random_state=42
    )

    model.fit(X_train, y_train)

    return model