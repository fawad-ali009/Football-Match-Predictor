from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import HistGradientBoostingClassifier


# =====================================
# LOGISTIC REGRESSION
# =====================================

def train_logistic_regression(X_train, y_train, class_weight=None):

    model = LogisticRegression(
        max_iter=1000,
        class_weight=class_weight
    )

    model.fit(X_train, y_train)

    return model


# =====================================
# RANDOM FOREST
# =====================================

def train_random_forest(X_train, y_train, class_weight=None):

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=12,
        min_samples_leaf=3,
        random_state=42,
        n_jobs=-1,
        class_weight=class_weight
    )

    model.fit(X_train, y_train)

    return model


# =====================================
# GRADIENT BOOSTING
# =====================================

def train_gradient_boosting(X_train, y_train, class_weight=None):

    model = HistGradientBoostingClassifier(
        max_iter=100,
        learning_rate=0.1,
        random_state=42,
        class_weight=class_weight
    )

    model.fit(X_train, y_train)

    return model