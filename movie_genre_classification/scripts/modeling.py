from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

def hyperparameter_tuning(X_train, y_train):
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('classifier', OneVsRestClassifier(LogisticRegression()))
    ])

    param_grid = {
        'classifier__estimator__C': [0.01, 0.1, 1, 10, 100],
        'classifier__estimator__max_iter': [100, 200, 300], 
        'tfidf__max_features': [5000, 10000, 15000],
        'tfidf__ngram_range': [(1, 1), (1, 2)],
    }

    grid_search = GridSearchCV(pipeline, param_grid, scoring='f1_macro', cv=5, n_jobs=-1)

    grid_search.fit(X_train, y_train)

    print("Best parameters found: ", grid_search.best_params_)
    print("Best cross-validation score: ", grid_search.best_score_)

    return grid_search.best_estimator_