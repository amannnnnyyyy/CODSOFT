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

import re
from nltk.tokenize import word_tokenize
from transformers import AutoTokenizer, AutoModel
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import torch

def get_bert_embeddings(text):
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    model = AutoModel.from_pretrained('bert-base-uncased')

    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()


def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    text = text.lower()  # Lowercase
    text = re.sub(r'[^a-z\s]', '', text)  # Remove special characters
    tokens = word_tokenize(text)  # Tokenize
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]  # Lemmatize and remove stopwords
    return ' '.join(tokens)

def predict_genre_from_saved_model(plot, xgb_model, label_encoder):
    cleaned_plot = preprocess_text(plot)
    bert_embedding = get_bert_embeddings(cleaned_plot).reshape(1, -1)
    predicted_label = xgb_model.predict(bert_embedding)
    predicted_genre = label_encoder.inverse_transform(predicted_label)
    return predicted_genre[0]
