import os
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

class LRTfidfModel:
    def __init__(self, model_dir='models/lr', analyzer='char_wb'):
        self.model_dir = model_dir
        self.analyzer = analyzer
        self.pipeline = None
        os.makedirs(self.model_dir, exist_ok=True)
        
    def train(self, df_train, df_val):
        """
        Trains the TF-IDF + Logistic Regression model.
        """
        print(f"Training Logistic Regression Model ({self.analyzer})...")
        
        # Configure TfidfVectorizer depending on analyzer type
        if self.analyzer == 'word':
            vectorizer = TfidfVectorizer(analyzer='word', ngram_range=(1, 3), max_features=50000)
        else:
            vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 5), max_features=50000)
            
        pipeline = Pipeline([
            ('tfidf', vectorizer),
            ('lr', LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced'))
        ])
        
        # Grid Search for C
        param_grid = {'lr__C': [0.1, 1, 10]}
        
        X_train = df_train['text']
        y_train = df_train['label']
        
        # We can combine train and val for GridSearch CV, or just use grid search on train
        search = GridSearchCV(pipeline, param_grid, cv=3, scoring='f1', n_jobs=-1, verbose=1)
        search.fit(X_train, y_train)
        
        self.pipeline = search.best_estimator_
        print(f"Best parameters: {search.best_params_}")
        print(f"Best CV F1 score: {search.best_score_:.4f}")
        
        # Validation score
        X_val = df_val['text']
        y_val = df_val['label']
        val_preds = self.pipeline.predict(X_val)
        from sklearn.metrics import f1_score
        val_f1 = f1_score(y_val, val_preds)
        print(f"Validation F1 score: {val_f1:.4f}")
        
        self.save()
        
    def predict(self, texts):
        if not self.pipeline:
            raise ValueError("Model not loaded/trained")
        return self.pipeline.predict(texts)
        
    def predict_proba(self, texts):
        if not self.pipeline:
            raise ValueError("Model not loaded/trained")
        return self.pipeline.predict_proba(texts)
        
    def save(self):
        if self.pipeline:
            path = os.path.join(self.model_dir, 'lr_model.joblib')
            joblib.dump(self.pipeline, path)
            print(f"Model saved to {path}")
            
    def load(self):
        path = os.path.join(self.model_dir, 'lr_model.joblib')
        if os.path.exists(path):
            self.pipeline = joblib.load(path)
            print("Model loaded.")
        else:
            raise FileNotFoundError(f"Model not found at {path}")

if __name__ == "__main__":
    if os.path.exists('data/processed/train.pkl'):
        train_df = pd.read_pickle('data/processed/train.pkl')
        val_df = pd.read_pickle('data/processed/val.pkl')
        
        model = LRTfidfModel()
        model.train(train_df, val_df)
    else:
        print("Please run data_utils.py first to generate data/processed/train.pkl")
