import pandas as pd
import re
from nltk import ngrams
from underthesea import word_tokenize
from sklearn import svm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
import os
from sklearn.model_selection import GridSearchCV

emb = None

# Load data from crawler file
def load_data():
    df = pd.read_csv("data_crawler.csv")
    return df

# Standardize text data
def standardize_data(row):
    # Xóa dấu chấm, phẩy, hỏi ở cuối câu
    row = re.sub(r"[\.,\?]+$-", "", row)
    # Xóa tất cả dấu chấm, phẩy, chấm phẩy, chấm thang, ... trong câu
    row = row.replace(",", " ").replace(".", " ") \
        .replace(";", " ").replace("“", " ") \
        .replace(":", " ").replace("”", " ") \
        .replace('"', " ").replace("'", " ") \
        .replace("!", " ").replace("?", " ") \
        .replace("-", " ").replace("?", " ")
    row = row.strip()
    return row

# Tokenizer
def tokenizer(row):
    return word_tokenize(row, format="text")

# Embeding
def embedding(X_train, X_test):
    global  emb
    emb = TfidfVectorizer(min_df=5, max_df=0.8,max_features=3000,sublinear_tf=True)
    emb.fit(X_train)
    X_train =  emb.transform(X_train)
    X_test = emb.transform(X_test)

    # Save pkl file
    joblib.dump(emb, 'tfidf.pkl')
    return X_train, X_test


# 1. Load data from file
data = load_data()

# 2. Standardize and tokenizer Text column
data["Text"] = data.Text.apply(standardize_data)
data["Text"] = data.Text.apply(tokenizer)

# 3. Convert to X_train, y_train
X_train,X_test,y_train,y_test = train_test_split(data["Text"],data["Sentiment"],test_size=0.2, random_state=42)

# 4. Embeding X_train
X_train,X_test  = embedding(X_train, X_test)

# 5. Train and save model
model = svm.SVC(kernel='linear', C = 1)
model.fit(X_train,y_train)
joblib.dump(model, 'saved_model.pkl')

# 6. Test
print("Model score=", model.score(X_test, y_test))
print("Done")


