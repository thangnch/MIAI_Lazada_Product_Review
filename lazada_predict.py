import urllib.request
from bs4 import BeautifulSoup
import re
import csv
import os
import json
import pandas as pd
from sklearn.externals import joblib
from underthesea import word_tokenize
import numpy as np

def load_url(url):
    print("Loading url=", url)
    page = urllib.request.urlopen(url)
    soup = BeautifulSoup(page,"html.parser")
    script = soup.find_all("script", attrs={"type": "application/ld+json"})[0]
    script = str(script)
    script = script.replace("</script>","").replace("<script type=\"application/ld+json\">","")

    csvdata = []

    for element in json.loads(script)["review"]:
        if "reviewBody" in element:
            csvdata.append([element["reviewBody"]])

    return csvdata


def standardize_data(row):
    # remove stopword

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

def analyze(result):
    bad = np.count_nonzero(result)
    good = len(result) - bad
    print("No of bad and neutral comments = ", bad)
    print("No of good comments = ", good)

    if good>bad:
        return "Good! You can buy it!"
    else:
        return "Bad! Please check it carefully!"

# 1. Load URL and print comments
url = input('Nhập url trang:')
if url== "":
    url = "https://www.lazada.vn/products/quan-boi-nam-hot-trend-i244541570-s313421582.html?spm=a2o4n.searchlist.list.11.515c365foL7kyZ&search=1"
data = load_url(url)

# 2. Standardize data
data_frame = pd.DataFrame(data)
data_frame[0] = data_frame[0].apply(standardize_data)

# 3. Tokenizer
data_frame[0] = data_frame[0].apply(tokenizer)

# 4. Embedding
X_val = data_frame[0]
emb = joblib.load('tfidf.pkl')
X_val = emb.transform(X_val)

# 5. Predict
model = joblib.load('saved_model.pkl')
result = model.predict(X_val)
print(analyze(result))
print("Done")




