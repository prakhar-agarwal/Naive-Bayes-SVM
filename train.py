from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import roc_auc_score
from numpy import float32

from nbsvm import NBSVM

import string

def load_data():
    vectorizer = CountVectorizer(binary=True)
    X_train = vectorizer.fit_transform(train.lowercase_parsed_content)
    X_train = X_train.astype(float32)
    y_train = np.array(train.response)
    
    X_test = vectorizer.transform(test.lowercase_parsed_content)
    X_test = X_test.astype(float32)
    y_test = np.array(test.response)
    return X_train, y_train, X_test, y_test
    
print("Loading data...")
X_train, y_train, X_test, y_test = load_data()
mnbsvm = NBSVM()
print("Training model...")
mnbsvm.fit(X_train, y_train)
predicted_NBSVM = mnbsvm.predict(X_test)

print roc_auc_score(test.response, predicted_NBSVM)
