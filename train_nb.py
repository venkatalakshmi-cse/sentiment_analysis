import pickle

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

from preprocess import load_and_preprocess


# Load dataset
X, y = load_and_preprocess("../data/imdb_reviews.csv")

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Model
model = MultinomialNB()

model.fit(X_train_vec, y_train)

# Prediction
pred = model.predict(X_test_vec)

accuracy = accuracy_score(y_test, pred)

print("Naive Bayes Accuracy:", accuracy)

# Save model
pickle.dump(model, open("../models/nb_model.pkl", "wb"))
pickle.dump(vectorizer, open("../models/tfidf_vectorizer.pkl", "wb"))
