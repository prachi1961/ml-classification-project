import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

# Load dataset
df = pd.read_csv("news_dataset_20000.csv")

X = df["text"]
y = df["label"]

# Convert text to numbers
vectorizer = TfidfVectorizer()
X_vec = vectorizer.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_vec, y, test_size=0.2, random_state=42
)

# Model
model = MultinomialNB()
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

print("\nNAIVE BAYES RESULTS")
print("Accuracy :", accuracy)
print("Precision:", precision)
print("Recall   :", recall)
print("TP:", tp, "TN:", tn, "FP:", fp, "FN:", fn)
