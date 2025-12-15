# ======================================================
# PROJECT: Multiclass News Topic Classification
# DOMAIN : NLP + Machine Learning
# DATASET: 20 Newsgroups (sklearn)
# MODELS : NB, Logistic Regression, SVM, Random Forest
# ======================================================


# ======================================================
# 1. IMPORT REQUIRED LIBRARIES
# ------------------------------------------------------
# - re           : Text cleaning using regular expressions
# - joblib       : Save and load trained ML models
# - numpy/pandas : Data handling and result comparison
# - nltk         : NLP preprocessing (tokenization, lemmatization)
# - sklearn      : Dataset, feature extraction, ML models
# ======================================================
import re
import joblib
import os
import numpy as np
import pandas as pd

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score, precision_recall_fscore_support


# ======================================================
# 2. FETCH DATASET
# ------------------------------------------------------
# - Fetches the 20 Newsgroups dataset from sklearn
# - Removes headers, footers, and quotes to reduce noise
# - X : raw text documents
# - y : numerical class labels
# ======================================================
print("\n" + "="*60)
print("FETCHING 20 NEWSGROUPS DATASET...")
print("="*60)
print("This may take a few minutes if downloading for the first time.")
print("Please wait...\n")

data = fetch_20newsgroups(
    subset='all',
    remove=('headers', 'footers', 'quotes')
)

X = data.data
y = data.target
target_names = data.target_names

print("\n✓ Dataset loaded successfully!")
print("Total Documents:", len(X))
print("Total Classes:", len(target_names))


# ======================================================
# 3. NLP PREPROCESSING FOR DATASET
# ------------------------------------------------------
# This function performs:
# - Lowercasing
# - URL removal
# - HTML tag removal
# - Special character & number removal
# - Tokenization
# - Stopword removal
# - Lemmatization
#
# Purpose:
# Convert raw unstructured text into clean normalized text
# ======================================================
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-z\s]', '', text)

    tokens = word_tokenize(text)

    tokens = [
        lemmatizer.lemmatize(word)
        for word in tokens
        if word not in stop_words and len(word) > 2
    ]

    return " ".join(tokens)


# Apply NLP preprocessing to entire dataset
print("\n" + "="*60)
print("PREPROCESSING DATASET...")
print("="*60)
print(f"Processing {len(X)} documents. This may take several minutes.")
print("Progress: ", end="", flush=True)

X_cleaned = []
for i, doc in enumerate(X):
    X_cleaned.append(preprocess_text(doc))
    # Print progress every 1000 documents
    if (i + 1) % 1000 == 0:
        print(f"{i + 1}...", end="", flush=True)

print(f"\n✓ Preprocessing complete! Processed {len(X_cleaned)} documents.")


# ======================================================
# 4. TRAIN-TEST SPLIT
# ------------------------------------------------------
# - Splits data into training and testing sets
# - Stratified split ensures class balance
# ======================================================
print("\n" + "="*60)
print("SPLITTING DATA INTO TRAIN AND TEST SETS...")
print("="*60)

X_train, X_test, y_train, y_test = train_test_split(
    X_cleaned,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# ======================================================
# 5. FEATURE EXTRACTION USING TF-IDF
# ------------------------------------------------------
# - Converts text into numerical vectors
# - Uses unigrams + bigrams
# - Removes very common and very rare words
# ======================================================
print("\n" + "="*60)
print("EXTRACTING TF-IDF FEATURES...")
print("="*60)

tfidf = TfidfVectorizer(
    ngram_range=(1, 2),
    max_df=0.75,
    min_df=5
)

X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

print("✓ TF-IDF Feature Shape:", X_train_tfidf.shape)
print(f"✓ Training samples: {X_train_tfidf.shape[0]}, Test samples: {X_test_tfidf.shape[0]}")


# ======================================================
# 6. NAIVE BAYES MODEL
# ------------------------------------------------------
# - Probabilistic classifier
# - Assumes feature independence
# - Fast and commonly used for text classification
# ======================================================
print("\n" + "="*60)
print("TRAINING MODEL 1/4: NAIVE BAYES")
print("="*60)
nb_model = MultinomialNB()
print("Training...", end="", flush=True)
nb_model.fit(X_train_tfidf, y_train)
print(" Done!")

nb_pred = nb_model.predict(X_test_tfidf)
nb_acc = accuracy_score(y_test, nb_pred)
nb_p, nb_r, nb_f1, _ = precision_recall_fscore_support(
    y_test, nb_pred, average='macro'
)

print("Accuracy:", nb_acc)
print("Precision:", nb_p)
print("Recall:", nb_r)
print("F1-Score:", nb_f1)


# ======================================================
# 7. LOGISTIC REGRESSION MODEL
# ------------------------------------------------------
# - Linear classifier
# - Uses One-vs-Rest strategy for multiclass
# - Strong baseline ML model
# ======================================================
print("\n" + "="*60)
print("TRAINING MODEL 2/4: LOGISTIC REGRESSION")
print("="*60)
lr_model = LogisticRegression(max_iter=2000)
print("Training...", end="", flush=True)
lr_model.fit(X_train_tfidf, y_train)
print(" Done!")

lr_pred = lr_model.predict(X_test_tfidf)
lr_acc = accuracy_score(y_test, lr_pred)
lr_p, lr_r, lr_f1, _ = precision_recall_fscore_support(
    y_test, lr_pred, average='macro'
)

print("Accuracy:", lr_acc)
print("Precision:", lr_p)
print("Recall:", lr_r)
print("F1-Score:", lr_f1)


# ======================================================
# 8. SUPPORT VECTOR MACHINE (SVM)
# ------------------------------------------------------
# - Margin-based classifier
# - Performs very well on high-dimensional text data
# - Usually gives best accuracy for NLP problems
# ======================================================
print("\n" + "="*60)
print("TRAINING MODEL 3/4: SUPPORT VECTOR MACHINE")
print("="*60)
svm_model = LinearSVC()
print("Training...", end="", flush=True)
svm_model.fit(X_train_tfidf, y_train)
print(" Done!")

svm_pred = svm_model.predict(X_test_tfidf)
svm_acc = accuracy_score(y_test, svm_pred)
svm_p, svm_r, svm_f1, _ = precision_recall_fscore_support(
    y_test, svm_pred, average='macro'
)

print("Accuracy:", svm_acc)
print("Precision:", svm_p)
print("Recall:", svm_r)
print("F1-Score:", svm_f1)


# ======================================================
# 9. RANDOM FOREST MODEL
# ------------------------------------------------------
# - Ensemble learning technique
# - Combines multiple decision trees
# - Handles non-linear patterns
# ======================================================
print("\n" + "="*60)
print("TRAINING MODEL 4/4: RANDOM FOREST")
print("="*60)
rf_model = RandomForestClassifier(n_estimators=200, random_state=42)
print("Training (this may take a while)...", end="", flush=True)
rf_model.fit(X_train_tfidf, y_train)
print(" Done!")

rf_pred = rf_model.predict(X_test_tfidf)
rf_acc = accuracy_score(y_test, rf_pred)
rf_p, rf_r, rf_f1, _ = precision_recall_fscore_support(
    y_test, rf_pred, average='macro'
)

print("Accuracy:", rf_acc)
print("Precision:", rf_p)
print("Recall:", rf_r)
print("F1-Score:", rf_f1)


# ======================================================
# 10. MODEL PERFORMANCE COMPARISON
# ------------------------------------------------------
# Displays Accuracy, Precision, Recall, F1-score
# for all four models
# ======================================================
results_df = pd.DataFrame({
    "Model": [
        "Naive Bayes",
        "Logistic Regression",
        "Support Vector Machine",
        "Random Forest"
    ],
    "Accuracy": [nb_acc, lr_acc, svm_acc, rf_acc],
    "Precision": [nb_p, lr_p, svm_p, rf_p],
    "Recall": [nb_r, lr_r, svm_r, rf_r],
    "F1-Score": [nb_f1, lr_f1, svm_f1, rf_f1]
})

print("\n===== MODEL COMPARISON =====\n")
print(results_df.sort_values(by="F1-Score", ascending=False))


# ======================================================
# 11. SAVE ALL TRAINED MODELS
# ------------------------------------------------------
# - Saves ALL trained models individually
# - Saves TF-IDF vectorizer using joblib
# ======================================================

# Create models directory if it doesn't exist
models_dir = "models"
if not os.path.exists(models_dir):
    os.makedirs(models_dir)
    print(f"\n✓ Created '{models_dir}/' directory for storing models")

# Save all individual models
print("\n" + "="*60)
print("SAVING ALL TRAINED MODELS...")
print("="*60)

joblib.dump(nb_model, os.path.join(models_dir, "naive_bayes_model.pkl"))
print("✓ Saved: naive_bayes_model.pkl")

joblib.dump(lr_model, os.path.join(models_dir, "logistic_regression_model.pkl"))
print("✓ Saved: logistic_regression_model.pkl")

joblib.dump(svm_model, os.path.join(models_dir, "svm_model.pkl"))
print("✓ Saved: svm_model.pkl")

joblib.dump(rf_model, os.path.join(models_dir, "random_forest_model.pkl"))
print("✓ Saved: random_forest_model.pkl")

joblib.dump(tfidf, os.path.join(models_dir, "tfidf_vectorizer.pkl"))
print("✓ Saved: tfidf_vectorizer.pkl")

print("\n" + "="*60)
print("ALL MODELS SAVED SUCCESSFULLY!")
print("="*60)


# ======================================================
# 12. NLP PREPROCESSING FOR USER INPUT (STEP-BY-STEP)
# ------------------------------------------------------
# Displays output after each NLP preprocessing stage
# for clear understanding and demonstration
# ======================================================

def preprocess_user_input_debug(text):
    print("\n========== NLP PREPROCESSING STEPS ==========\n")

    # Original Text
    print("1️⃣ Original Text:")
    print(text)

    # Lowercasing
    text_lower = text.lower()
    print("\n2️⃣ After Lowercasing:")
    print(text_lower)

    # URL Removal
    text_no_url = re.sub(r'http\S+|www\S+', '', text_lower)
    print("\n3️⃣ After URL Removal:")
    print(text_no_url)

    # Emoji Removal (Complete Coverage)
    emoji_pattern = re.compile(
        "["
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F680-\U0001F6FF"  # transport & map
        "\U0001F700-\U0001F77F"
        "\U0001F780-\U0001F7FF"
        "\U0001F800-\U0001F8FF"
        "\U0001F900-\U0001F9FF"  # supplemental symbols (includes 🤯)
        "\U0001FA00-\U0001FAFF"
        "\U00002702-\U000027B0"
        "\U000024C2-\U0001F251"
        "]+",
        flags=re.UNICODE
    )
    text_no_emoji = emoji_pattern.sub(r'', text_no_url)
    print("\n4️⃣ After Emoji Removal:")
    print(text_no_emoji)

    # Normalize Repeated Characters
    text_normalized = re.sub(r'(.)\1{2,}', r'\1\1', text_no_emoji)
    print("\n5️⃣ After Repeated Character Normalization:")
    print(text_normalized)

    # Remove Special Characters & Numbers
    text_clean = re.sub(r'[^a-z\s]', '', text_normalized)
    print("\n6️⃣ After Special Character & Number Removal:")
    print(text_clean)

    # Tokenization
    tokens = word_tokenize(text_clean)
    print("\n7️⃣ After Tokenization:")
    print(tokens)

    # Stopword Removal
    tokens_no_stop = [
        word for word in tokens
        if word not in stop_words and len(word) > 2
    ]
    print("\n8️⃣ After Stopword Removal:")
    print(tokens_no_stop)

    # Lemmatization
    tokens_lemma = [
        lemmatizer.lemmatize(word)
        for word in tokens_no_stop
    ]
    print("\n9️⃣ After Lemmatization:")
    print(tokens_lemma)

    # Final Cleaned Text
    final_text = " ".join(tokens_lemma)
    print("\n🔟 Final Cleaned Text (Used for ML):")
    print(final_text)

    print("\n===========================================\n")

    return final_text


# ======================================================
# 13. REAL-TIME USER INPUT & PREDICTION (WITH NLP TRACE)
# ======================================================

# Load all trained models
print("\n" + "="*60)
print("LOADING ALL TRAINED MODELS...")
print("="*60)

nb_model_loaded = joblib.load(os.path.join("models", "naive_bayes_model.pkl"))
print("✓ Loaded: Naive Bayes")

lr_model_loaded = joblib.load(os.path.join("models", "logistic_regression_model.pkl"))
print("✓ Loaded: Logistic Regression")

svm_model_loaded = joblib.load(os.path.join("models", "svm_model.pkl"))
print("✓ Loaded: Support Vector Machine")

rf_model_loaded = joblib.load(os.path.join("models", "random_forest_model.pkl"))
print("✓ Loaded: Random Forest")

vectorizer = joblib.load(os.path.join("models", "tfidf_vectorizer.pkl"))
print("✓ Loaded: TF-IDF Vectorizer")

print("\nEnter a news article (type 'exit' to quit):\n")

while True:
    user_text = input("News Text: ")

    if user_text.lower() == "exit":
        print("Exiting...")
        break

    # Step-by-step NLP preprocessing
    clean_text = preprocess_user_input_debug(user_text)

    # Vectorization
    text_vector = vectorizer.transform([clean_text])

    # Get predictions from all models
    print("\n" + "="*60)
    print("PREDICTIONS FROM ALL MODELS")
    print("="*60)
    
    nb_prediction = nb_model_loaded.predict(text_vector)[0]
    print(f"1️⃣ Naive Bayes: {target_names[nb_prediction]}")
    
    lr_prediction = lr_model_loaded.predict(text_vector)[0]
    print(f"2️⃣ Logistic Regression: {target_names[lr_prediction]}")
    
    svm_prediction = svm_model_loaded.predict(text_vector)[0]
    print(f"3️⃣ Support Vector Machine: {target_names[svm_prediction]}")
    
    rf_prediction = rf_model_loaded.predict(text_vector)[0]
    print(f"4️⃣ Random Forest: {target_names[rf_prediction]}")
    
    print("="*60)
    print("-" * 70)
