# ======================================================
# NEWS CLASSIFICATION - MEMORY-SAFE TRAINING SCRIPT
# ======================================================
# Trains 6 lightweight models for deployment on 512MB RAM
# Models: Logistic Regression, Linear SVM, Multinomial NB,
#         Complement NB, Bernoulli NB, SGD Classifier
# ======================================================

import re
import joblib
import os
import gc
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB, ComplementNB, BernoulliNB
from sklearn.metrics import accuracy_score, f1_score

# Download required NLTK data
print("Downloading NLTK data...")
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('punkt_tab', quiet=True)

print("\n" + "="*70)
print("MEMORY-SAFE NEWS CLASSIFICATION TRAINING")
print("="*70)

# ======================================================
# 1. LOAD DATASET
# ======================================================
print("\n[1/6] Loading 20 Newsgroups dataset...")
data = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))
X = data.data
y = data.target
target_names = data.target_names

print(f"✓ Loaded {len(X)} documents across {len(target_names)} categories")

# ======================================================
# 2. TEXT PREPROCESSING
# ======================================================
print("\n[2/6] Preprocessing text...")
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    """Clean and normalize text"""
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(w) for w in tokens 
              if w not in stop_words and len(w) > 2]
    return " ".join(tokens)

X_cleaned = [preprocess_text(doc) for doc in X]
print(f"✓ Preprocessed {len(X_cleaned)} documents")

# Free memory
del X
gc.collect()

# ======================================================
# 3. TRAIN/TEST SPLIT
# ======================================================
print("\n[3/6] Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(
    X_cleaned, y, test_size=0.2, random_state=42, stratify=y
)
print(f"✓ Train: {len(X_train)}, Test: {len(X_test)}")

# Free memory
del X_cleaned
gc.collect()

# ======================================================
# 4. TF-IDF VECTORIZATION
# ======================================================
print("\n[4/6] Extracting TF-IDF features...")
vectorizer = TfidfVectorizer(
    max_features=10000,
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)
print(f"✓ Feature matrix shape: {X_train_vec.shape}")

# Free memory
del X_train, X_test
gc.collect()

# Save vectorizer
os.makedirs('models', exist_ok=True)
joblib.dump(vectorizer, 'models/tfidf_vectorizer.pkl')
print("✓ Saved TF-IDF vectorizer")

# Free vectorizer from memory (we'll reload it later)
del vectorizer
gc.collect()

# ======================================================
# 5. TRAIN MODELS ONE-BY-ONE (MEMORY-SAFE)
# ======================================================
print("\n[5/6] Training models sequentially...")

models_config = [
    ("Logistic Regression", LogisticRegression(max_iter=1000, random_state=42)),
    ("Linear SVM", LinearSVC(max_iter=2000, random_state=42)),
    ("Multinomial Naive Bayes", MultinomialNB()),
    ("Complement Naive Bayes", ComplementNB()),
    ("Bernoulli Naive Bayes", BernoulliNB()),
    ("SGD Classifier", SGDClassifier(max_iter=1000, random_state=42))
]

results = []

for idx, (name, model) in enumerate(models_config, 1):
    print(f"\n  [{idx}/6] Training {name}...")
    
    # Train model
    model.fit(X_train_vec, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test_vec)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    print(f"      Accuracy: {accuracy:.4f}")
    print(f"      F1-Score: {f1:.4f}")
    
    # Save model
    filename = name.lower().replace(' ', '_') + '.pkl'
    filepath = os.path.join('models', filename)
    joblib.dump(model, filepath)
    print(f"      ✓ Saved to {filepath}")
    
    # Store results
    results.append({
        'name': name,
        'filename': filename,
        'accuracy': accuracy,
        'f1_score': f1
    })
    
    # CRITICAL: Free memory immediately
    del model, y_pred
    gc.collect()

# ======================================================
# 6. SAVE METADATA
# ======================================================
print("\n[6/6] Saving metadata...")

metadata = {
    'target_names': target_names,
    'models': results
}

joblib.dump(metadata, 'models/metadata.pkl')
print("✓ Saved metadata")

# Free remaining data
del X_train_vec, X_test_vec, y_train, y_test
gc.collect()

# ======================================================
# SUMMARY
# ======================================================
print("\n" + "="*70)
print("TRAINING COMPLETE")
print("="*70)
print("\nModel Performance Summary:")
print(f"{'Model':<30} {'Accuracy':<12} {'F1-Score'}")
print("-" * 70)

for r in results:
    print(f"{r['name']:<30} {r['accuracy']:<12.4f} {r['f1_score']:.4f}")

best_model = max(results, key=lambda x: x['f1_score'])
print(f"\n✓ Best Model: {best_model['name']} (F1: {best_model['f1_score']:.4f})")
print("\n✓ All models saved to ./models/")
print("✓ Ready for deployment!")
