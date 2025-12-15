# NewsSense - An Explainable NLP-Based News Classification System

## 🎯 Project Overview

**NewsSense** is a full-stack machine learning application that demonstrates the complete lifecycle of a text classification system. The project classifies news articles into 20 distinct categories using advanced Natural Language Processing (NLP) and multiple machine learning algorithms. Rather than being a black-box solution, NewsSense emphasizes **transparency**, **explainability**, and **educational value** by exposing the inner workings of the ML pipeline.

**The goal of NewsSense is to bridge the gap between academic machine learning and real-world deployment by building a transparent, explainable, and full-stack news classification system.**

## 🏗️ System Architecture

The application follows a **three-tier architecture** designed for scalability, modularity, and separation of concerns:

### **1. Machine Learning Layer (Training Pipeline)**
- **File**: `news_classification.py`
- **Purpose**: Offline training, model comparison, and artifact generation
- **Workflow**:
  1. **Data Acquisition**: Fetches the 20 Newsgroups dataset from scikit-learn
  2. **NLP Preprocessing Pipeline**: Applies comprehensive text cleaning and normalization
  3. **Feature Engineering**: Converts text to numerical representations using TF-IDF vectorization
  4. **Model Training**: Trains four distinct ML algorithms in parallel
  5. **Evaluation & Selection**: Compares models using accuracy and F1-score metrics
  6. **Model Persistence**: Serializes trained models and preprocessing artifacts using joblib

### **2. Backend API Layer (Inference Service)**
- **File**: `app.py`
- **Framework**: FastAPI (high-performance async Python web framework)
- **Purpose**: Serves as the ML inference microservice
- **Key Components**:
  - **Lifespan Management**: Loads all models once at startup for fast inference
  - **Request Handling**: Receives raw text via REST API
  - **Text Processing**: Applies the same NLP pipeline used during training
  - **Multi-Model Prediction**: Runs inference across all trained models simultaneously
  - **Response Formatting**: Returns predictions with performance metrics and metadata
- **Endpoints**:
  - `GET /models` - Returns model performance metrics
  - `POST /predict` - Accepts text and returns classification results
  - `GET /categories` - Lists all 20 news categories

### **3. Frontend Layer (User Interface)**
- **Framework**: React + TypeScript with Vite build tool
- **UI Library**: shadcn/ui components built on Radix UI and Tailwind CSS
- **Purpose**: Provides an intuitive interface for model interaction
- **Key Pages**:
  - **HomePage** (`HomePage.tsx`): Explains project architecture, features, and tech stack
  - **ClassifyPage** (`ClassifyPage.tsx`): Interactive classification interface with real-time results
- **Features**:
  - Real-time text classification
  - Visual comparison of all model predictions
  - Performance metrics visualization
  - Best model highlighting
  - Responsive design with modern UI components

---

## 🧠 Machine Learning Pipeline

### **Dataset: 20 Newsgroups**
- **Source**: sklearn.datasets
- **Size**: ~18,000 documents
- **Categories**: 20 distinct news topics including:
  - Technology (comp.graphics, comp.os.ms-windows.misc, etc.)
  - Recreation (rec.autos, rec.motorcycles, rec.sport.baseball, etc.)
  - Science (sci.crypt, sci.electronics, sci.med, sci.space)
  - Politics & Religion (talk.politics.misc, talk.religion.misc, etc.)
  - Miscellaneous (misc.forsale, alt.atheism, etc.)

### **NLP Preprocessing Pipeline**

The text preprocessing pipeline ensures consistent and clean input for the ML models:

1. **Lowercasing**: Normalizes text to lowercase to reduce vocabulary size
2. **URL Removal**: Strips HTTP/HTTPS links using regex
3. **HTML Tag Removal**: Cleans residual HTML markup
4. **Emoji & Special Character Removal**: Keeps only alphabetic characters and spaces
5. **Tokenization**: Splits text into individual words using NLTK
6. **Stopword Removal**: Filters out common English words (a, the, is, etc.)
7. **Lemmatization**: Reduces words to their base form (running → run)
8. **Short Word Filtering**: Removes words with fewer than 3 characters

**Example Transformation**:
```
Input:  "Breaking News!!! 🔥 The latest iPhone 15 Pro review is here: https://example.com"
Output: "breaking news latest iphone pro review"
```

### **Feature Extraction: TF-IDF Vectorization**

- **Algorithm**: Term Frequency-Inverse Document Frequency
- **Configuration**:
  - n-grams: Unigrams + Bigrams (captures single words and two-word phrases)
  - Max features: 10,000 (limits vocabulary to most informative terms)
  - Min/Max document frequency: Removes extremely rare and extremely common words
- **Purpose**: Converts text into high-dimensional numerical vectors suitable for ML algorithms

### **Machine Learning Models**

The system trains and compares **four distinct algorithms**:

#### **1. Multinomial Naive Bayes**
- **Type**: Probabilistic classifier based on Bayes' theorem
- **Strengths**: Fast training and inference, works well with text data
- **Use Case**: Baseline model for text classification
- **Performance**: ~71.35% accuracy

#### **2. Logistic Regression**
- **Type**: Linear classification model with sigmoid activation
- **Strengths**: Interpretable, efficient, strong baseline for high-dimensional data
- **Use Case**: Provides a solid linear decision boundary
- **Performance**: ~73.40% accuracy

#### **3. Support Vector Machine (LinearSVC)**
- **Type**: Maximum-margin linear classifier
- **Strengths**: Excellent for high-dimensional sparse data (like text)
- **Use Case**: Best overall performer for this task
- **Performance**: ~74.89% accuracy ⭐ **Best Model**

#### **4. Random Forest**
- **Type**: Ensemble of decision trees
- **Strengths**: Captures non-linear patterns, robust to overfitting
- **Use Case**: Explores whether ensemble methods improve accuracy
- **Performance**: ~66.23% accuracy

### **Model Evaluation & Selection**

Each model is evaluated using:
- **Accuracy**: Overall correctness across all categories
- **Precision, Recall, F1-Score**: Per-class and macro-averaged metrics
- **Best Model Selection**: Based on weighted F1-score (balances precision and recall)

The system identifies **Support Vector Machine** as the best-performing model and highlights it in the UI.

---

## 🔄 Data Flow & System Workflow

### **Training Phase** (Offline)
```
20 Newsgroups Dataset
    ↓
NLP Preprocessing (lowercase, tokenize, lemmatize, etc.)
    ↓
TF-IDF Vectorization (convert text → numerical vectors)
    ↓
Train/Test Split (80/20 stratified)
    ↓
Train 4 Models in Parallel
    ↓
Evaluate & Compare Performance
    ↓
Save Models & Vectorizer to /models folder
```

### **Inference Phase** (Real-Time)
```
User enters news article in Frontend
    ↓
POST /predict request to FastAPI Backend
    ↓
Apply same NLP preprocessing pipeline
    ↓
Transform text using saved TF-IDF vectorizer
    ↓
Run prediction on all 4 models
    ↓
Collect predictions, performance metrics, and metadata
    ↓
Return JSON response to Frontend
    ↓
Display results with visual comparison
```

---

## 📂 Project Structure

```
NewsSense/
│
├── app.py                          # FastAPI backend (ML inference service)
├── news_classification.py          # Training pipeline (offline ML training)
├── requirements-api.txt            # Python dependencies
│
├── models/                         # Saved ML artifacts (generated after training)
│   ├── naive_bayes_model.pkl
│   ├── logistic_regression_model.pkl
│   ├── svm_model.pkl
│   ├── random_forest_model.pkl
│   └── tfidf_vectorizer.pkl
│
└── frontend/                       # React + TypeScript UI
    ├── src/
    │   ├── App.tsx                 # Main app with routing
    │   ├── pages/
    │   │   ├── HomePage.tsx        # Landing page with project info
    │   │   ├── ClassifyPage.tsx    # Classification interface
    │   │   └── NotFound.tsx        # 404 page
    │   ├── components/
    │   │   ├── NavLink.tsx         # Navigation component
    │   │   └── ui/                 # shadcn/ui components
    │   ├── hooks/                  # Custom React hooks
    │   └── lib/                    # Utility functions
    │
    ├── package.json                # Node.js dependencies
    ├── vite.config.ts              # Vite build configuration
    └── tailwind.config.ts          # Tailwind CSS configuration
```

---

## 🔧 Technology Stack

### **Backend**
- **Python 3.x**: Core programming language
- **FastAPI**: Modern async web framework for building APIs
- **Uvicorn**: ASGI server for running FastAPI
- **scikit-learn**: Machine learning library (model training & evaluation)
- **NLTK**: Natural Language Toolkit for text preprocessing
- **joblib**: Model serialization and persistence
- **NumPy & Pandas**: Data manipulation and numerical operations

### **Frontend**
- **React 18**: Component-based UI library
- **TypeScript**: Type-safe JavaScript superset
- **Vite**: Fast build tool and dev server
- **Tailwind CSS**: Utility-first CSS framework
- **shadcn/ui**: High-quality React components
- **React Router**: Client-side routing
- **TanStack Query**: Server state management

### **Development Tools**
- **Bun**: Fast JavaScript runtime and package manager (frontend)
- **npm**: Alternative package manager (frontend)
- **Git**: Version control

---

## 🎓 Key Technical Concepts Demonstrated

### **1. End-to-End ML System Design**
The project showcases the complete ML lifecycle: data preprocessing, model training, evaluation, deployment via API, and user-facing application.

### **2. Model Comparison & Transparency**
Instead of hiding model selection behind a single prediction, the system exposes all model outputs, allowing users to compare algorithmic approaches.

### **3. Consistent Training-Inference Pipeline**
The same preprocessing steps (text cleaning, vectorization) are applied during both training and inference, ensuring reproducibility and preventing data drift.

### **4. RESTful API Design**
The backend follows REST principles with clear endpoints, JSON responses, and proper HTTP status codes.

### **5. Modern Frontend Architecture**
Uses React hooks, TypeScript for type safety, component composition, and responsive design patterns.

### **6. Performance Optimization**
- Models are loaded once at startup (not per request)
- Async/await for non-blocking I/O
- Efficient TF-IDF sparse matrix operations

### **7. Scalability Considerations**
The separation of training (offline) and inference (API) allows independent scaling and model updates without affecting the frontend.

---

## 🌟 Real-World Applications

This architecture can be adapted for:
- **Sentiment Analysis**: Classify customer reviews as positive/negative/neutral
- **Spam Detection**: Filter spam emails or messages
- **Content Moderation**: Detect toxic or harmful content
- **Document Categorization**: Organize legal, medical, or financial documents
- **Intent Recognition**: Classify user queries in chatbots

---

## 📊 Model Performance Summary

| Model                  | Accuracy | F1-Score | Speed     | Interpretability |
|------------------------|----------|----------|-----------|------------------|
| Naive Bayes            | 71.35%   | 68.53%   | ⚡ Fastest | ⭐⭐⭐             |
| Logistic Regression    | 73.40%   | 71.12%   | ⚡ Fast    | ⭐⭐⭐⭐            |
| **Support Vector Machine** | **74.89%** | **72.89%** | ⚡ Fast    | ⭐⭐              |
| Random Forest          | 66.23%   | 62.34%   | 🐢 Slower  | ⭐                |

---

## 🧪 API Documentation

### **GET /models**
Returns metadata about all trained models.

**Response**:
```json
{
  "models": [
    {
      "name": "Naive Bayes",
      "accuracy": 0.7135,
      "f1_score": 0.6853
    },
    ...
  ],
  "best_model": "Support Vector Machine"
}
```

### **POST /predict**
Classifies input text using all models.

**Request**:
```json
{
  "text": "NASA announces new Mars rover mission..."
}
```

**Response**:
```json
{
  "cleaned_text": "nasa announces new mars rover mission",
  "results": [
    {
      "model": "Naive Bayes",
      "prediction": "sci.space",
      "prediction_confidence": 0.92
    },
    ...
  ],
  "best_model": "Support Vector Machine"
}
```

### **GET /categories**
Lists all 20 news categories.

**Response**:
```json
{
  "categories": [
    "alt.atheism",
    "comp.graphics",
    "sci.space",
    ...
  ]
}
```

---

## 🎯 Design Philosophy

**NewsSense** is built with three core principles:

1. **Transparency**: Users see how different models interpret the same input
2. **Education**: The system teaches ML concepts through interactive visualization
3. **Production-Readiness**: Follows best practices for real-world ML deployment

The project bridges the gap between academic ML tutorials and production systems, demonstrating how to build a robust, scalable, and user-friendly ML application.

---

## 📝 License

This project is open-source and available for educational purposes.

---

**Built with ❤️ by Naveen Arul**
