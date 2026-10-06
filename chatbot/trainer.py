import nltk
import json
import pickle
import numpy as np
import random
import re
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score
from collections import Counter
import os

# Download necessary NLTK data
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('punkt_tab', quiet=True)

lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    """
    Preprocess a single text string - IMPROVED VERSION
    - Keeps stop words for context
    - Only removes special characters, keeps letters and spaces
    - Lemmatizes but does not remove short words
    """
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters but keep letters and spaces
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Tokenize
    words = word_tokenize(text)
    
    # Lemmatize each word (keep all words, even short ones)
    words = [lemmatizer.lemmatize(w) for w in words if w]
    
    return ' '.join(words)

def load_and_prepare_data(file_path):
    """Load intents.json and prepare training data"""
    with open(file_path, 'r', encoding='utf-8') as file:
        intents = json.load(file)

    training_sentences = []
    training_labels = []
    labels = []
    responses_for_tag = {}

    for intent in intents['intents']:
        tag = intent['tag']
        labels.append(tag)
        responses_for_tag[tag] = intent['responses']
        
        print(f"Processing intent: {tag} with {len(intent['patterns'])} patterns")
        
        for pattern in intent['patterns']:
            processed_pattern = preprocess_text(pattern)
            if processed_pattern:
                training_sentences.append(processed_pattern)
                training_labels.append(tag)

    unique_labels = sorted(list(set(labels)))
    return training_sentences, training_labels, unique_labels, responses_for_tag

def train_model(training_sentences, training_labels):
    """
    Train classifier with multiclass support
    - solver='lbfgs' works for multiclass (n_classes >= 3)
    - penalty removed (handled via C)
    """
    vectorizer = TfidfVectorizer(
        lowercase=False,
        token_pattern=r'(?u)\b\w+\b',
        analyzer='word',
        min_df=1,
        max_df=1.0,
        ngram_range=(1, 3),          # unigrams, bigrams, trigrams
        sublinear_tf=True,
        use_idf=True,
        smooth_idf=True,
    )
    
    classifier = LogisticRegression(
        max_iter=3000,
        random_state=42,
        C=2.0,                         # less regularization
        class_weight='balanced',        # handle imbalanced classes
        solver='lbfgs'                  # ✅ supports multiclass
    )
    
    # Create pipeline (vectorizer + classifier)
    model = make_pipeline(vectorizer, classifier)
    model.fit(training_sentences, training_labels)
    
    # Print features
    feature_names = vectorizer.get_feature_names_out()
    print(f"\n✅ Vectorizer created with {len(feature_names)} features")
    print(f"Sample features (including trigrams): {list(feature_names[:15])}")
    
    # ===== FIXED CROSS-VALIDATION =====
    # Now using the pipeline (model) instead of just classifier
    if len(training_sentences) >= 50:   # ensure enough data
        try:
            # Use model (pipeline) for cross-validation
            scores = cross_val_score(model, training_sentences, training_labels, cv=5)
            print(f"\n📊 Cross-validation accuracy: {scores.mean():.3f} (+/- {scores.std() * 2:.3f})")
            print(f"   Individual fold scores: {[f'{s:.3f}' for s in scores]}")
        except Exception as e:
            print(f"\n⚠️ Cross-validation skipped due to error: {e}")
    else:
        print("\n⚠️ Cross-validation skipped: Not enough data (minimum 50 patterns required).")
    
    return model

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    file_path = os.path.join(project_root, 'data', 'intents.json')
    
    print(f"Looking for intents file at: {file_path}")
    
    if not os.path.exists(file_path):
        print(f"❌ Error: File not found at {file_path}")
        exit(1)
    
    sentences, labels_list, unique_labels, response_dict = load_and_prepare_data(file_path)
    print(f"\n✅ Loaded {len(sentences)} training patterns for {len(unique_labels)} intents.")
    
    class_dist = Counter(labels_list)
    print("Class distribution:")
    for intent, count in class_dist.items():
        print(f"  {intent}: {count} patterns")
    
    if len(sentences) == 0:
        print("❌ Error: No training data found!")
        exit(1)

    # Train model (this returns the pipeline)
    model = train_model(sentences, labels_list)
    print("✅ Model trained successfully.")

    models_dir = os.path.join(project_root, 'models')
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
        print(f"✅ Created models directory at: {models_dir}")

    try:
        # Save the entire pipeline (contains both vectorizer and classifier)
        with open(os.path.join(models_dir, 'classifier.pkl'), 'wb') as f:
            pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)
        with open(os.path.join(models_dir, 'response_dict.pkl'), 'wb') as f:
            pickle.dump(response_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
        with open(os.path.join(models_dir, 'labels.pkl'), 'wb') as f:
            pickle.dump(unique_labels, f, protocol=pickle.HIGHEST_PROTOCOL)
        print("\n🎉 All files saved successfully!")
    except Exception as e:
        print(f"❌ Error saving files: {e}")
        import traceback
        traceback.print_exc()