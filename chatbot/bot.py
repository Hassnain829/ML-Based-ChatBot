import pickle
import nltk
import random
import re
import os
import numpy as np
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

lemmatizer = WordNetLemmatizer()

# Get paths
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
models_dir = os.path.join(project_root, 'models')

# Load the trained model and data
try:
    classifier_path = os.path.join(models_dir, 'classifier.pkl')
    response_path = os.path.join(models_dir, 'response_dict.pkl')
    labels_path = os.path.join(models_dir, 'labels.pkl')
    
    with open(classifier_path, 'rb') as f:
        classifier = pickle.load(f)
    with open(response_path, 'rb') as f:
        response_dict = pickle.load(f)
    with open(labels_path, 'rb') as f:
        labels = pickle.load(f)
    
    # Get the vectorizer from pipeline
    vectorizer = classifier.named_steps['tfidfvectorizer']
    
    model_loaded = True
    print("✅ Bot model loaded successfully from:", models_dir)
    print(f"✅ Available intents: {list(response_dict.keys())}")
    
except FileNotFoundError as e:
    print(f"❌ Model files not found in {models_dir}")
    print("Please run trainer.py first to train the model.")
    model_loaded = False
    classifier, response_dict, labels = None, None, None
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model_loaded = False
    classifier, response_dict, labels = None, None, None

def preprocess_text(text):
    """Preprocess text exactly like training"""
    # Tokenize and lemmatize
    words = word_tokenize(text.lower())
    words = [lemmatizer.lemmatize(w) for w in words]
    # Remove punctuation and non-alphabetic tokens
    words = [w for w in words if w.isalpha()]
    return ' '.join(words)

def get_response(user_message):
    """Get a response from the bot with debugging."""
    if not model_loaded:
        return "I'm sorry, the bot model is currently unavailable."
    
    print(f"\n🔍 Debug: User message: '{user_message}'")
    
    # Clean and preprocess the message
    cleaned_message = preprocess_text(user_message)
    print(f"🔍 Debug: Cleaned message: '{cleaned_message}'")
    
    if not cleaned_message:  # Handle empty message after cleaning
        return "Please ask me something about BMYBrand services, pricing, or policies!"
    
    try:
        # Transform using the vectorizer
        message_vectorized = vectorizer.transform([cleaned_message])
        
        # Predict the intent
        predicted_tag = classifier.predict([cleaned_message])[0]
        
        # Get confidence scores for all classes
        probabilities = classifier.predict_proba([cleaned_message])[0]
        confidence = max(probabilities)
        
        # Get all classes and their probabilities
        all_probs = dict(zip(classifier.classes_, probabilities))
        
        print(f"🔍 Debug: Predicted tag: {predicted_tag}")
        print(f"🔍 Debug: Confidence: {confidence:.2f}")
        print("🔍 Debug: Top 3 predictions:")
        
        # Sort and show top 3 predictions
        sorted_probs = sorted(all_probs.items(), key=lambda x: x[1], reverse=True)[:3]
        for tag, prob in sorted_probs:
            print(f"   - {tag}: {prob:.2f}")
        
        # Set confidence threshold
        if confidence < 0.05:  # Lowered threshold for testing
            return "I'm not quite sure I understand. Could you rephrase that? You can ask me about our website packages, logo design, refund policy, or services."
        
        # Get a random response from the list for that tag
        if predicted_tag in response_dict and response_dict[predicted_tag]:
            return random.choice(response_dict[predicted_tag])
        else:
            return "I'm not sure how to respond to that."
            
    except Exception as e:
        print(f"❌ Error in prediction: {e}")
        import traceback
        traceback.print_exc()
        return "I encountered an error processing your request. Please try again."

# Test function with known patterns from training
def test_bot():
    """Test the bot with exact patterns from training"""
    print("\n=== Testing with Training Patterns ===\n")
    
    # These should exactly match patterns in intents.json
    test_queries = [
        "Hi",
        "Hello",
        "What services do you offer?",
        "How much is the basic website?",
        "What is your refund policy?",
        "Bye"
    ]
    
    for query in test_queries:
        response = get_response(query)
        print(f"\n📝 Query: {query}")
        print(f"🤖 Response: {response}")
        print("-" * 50)

if __name__ == "__main__":
    test_bot()
    
    print("\n=== Interactive Mode (type 'quit' to exit) ===\n")
    while True:
        msg = input("\nYou: ")
        if msg.lower() in ['quit', 'exit', 'bye']:
            print("Bot: Goodbye! Have a great day!")
            break
        resp = get_response(msg)
        print(f"Bot: {resp}")