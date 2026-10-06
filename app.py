import os
import sys
from flask import Flask, render_template, request, jsonify

# Add project root to path so we can import from chatbot
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from chatbot.bot import get_response, model_loaded

app = Flask(__name__)

@app.route('/')
def index():
    """Render the chat interface"""
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    """API endpoint to get bot response"""
    data = request.get_json()
    user_message = data.get('message', '').strip()
    
    if not user_message:
        return jsonify({'error': 'No message provided'}), 400
    
    bot_reply = get_response(user_message)
    return jsonify({'response': bot_reply})

@app.route('/health', methods=['GET'])
def health():
    """Check if model is loaded"""
    return jsonify({'status': 'ok', 'model_loaded': model_loaded})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)