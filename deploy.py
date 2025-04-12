import os
import json
import random
import nltk
from joblib import load
from flask import Flask, request, jsonify
from nltk.stem import WordNetLemmatizer

# Ensure necessary NLTK resources are downloaded
nltk.download('punkt')
nltk.download('stopwords')

# Import your PreProcessing module
from preprocessing import PreProcessing
from models import RandomForestClassifierFromScratch, DecisionTreeClassifierFromScratch

# Define paths to required files in the dataset
DATASET_DIR = os.path.join(os.getcwd(), 'dataset')
INTENTS_PATH = os.path.join(DATASET_DIR, 'intents.json')
CONTRACTIONS_PATH = os.path.join(DATASET_DIR, 'contractions_dict.json')

# Load intents and contractions dictionary
with open(INTENTS_PATH, 'r') as f:
    intents = json.load(f)
with open(CONTRACTIONS_PATH, 'r') as f:
    contractions_dict = json.load(f)

# Load the saved vectorizer and model.
vectorizer = load('vectorizer.joblib')
best_model = load('Random_forest.joblib')

# Initialize your preprocessing objects.
tokenizer = PreProcessing.Tokenizer(contractions_dict=contractions_dict)
lemmatizer = WordNetLemmatizer()
stopwords = set(nltk.corpus.stopwords.words('english'))

# Define a helper function for generating a chatbot response.
def chatbot_response(user_input, threshold=0.2):    
    # Tokenize and process the input using your custom tokenizer.
    tokens = tokenizer.tokenize(user_input)
    filtered_tokens = [
        lemmatizer.lemmatize(token) for token in tokens 
        if token not in stopwords
    ]
    processed_input = ' '.join(filtered_tokens)
    
    # Transform the processed text into a vector.
    input_vector = vectorizer.transform([processed_input])
    
    # Get predicted probabilities using the model's predict_proba method.
    probabilities = best_model.predict_proba(input_vector)
    max_prob = probabilities.max()
    predicted_intent = best_model.classes_[probabilities.argmax()]
    
    # Check if the maximum probability is below the threshold.
    if max_prob < threshold:
        return "I'm sorry, I don't understand."
    else:
        # Return a random response for the predicted intent from the intents file.
        for intent in intents['intents']:
            if intent['tag'] == predicted_intent:
                return random.choice(intent['responses'])
        # Fallback in case no matching intent is found.
        return "I'm not sure how to respond to that."

# Create the Flask application.
app = Flask(__name__)

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    if not data or 'user_input' not in data:
        return jsonify({'error': 'Missing user_input in request'}), 400
    response = chatbot_response(data['user_input'])
    return jsonify({'response': response})

if __name__ == '__main__':
    # Run the Flask app.
    app.run(host='0.0.0.0', port=8080, debug=True)
