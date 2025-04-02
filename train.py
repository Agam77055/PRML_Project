import os
import json
import random
import nltk
import numpy as np
from joblib import dump
from collections import Counter
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import sys
sys.setrecursionlimit(10000)

# Ensure necessary NLTK resources are downloaded
nltk.download('punkt')
nltk.download('stopwords')

# Import preprocessing and models from your package
from preprocessing import PreProcessing
from models import RandomForestClassifierFromScratch

# Define paths to your dataset files
DATASET_DIR = os.path.join(os.getcwd(), 'dataset')
INTENTS_PATH = os.path.join(DATASET_DIR, 'intents.json')
CONTRACTIONS_PATH = os.path.join(DATASET_DIR, 'contractions_dict.json')

# Loading the dependencies
with open(INTENTS_PATH, 'r') as f:
    intents = json.load(f)
with open(CONTRACTIONS_PATH, 'r') as f:
    contractions_dict = json.load(f)

# Optionally, define a synonym replacement function for data augmentation.
def synonym_replacement(tokens, limit):
    augmented_sentences = []
    for i in range(len(tokens)):
        synonyms = []
        for syn in wordnet.synsets(tokens[i]):
            for lemma in syn.lemmas():
                synonyms.append(lemma.name())
        if synonyms:
            num_augmentations = min(limit, len(synonyms))
            sampled_synonyms = random.sample(synonyms, num_augmentations)
            for synonym in sampled_synonyms:
                augmented_tokens = tokens[:i] + [synonym] + tokens[i+1:]
                augmented_sentences.append(' '.join(augmented_tokens))
    return augmented_sentences

# Initialize your preprocessing components.
tokenizer = PreProcessing.Tokenizer(contractions_dict=contractions_dict)
vectorizer = PreProcessing.tf_idf_Vectorizer(max_features=1000)

lemmatizer = WordNetLemmatizer()
stopwords = set(nltk.corpus.stopwords.words('english'))

text_data = []
labels = []
limit_per_tag = 40

# Loop over each intent and its pattern examples.
for intent in intents['intents']:
    augmented_sentences_per_tag = 0
    for example in intent['patterns']:
        # Tokenize using NLTK (or you can use your custom tokenizer here too)
        tokens = nltk.word_tokenize(example.lower())
        # Remove stopwords and lemmatize
        filtered_tokens = [
            lemmatizer.lemmatize(token)
            for token in tokens if token not in stopwords and token.isalpha()
        ]
        if filtered_tokens:
            # Save the original (processed) sentence.
            text_data.append(' '.join(filtered_tokens))
            labels.append(intent['tag'])
            # Generate augmented sentences using synonym replacement.
            augmented_sentences = synonym_replacement(filtered_tokens, limit_per_tag - augmented_sentences_per_tag)
            for aug_sent in augmented_sentences:
                text_data.append(aug_sent)
                labels.append(intent['tag'])
                augmented_sentences_per_tag += 1
                if augmented_sentences_per_tag >= limit_per_tag:
                    break

# Transform the text data using your custom TF-IDF vectorizer.
X = vectorizer.fit_transform(text_data)
y = np.array(labels)

# # Train the RandomForest model from scratch.
model = RandomForestClassifierFromScratch(
    n_estimators=200,
    max_depth=None,
    min_samples_split=5,
    min_samples_leaf=1
)
model.fit(X, y)

# Optionally, evaluate your model on a hold-out set.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy: {accuracy:.4f}")

# Save the trained model and vectorizer.
dump(model, 'Random_forest.joblib')
dump(vectorizer, 'vectorizer.joblib')

# with open('best_model.pkl', 'wb') as f:
#     pickle.dump(model, f)
# with open('vectorizer.pkl', 'wb') as f:
#     pickle.dump(vectorizer, f)

print("Training completed. Model and vectorizer saved as 'Random_forest.joblib' and 'vectorizer.joblib'.")
