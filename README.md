# ChatSphere - Advanced College Chatbot Framework

## Introduction

ChatSphere is an advanced chatbot framework developed from scratch using Python with enhanced NLP capabilities, specifically designed for college environments. Built upon our initial work with DailyDialog dataset, this final version features a custom-built institutional dataset and robust deployment pipeline.

Key capabilities:
- Intent classification (Naive Bayes, Decision Tree, Random Forest)
- Advanced text preprocessing with MWE recognition and contraction handling
- TF-IDF feature extraction with mathematical rigor
- Containerized deployment on Google Cloud with Vercel frontend
- Contextual dialogue management

## System Architecture

### Core Components

   **Text Preprocessing Pipeline**
    Multi-Word Expression (MWE) Recognition using WordNet
    Comprehensive contraction handling
    Advanced hyphen/punctuation management
    Case normalization and token splitting

 **Feature Extraction**
   - Custom TF-IDF Vectorizer with smoothing:
     ```
     TF-IDF(t,d) = (Count(t,d)/∑Count(t',d)) × [log((N+1)/(DF(t)+1)) + 1]
     ```
   - Vocabulary construction from tokenized corpus

 **Intent Classifiers**
   - **Multinomial Naive Bayes** (90.48% accuracy)
     - Laplace smoothing implementation
     ```
     P(t|c) = (Count(t,c)+1)/(∑Count(t',c)+|V|)
     ```
   - **Decision Tree** (89.08% accuracy)
     - Gini Index and Entropy based splitting
   - **Random Forest** (91.11% accuracy - best performing)
     - Ensemble of decision trees with mode aggregation

 **Deployment Architecture**
   - Containerized using Docker
   - Flask backend on Google Cloud
   - Express/Node.js frontend on Vercel
   - Vertex AI with L4 GPU for training

## Data Files
- Place these files in your working directory:

- intents.json - Custom college dataset with tags, patterns and responses

- (Optional) DailyDialog files for baseline comparison
## 3. Initialize the Chatbot
    from chatbot import ChatSphere

### Create chatbot instance (default: Random Forest)
    chatbot = ChatSphere(bot_name="CollegeBot")

### Train with custom college data
    chatbot.train(data_file="intents.json")
## Deployment Guide

- Containerization

### Sample Dockerfile
    FROM python:3.9
    WORKDIR /app
    COPY requirements.txt .
    RUN pip install -r requirements.txt
    COPY . .
    CMD ["python", "app.py"]

- Google Cloud Deployment
    Build Docker image: docker build -t chatsphere .

    Push to Google Container Registry

    Deploy to Cloud Run with automatic scaling

- Frontend on Vercel
    Configure frontend with backend API key

    Deploy using Vercel CLI: vercel --prod

- Training Infrastructure
    Use College_Chatbot.ipynb on Vertex AI (Colab Enterprise)

    L4 GPU recommended for training
## Entity Definitions

- Academic Entities

    chatbot.add_entity_pattern(
        "course_code",
        r"\b[A-Z]{2,4}\s?\d{3}\b"  # e.g., "CS 101"
    )

- Location Entities
    chatbot.add_entity_pattern(
        "campus_location",
        r"\b(?:library|lab complex|academic block)\b"
    )

- Administrative Terms

    chatbot.add_entity_pattern(
        "admin_term",
        r"\b(?:registrar|time table|exam schedule)\b"
    )


## Training Options

- With Custom College Data
    chatbot.train(
        data_file="intents.json",
        test_size=0.2,
        vectorizer_params={"max_features": 1500}
    )

- Performance Tuning

## Random Forest hyperparameters
    chatbot.set_model_params(
        n_estimators=200,
        max_depth=10,
        min_samples_split=5
    )
## Interactive Chat

    Example Session

    You: Where is the CS101 class?
    ChatSphere: CS101 is held in Academic Block B, Room 203.

    You: When is the exam registration deadline?
    ChatSphere: The deadline for exam registration is November 15th.
    I detected: date: November 15th

    Reset Conversation

    chatbot.reset()  # Clears dialogue history
## Customization

- Add New Intents
    Edit intents.json to include:
    {
      "tag": "new_intent",
      "patterns": ["sample queries"],
      "responses": ["appropriate responses"]
    }

- Modify Classifier
    Options: 'naive_bayes', 'decision_tree', 'random_forest'
    chatbot = ChatSphere(model_type="random_forest")

## Troubleshooting

- WordNet Issues
    import wn
    wn.download("omw-en")

- Deployment Errors
    Verify API keys in Google Cloud and Vercel

- Check container logs: gcloud logging read

- Performance Issues
    Increase TF-IDF max_features

- Adjust classifier hyperparameters

- Expand training data in intents.json

## Future Work

- Implement transformer architectures

- Add continuous learning pipeline

- Integrate with college databases

- Develop mobile interface
## Contributors

- Aarav Dawer Bhojwani (Naive Bayes)

- Agam Harpreet Singh & Ishan Shah (Random Forest, Deployment)

- Nirmal Kumar Godara (Decision Tree)

- Mahi Chouhan (Text Preprocessing)
# Project Links

- [Live Demo](https://prml-project-tan.vercel.app/)
- [GitHub Repository](https://github.com/Agam77055/PRML_Project) 
- [Training Notebook](https://colab.research.google.com/drive/1BIHniBGs5HLxwrkyzVveOI5J0HNIwZl0)
