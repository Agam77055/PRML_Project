
## Introduction - 

This chatbot is built from scratch using Python with advanced NLP capabilities. 

- It supports:

    Intent classification (SVM, Naive Bayes, or Random Forest)

    Entity extraction (dates, times, locations, etc.)

    Contextual dialogue management

    Response generation based on DailyDialog dataset



## Setup Instructions 

### 1. Prerequisites : 

    # Install required packages
    pip install numpy pandas wn

### 2. Data Files : 
    Place these files in your working directory (or the chatbot will use sample data):

    dialogues_text.txt - Conversation transcripts

    dialogues_topic.txt - Topic labels (1-10)

    dialogues_act.txt - Dialogue acts (1-4)

    dialogues_emotion.txt - Emotion labels (0-6)

### 3. Initialize the Chatbot : 
    from chatbot import EnhancedChatbot

    # Create chatbot instance
    chatbot = EnhancedChatbot(bot_name="YourBotName", model_type="naive_bayes")

    # Train with DailyDialog data
    chatbot.train(
        dialogues_file="dialogues_text.txt",
        topics_file="dialogues_topic.txt",
        acts_file="dialogues_act.txt",
        emotions_file="dialogues_emotion.txt"
    )

### 
## Core Components

### 1. Intent Classifier
- Models Available: Naive Bayes (default), SVM, Random Forest

- Training: Uses dialog acts from DailyDialog as intent labels

- Accuracy: Typically achieves 75-85% on test data

### 2. Entity Extractor
- Predefined patterns for detecting:

- Dates/Times

- Locations

- Numbers

- Emails

- Domain-specific terms (health, finance)

### 3. Dialogue Manager
- State machine with transitions based on dialog acts

- Maintains conversation context

- Handles 4 core states: inform, question, directive, commissive

### 4. Response Generator
- Retrieval-based using TF-IDF similarity

- Falls back to intent-specific templates

- Incorporates emotion and topic context
## Entity Definitions 
### 1. Date Entities: 
    # Pattern examples:
    "January 15, 2023", "01/15/2023", "today", "tomorrow"
    chatbot.entity_extractor.add_entity_pattern(
        "date",
        r"\b(?:jan(?:uary)?|feb(?:ruary)?|...)\s+\d{1,2}...\b"
    )

### 2. Time Entities : 
    # Pattern examples:
    "3:45 PM", "14:30", "10am"
    chatbot.entity_extractor.add_entity_pattern(
        "time",
        r"\b(?:\d{1,2}:\d{2}(?::\d{2})?(?:\s*[ap]\.?m\.?)?\b"
    )

### 3. Location Entities : 
    # Pattern examples:
    "New York, NY", "Paris France", "Tokyo"
    chatbot.entity_extractor.add_entity_pattern(
        "location",
        r"\b(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*,?\s+(?:[A-Z]{2}|[A-Z][a-z]+)\b"
    )

### 4. Health Terms : 
    # Pattern examples:
    "doctor", "symptoms", "prescription"
    chatbot.entity_extractor.add_entity_pattern(
        "health_terms",
        r"\b(?:doctor|hospital|symptoms|treatment|medicine)\b"
    )

### 5. Finance Terms : 
    # Pattern examples:
    "bank account", "loan", "investment"
    chatbot.entity_extractor.add_entity_pattern(
        "finance_terms",
        r"\b(?:money|bank|account|loan|credit|investment)\b"
    )


## Training the Chatbot 

### 1. With DailyDialog Data : 
    chatbot.train(
        dialogues_file="dialogues_text.txt",
        topics_file="dialogues_topic.txt",
        acts_file="dialogues_act.txt",
        emotions_file="dialogues_emotion.txt",
        test_size=0.2  # 20% for evaluation
    )


### 2. With Sample Data (Fallback) : 
    # Automatically used if DailyDialog files not found
    chatbot._train_with_sample_data()

### 3. Custom Training : 

    # Prepare your own data in this format:
    custom_data = {
        'utterances': ["hello", "what time is it"],
        'acts_labels': ["greeting", "question"],
        # ... other fields
    }



## Interactive Chat  

### 1. Start Chat Session : 
    chatbot.chat() 

### 2. Example Conversation Flow : 
    You: Hi there!
    DailyBot: Hello! How can I help you today?

    You: What's the weather like in Paris?
    DailyBot: I don't have access to weather information currently.
    I detected: location: Paris

    You: Can you tell me a joke?
    DailyBot: Why don't scientists trust atoms? Because they make up everything!

### 3. Reset Conversation : 

    chatbot.reset()  # Clears dialogue history and context 


## Customize Options 

### 1. Change Model Type : 

    # Options: 'naive_bayes' (default), 'svm', 'random_forest'
    chatbot = EnhancedChatbot(model_type="random_forest")

### 2. Add Custom Entities : 
    chatbot.entity_extractor.add_entity_pattern(
        "food_item",
        r"\b(?:pizza|burger|sushi|pasta)\b"
    )

### 3. Modify Response Templates

- Edit the _create_sample_data() method or provide your own responses DataFrame. 



## Troubleshooting 

### 1. WordNet Download Issues : 

    # Manual download if automatic fails
    import wn
    wn.download("omw-en")

### 2. Missing Data Files : 
- The chatbot will automatically use sample data if DailyDialog files aren't found

- Download DailyDialog dataset from: DailyDialog Website

### 3. Performance Tuning :
- Increase max_features in vectorizer for better accuracy (default: 1000)

- Adjust model hyperparameters in the classifier classes 

