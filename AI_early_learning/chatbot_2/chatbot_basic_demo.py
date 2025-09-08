# Author: Ryan Wilkerson
# Version: 8/4/25 / 1.1
# Description: A simple chatbot that uses machine learning to classify user input into predefined intents and respond accordingly.

import random
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

import string

def preprocess(text):
    return text.lower().translate(str.maketrans('', '', string.punctuation))

def train_chatbot(data):
    x = [preprocess(item[0]) for item in data] # Phrases
    y = [item[1] for item in data] # Intents

    v = TfidfVectorizer()
    x_vectors = v.fit_transform(x) # Convert phrases to vectors

    model = LogisticRegression()
    model.fit(x_vectors, y)

    return model, v

def chat(model, vectorizer):
    print("Chatbot: Type 'quit' to exit this program")

    responses = {
        "greeting": ["Hello!", "Hi there!", "Hey!", "Greetings!", "How can I help you!", "Howdy there!"],
        "ask_status": ["I'm doing great, how about you?", "All good here, thanks for asking!", "Just fine, what about you?", "Still swinging, how about yourself?"],
        "farewell": ["Goodbye!", "See you later!", "Take care!", "Catch you later!", "Peace out!", "See yah!", "Have a good day!"],
        "gratitude": ["You're welcome!", "Anytime!", "No problem!", "Glad to help!", "My pleasure!", "Glad to be of service!", "Always glad to help!"],
        "response_status": ["Glad to hear that!", "That's great!", "Awesome!", "Good to know!", "Happy to hear that!", "Nice to hear!"],
        "unknown": ["I'm not sure how to respond to that.", "Could you please rephrase?", "I didn't quite get that.", "Can you clarify?", "I'm not sure what you mean."]
    }

    while True:
        user_input = preprocess(input("You: "))

        if user_input == "quit":
            print("ChatBot: Goodbye!")
            break

        input_vector = vectorizer.transform([user_input])

        input_vector = vectorizer.transform([user_input])
        intent = model.predict(input_vector)[0]

        if intent in responses:
            print("ChatBot:", random.choice(responses[intent]))
        else:
            print("ChatBot:", random.choice(responses["unknown"]))

        

if __name__ == "__main__":
    training_data = [
        ("hello", "greeting"),
        ("hi", "greeting"),
        ("hey there", "greeting"),
        ("howdy", "greeting"),
        ("greetings", "greeting"),
        ("what's up", "greeting"),

        ("how are you", "ask_status"),
        ("how's it going", "ask_status"),
        ("what's cooking", "ask_status"),
        ("you good?", "ask_status"),
        ("how is everything", "ask_status"),

        ("bye", "farewell"),
        ("see you later", "farewell"),
        ("farewell", "farewell"),
        ("take care", "farewell"),
        ("peace out", "farewell"),
        ("see yah", "farewell"),

        ("thank you", "gratitude"),
        ("thanks", "gratitude"), 
        ("much appreciated", "gratitude"),
        ("much obliged", "gratitude"),
        ("i appreciate it", "gratitude"),
        ("i'm grateful", "gratitude"),

        ("i'm doing fine", "response_status"),
        ("i'm okay", "response_status"),
        ("feeling great", "response_status"),
        ("i feel sad", "response_status"),
        ("i'm really mad", "response_status"),
        ("i'm happy", "response_status"),
        ("i'm excited", "response_status"),
        ("i'm bored", "response_status"),
        ("i'm tired", "response_status"),
        ("i'm confused", "response_status"),
        ("i'm chilling", "response_status"),
        ("i'm stressed", "response_status")
    ]

    model, vectorizer = train_chatbot(training_data)
    chat(model, vectorizer)
