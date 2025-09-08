# Author: Ryan Wilkerson
# Version: 8/1/2025 / 1.0
# Description: A simple chatbot that uses logistic regression for intent classification.

import random
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

def train_chatbot(training_data):
    texts = [x[0] for x in training_data]
    labels = [x[1] for x in training_data]
    v = CountVectorizer()
    X = v.fit_transform(texts)
    model = LogisticRegression()
    model.fit(X, labels)
    return model, v

def chat(model, vectorizer, responses):
    print("Chatbot: Type 'quit' to exit this program")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "quit":
            print("ChatBot: Goodbye!")
            break
        X_test = vectorizer.transform([user_input])
        intent = model.predict(X_test)[0]
        reply = random.choice(responses.get(intent, ["I'm not sure how to respond to that."]))
        print("ChatBot:", reply)

if __name__ == "__main__":
    training_data = [
        ("hi", "greeting"),
        ("hello", "greeting"),
        ("how are you", "ask_status"),
        ("bye", "farewell"),
        ("thank you", "gratitude"),
        ("good", "feeling_ok"),
        ("i'm good", "feeling_ok"),
        ("i'm fine", "feeling_ok"),
        ("doing well", "feeling_ok")
    ]

    responses = {
        "greeting": ["Hello!", "Hi there!"],
        "ask_status": ["I'm doing great, how about you?"],
        "farewell": ["Goodbye!", "See you later!"],
        "gratitude": ["You're welcome!", "Anytime!"],
        "feeling_ok": ["Glad to hear that!", "That's great!", "Awesome!"]

    }

    model, vectorizer = train_chatbot(training_data)
    chat(model, vectorizer, responses)
