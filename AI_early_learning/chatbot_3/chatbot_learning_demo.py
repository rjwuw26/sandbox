# Author: Ryan Wilkerson
# Version: 8/5/25 / 1.2
# This is a simple, experimental AI chatbot designed for learning purposes. It uses 
# sentence embeddings from the SentenceTransformers library (SBERT) to understand 
# user input and predict intents based on similarity to predefined training phrases. 

import string
import random
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

class IntelligentChatBot:
    """
    An intelligent chatbot that uses sentence embeddings and a machine learning 
    model to classify user input into predefined intents and respond with context-appropriate responses.

    This chatbot uses a sentence transformer model (SBERT) to convert user input and training phrases into high dimensional vectors.
    It then compares similarity scores to predict the closest matching intent and respond accordingly.
    """

    def __init__(self, training_data, responses):
        """
        Initializes the chatbot with training data and predefined responses.

        Args:
            training_data (list): A list of tuples containing phrases and their corresponding intents.
            responses (dict): A dictionary mapping intents to lists of possible responses.
        """
        self.training_data = training_data
        self.responses = responses
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

        self.phrases = [self.preprocess(item[0]) for item in training_data]
        self.intents = [item[1] for item in training_data]

        self.embedding = self.model.encode(self.phrases)

    def preprocess(self, text):
        """
        Preprocesses the input text by converting it to lowercase and removing punctuation.
        
        Args:
            text (str): The input text to preprocess.
        """
        return text.lower().translate(str.maketrans('', '', string.punctuation))
        

    def embed_input(self, text):
        """
        Converts the input text into a high-dimensional vector using a sentence transformer model.
        
        Args:
            text (str): The input text to embed.
        """
        return self.model.encode([self.preprocess(text)])[0]

    def predict_intent(self, text, threshold=0.5):
        """
        Predicts the intent of the input text by comparing its embedding with those of training phrases.
        
        Args:
            text (str): The input text to classify.
        """
        input_vec = self.embed_input(text)
        similarities = cosine_similarity([input_vec], self.embedding)[0]
        best_index = similarities.argmax()
        if similarities[best_index] < threshold:
            return "unknown"
        return self.intents[best_index]

    def get_response(self, intent):
        """
        Retrieves a random response for the given intent from the predefined responses.
        
        Args:
            intent (str): The predicted intent for which to retrieve a response.
        """
        if intent in self.responses:
            return random.choice(self.responses[intent])
        else:
            return random.choice(self.responses["unknown"])

    def chat(self):
        """
        Starts the chatbot interaction loop, allowing users to input text and receive responses.
        """
        print("Sora: Hi! I'm Sora, your intelligent chatbot. Type 'quit' or 'exit' to stop chatting.")

        while True:
            user_input = input("You: ")
            if user_input.lower() in ["quit", "exit"]:
                print("Sora: Goodbye!")
                break

            # Predict the intent of the user input
            intent = self.predict_intent(user_input)

            # Get a response based on the predicted intent
            response = self.get_response(intent)

            # Output the chatbot response
            print(f"Sora: {response}")

training_data = [

    # Greetings

    ("hello", "greeting"),
    ("hi", "greeting"),
    ("hey there", "greeting"),
    ("howdy", "greeting"),
    ("greetings", "greeting"),
    ("what's up", "greeting"),
    ("yo", "greeting"),
    ("hiya", "greeting"),

    # Asking about status

    ("how are you", "ask_status"),
    ("how's it going", "ask_status"),
    ("what's new", "ask_status"),
    ("how are things", "ask_status"),
    ("you good?", "ask_status"),
    ("how's everything", "ask_status"),
    ("how's life", "ask_status"),
    ("how's your day", "ask_status"),
    ("how's your week", "ask_status"),

    # Farewells

    ("bye", "farewell"),
    ("see you later", "farewell"),
    ("farewell", "farewell"),
    ("take care", "farewell"),
    ("peace out", "farewell"),
    ("see yah", "farewell"),
    ("catch you later", "farewell"),
    ("goodbye", "farewell"),
    ("later", "farewell"),
    ("until next time", "farewell"),
    ("have a good day", "farewell"),
    ("have a nice day", "farewell"),
    ("have a great day", "farewell"),
    ("have a wonderful day", "farewell"),
    ("have a lovely day", "farewell"),

    # Gratitude

    ("thank you", "gratitude"),
    ("thanks", "gratitude"),
    ("much appreciated", "gratitude"),
    ("much obliged", "gratitude"),
    ("i appreciate it", "gratitude"),
    ("i'm grateful", "gratitude"),
    ("thanks a lot", "gratitude"),
    ("thank you very much", "gratitude"),
    ("i owe you one", "gratitude"),
    ("i can't thank you enough", "gratitude"),
    ("you're a lifesaver", "gratitude"),
    ("you're awesome", "gratitude"),
    ("you're the best", "gratitude"),
    ("i really appreciate it", "gratitude"),
    ("i'm thankful", "gratitude"),
    
    # Feeling Sad

    ("I'm feeling sad", "feeling_sad"),
    ("I'm feeling down", "feeling_sad"),
    ("I'm feeling blue", "feeling_sad"),
    ("I'm feeling depressed", "feeling_sad"),
    ("I'm feeling low", "feeling_sad"),
    ("Not doing too well", "feeling_sad"),
    ("I'm pretty upset", "feeling_sad"),
    ("I feel heartbroken", "feeling_sad"),
    ("I'm feeling lonely", "feeling_sad"),
    ("I'm really disappointed", "feeling_sad"),
    ("I'm feeling hopeless", "feeling_sad"),
    ("Things aren't going well", "feeling_sad"),
    ("Feeling down in the dumps", "feeling_sad"),


    # Feeling Angry

    ("I'm really mad", "feeling_angry"),
    ("I'm angry", "feeling_angry"),
    ("I'm furious", "feeling_angry"),
    ("That made me so mad", "feeling_angry"),
    ("I'm irritated", "feeling_angry"),
    ("I'm annoyed", "feeling_angry"),
    ("I'm frustrated", "feeling_angry"),
    ("I'm livid", "feeling_angry"),
    ("I'm pissed off", "feeling_angry"),
    ("I can't stand this", "feeling_angry"),
    ("This is so infuriating", "feeling_angry"),


    # Feeling Worried/Anxious

    ("I'm stressed", "feeling_worried"),
    ("I'm anxious", "feeling_worried"),
    ("I'm nervous", "feeling_worried"),
    ("I feel uneasy", "feeling_worried"),
    ("I'm worried about something", "feeling_worried"),
    ("I can't stop worrying", "feeling_worried"),
    ("I'm on edge", "feeling_worried"),
    ("I'm tense", "feeling_worried"),
    ("Feeling overwhelmed", "feeling_worried"),
    ("I feel restless", "feeling_worried"),


    # Feeling Happy

    ("I'm feeling great", "feeling_happy"),
    ("I'm so happy", "feeling_happy"),
    ("I'm excited", "feeling_happy"),
    ("I'm joyful", "feeling_happy"),
    ("Life is good", "feeling_happy"),
    ("I'm thrilled", "feeling_happy"),
    ("I'm content", "feeling_happy"),
    ("I'm really pleased", "feeling_happy"),
    ("I'm delighted", "feeling_happy"),
    ("I'm feeling fantastic", "feeling_happy"),


    # Feeling Disgust

    ("That disgusts me", "feeling_disgust"),
    ("I'm grossed out", "feeling_disgust"),
    ("That's really unpleasant", "feeling_disgust"),
    ("I feel sick about it", "feeling_disgust"),
    ("That makes me cringe", "feeling_disgust"),
    ("I'm repulsed", "feeling_disgust"),
    ("That's nasty", "feeling_disgust"),


    # Feeling Fear/Scared

    ("I'm scared", "feeling_fear"),
    ("I'm afraid", "feeling_fear"),
    ("I'm terrified", "feeling_fear"),
    ("I feel anxious about this", "feeling_fear"),
    ("I'm nervous about what's coming", "feeling_fear"),
    ("I feel uneasy", "feeling_fear"),
    ("I'm worried this will go wrong", "feeling_fear"),
    ("I'm frightened", "feeling_fear"),


    # Feeling Confused

    ("I'm confused", "feeling_confused"),
    ("I don't understand", "feeling_confused"),
    ("I'm lost", "feeling_confused"),
    ("This doesn't make sense", "feeling_confused"),
    ("I'm puzzled", "feeling_confused"),
    ("I'm unsure", "feeling_confused"),
    ("I feel mixed up", "feeling_confused"),


    # Feeling Tired

    ("I'm tired", "feeling_tired"),
    ("I'm exhausted", "feeling_tired"),
    ("I'm worn out", "feeling_tired"),
    ("I need to rest", "feeling_tired"),
    ("I'm drained", "feeling_tired"),
    ("I can't keep going", "feeling_tired"),


    # Feeling Calm/Relaxed

    ("I'm relaxed", "feeling_calm"),
    ("I'm chilling", "feeling_calm"),
    ("Feeling peaceful", "feeling_calm"),
    ("I'm at ease", "feeling_calm"),
    ("I'm calm", "feeling_calm"),
    ("Everything's cool", "feeling_calm"),


    # General Positive/Neutral

    ("I'm okay", "feeling_neutral"),
    ("I'm fine", "feeling_neutral"),
    ("I'm doing well", "feeling_neutral"),
    ("Not bad", "feeling_neutral"),
    ("So-so", "feeling_neutral"),
    ("Pretty good", "feeling_neutral"),


    # Help requests
    ("Can you help me?", "help_request"),
    ("I need some help", "help_request"),
    ("Can you assist me?", "help_request"),
    ("I need assistance", "help_request"),

    # Time queries
    ("What time is it?", "time_query"),
    ("Can you tell me the time?", "time_query"),
    ("Do you know what time it is?", "time_query"),

    # Capability questions
    ("What can you do?", "capabilities_query"),
    ("Tell me your capabilities", "capabilities_query"),
    ("What are your features?", "capabilities_query"),

]


responses = {
    "greeting": [
        "Hey! How's it going?",
        "Hello there! What can I do for you today?",
        "Hi! Glad to see you.",
        "Howdy! What's on your mind?",
        "Yo! Ready to chat?",
        "Greetings! How can I assist?",
        "Hi! What brings you here today?",
        "Hello! How can I help you?",
        "Hey there! What’s up?",
        "Hi! How can I assist you today?",
    ],
    "ask_status": [
        "I'm doing great, thanks for asking! How about you?",
        "All good here. What's new with you?",
        "I'm here and ready to help!",
        "Doing well! What about you?",
        "I'm fine, thanks! How's your day going?",
        "Pretty good! How about yourself?",
        "I'm doing well, thanks! What can I do for you?",
        "I'm doing great, how about you?",
        "All good here, thanks for asking! What about you?",
        "Just fine, what about you?",
        "Still swinging, how about yourself?",
        "I'm doing well, thanks! How can I assist you today?",
    ],
    "farewell": [
        "Goodbye! Take care.",
        "See you later! Have a great day!",
        "Catch you next time!",
        "Peace out! Stay safe.",
        "Until next time, friend.",
        "Have a wonderful day ahead!",
        "See you later! Don't be a stranger!",
        "Take care! Looking forward to our next chat.",
        "Catch you later! Stay awesome!",
    ],
    "gratitude": [
        "You're very welcome!",
        "No problem at all!",
        "Happy to help anytime!",
        "Glad I could assist!",
        "My pleasure!",
        "Always here for you!",
        "You're welcome! Let me know if you need anything else.",
        "Anytime! I'm here to help.",
        "No problem! I'm glad I could assist.",
        "Glad to help! If you have more questions, just ask.",
    ],
    "feeling_sad": [
        "I'm sorry to hear that. Want to talk about it?",
        "That sounds tough. I'm here if you want to chat.",
        "It's okay to feel down sometimes. You're not alone.",
        "Sending good vibes your way.",
        "I hope things get better soon.",
        "Take your time. I'm here to listen.",
        "It's okay to feel sad. I'm here for you.",
        "I'm sorry you're feeling this way. Want to share more?",
        "It's okay to feel sad sometimes. I'm here to listen.",
        "I understand. It's tough to feel this way. What's on your mind?",
    ],
    "feeling_angry": [
        "That sounds frustrating. Want to vent?",
        "I get why you’re mad. Sometimes things suck.",
        "Try to take a deep breath. It helps!",
        "I'm here if you want to talk about what’s bugging you.",
        "Anger is valid, but don’t let it consume you.",
        "Let’s try to find a way to make it better.",
        "I understand why you're upset. Want to talk about it?",
        "It's okay to feel angry. I'm here to listen if you want to share.",
        "I totally get it. Want to talk about what's making you mad?",
        "I understand. Anger can be tough to deal with. Want to share more?",
    ],
    "feeling_worried": [
        "I understand. Worrying can be exhausting.",
        "Try to focus on what you can control.",
        "I'm here for you — we can figure this out.",
        "It’s okay to be anxious sometimes. You got this.",
        "Taking a break might help clear your mind.",
        "Remember to breathe. One step at a time.",
        "I understand why you're worried. Want to talk about it?",
        "It's okay to feel anxious. I'm here to listen if you want to share.",
        "I get it. Worrying can be tough. Want to share what's on your mind?",
        "I understand. Worrying can be overwhelming. Want to talk about it?",
    ],
    "feeling_happy": [
        "Awesome! I’m glad to hear that!",
        "That’s fantastic news!",
        "Keep that positive energy going!",
        "You deserve to feel great!",
        "I love hearing that!",
        "Smile big — you earned it!",
        "That's wonderful! I'm so happy for you!",
        "Great to hear! Keep that positivity flowing!",
        "That's amazing! I'm glad you're feeling good!",
        "So happy to hear that! Keep shining!",
    ],
    "feeling_disgust": [
        "Yikes, that sounds unpleasant.",
        "I get why you feel that way.",
        "Sometimes things really gross us out.",
        "Try to focus on something positive!",
        "Thanks for sharing. That sounds rough.",
        "Let’s find a better topic to lighten the mood.",
        "I understand why that would disgust you. Want to talk about it?",
        "That sounds really unpleasant. I'm here if you want to share more.",
        "I get it. That can be really off-putting. Want to talk about it?",
    ],
    "feeling_fear": [
        "It’s okay to be scared sometimes.",
        "You’re stronger than you think.",
        "Take things one step at a time.",
        "I'm here to support you.",
        "Fear is natural, but don’t let it stop you.",
        "Let’s breathe through this together.",
        "I understand why you're scared. Want to talk about it?",
        "It's okay to be afraid. I'm here to listen if you want to share.",
        "I get it. Fear can be tough to deal with. Want to share what's on your mind?",
    ],
    "feeling_confused": [
        "No worries, I can help clarify if you want.",
        "Sometimes things don’t make sense right away.",
        "Let’s try to break it down together. Tell me what's up.",
        "Feel free to ask me anything!",
        "Confusion is just the first step to understanding.",
        "I’m here to help untangle things.",
        "I understand why you're confused. Want to talk about it?",
        "It's okay to feel confused. I'm here to listen if you want to share.",
        "I get it. Confusion can be frustrating. Want to share what's on your mind?",
        "I understand. Confusion is normal. Want to talk about it?",
    ],
    "feeling_tired": [
        "Sounds like you need some rest.",
        "Make sure to take care of yourself.",
        "A little break might do wonders!",
        "Listen to your body — it knows best.",
        "Try to relax and recharge.",
        "Rest up and come back stronger!",
    ],
    "feeling_calm": [
        "Glad you’re feeling peaceful!",
        "That’s a great place to be.",
        "Enjoy the calm while it lasts.",
        "Stay relaxed and positive!",
        "Peace is priceless.",
        "Keep that chill vibe going.",
    ],
    "feeling_neutral": [
        "Thanks for sharing how you feel.",
        "Got it, let me know if you want to talk more.",
        "Sometimes neutral is just fine!",
        "Let me know if I can help brighten your day.",
        "Here whenever you want to chat.",
        "Okay, I’m listening!",
    ],
    "help_request": [
        "Of course! What do you need help with?",
        "I'm here to assist. What can I do?",
        "Happy to help! Just ask away.",
        "Sure thing! How can I support you today?",
        "Tell me what you need, and I’ll do my best.",
        "Let me know how I can make things easier.",
    ],
    "time_query": [
        "I don't have a clock on me, but you can check your device!",
        "Time’s ticking! Check your watch or phone.",
        "I’m not linked to real-time, sorry!",
        "You might want to glance at your clock for the current time.",
        "I recommend looking at a clock nearby.",
        "Wish I could tell you! Try your phone’s clock.",
    ],
    "capabilities_query": [
        "I can chat with you, respond to your feelings, and keep you company!",
        "I'm here to talk, listen, and help where I can.",
        "I can understand your intent and respond accordingly.",
        "Let’s chat about whatever’s on your mind!",
        "I’m an evolving chatbot — learning more every day.",
        "I’m designed to understand and respond to your inputs.",
    ],
    "unknown": [
        "Hmm, I’m not quite sure how to respond to that.",
        "Can you try rephrasing?",
        "I didn’t quite catch that. Could you say it differently?",
        "Sorry, I’m still learning. Can you explain more?",
        "I’m not sure what you mean. Could you clarify?",
        "That’s new to me — want to try saying it another way?",
    ]
}

if __name__ == "__main__":
    bot = IntelligentChatBot(training_data, responses)
    bot.chat()
