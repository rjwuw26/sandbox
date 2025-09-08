# AI Early Learning Projects

This folder contains three mini chatbot projects created as part of early AI experimentation and learning.  
These projects are designed to explore basic conversational AI concepts, embeddings, and simple intent recognition.

## Projects

1. **chatbot_1**  
   - A simple chatbot using logistic regression (or basic learning techniques).  
   - Core goal: Practice handling user input and generating responses.

2. **chatbot_2**  
   - An experimental chatbot exploring interactive responses.  
   - Core goal: Learn basic conversational flow and structure.

3. **chatbot_3 (IntelligentChatBot v1.2)**  
   - Uses sentence embeddings (`SentenceTransformer`) and cosine similarity for intent prediction.  
   - Core goal: Explore embeddings and context-aware responses.

## Python Version & Dependencies

- These projects were developed using **Python 3.10**.  
- Required packages include:
  - `scikit-learn` (for LogisticRegression, CountVectorizer, TfidfVectorizer, cosine_similarity)
  - `sentence-transformers` (for Project 3 embeddings)
  - `numpy` (installed automatically with scikit-learn)
  
> **Disclaimer:** For best results, run these projects in Python 3.10 where the required packages are installed. Future projects may use Python 3.13 or newer.

## How to Run

1. Navigate to the folder of the project you want to run (`chatbot_1`, `chatbot_2`, or `chatbot_3`).  
2. Install dependencies (in terminal):
```bash
pip install scikit-learn sentence-transformers
