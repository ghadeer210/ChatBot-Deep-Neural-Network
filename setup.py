import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import nltk
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

import numpy as np
import json
import random
import tensorflow as tf
from keras.models import load_model
import pickle

seed = 42
np.random.seed(seed=seed)
tf.random.set_seed(seed=seed)

try: 
    with open(r'C:\Users\Ghadeer\Desktop\github like\ChatBot-Deep-Neural-Network\dataset\intents.json') as file:
        intents = json.load(file)
        
    with open(r'C:\Users\Ghadeer\Desktop\github like\ChatBot-Deep-Neural-Network\model\data.pickle', 'rb') as f:
        words, classes, train_x, train_y = pickle.load(f)
        
    model = load_model(r'C:\Users\Ghadeer\Desktop\github like\ChatBot-Deep-Neural-Network\model\chatbot_model.h5')
except:
    print("Warning ------------>>> run file maim.py in path: '...\ChatBot-Deep-Neural-Network\code\main.py'")
    
def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [lemmatizer.lemmatize(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1
            
    return np.array(bag)


def chat():
    print("Hi, How can I help you?")
    while True:
        print("if end chat with ChatBot write 'quit or QUIT'")
        inp = input("You: ")
        if inp.lower() == "quit":
            print("Goodbye!")
            break

        # Generate bag-of-words representation and reshape it
        bow = bag_of_words(inp, words)
        bow = np.expand_dims(bow, axis=0)  # Reshape to (1, number_of_features)

        # Predict the intent
        results = model.predict(bow, verbose=0)[0]
        results_index = np.argmax(results)
        tag = classes[results_index]

        # Confidence threshold
        if results[results_index] > 0.8:
            for tg in intents["intents"]:
                if tg['tag'] == tag:
                    responses = tg['responses']
            print(random.choice(responses))
        else:
            print("I don't understand!")

chat()