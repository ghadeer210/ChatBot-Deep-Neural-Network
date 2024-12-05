import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import nltk
# nltk.download('punkt')
# nltk.download('wordnet') #lexical database for the English language
# nltk.download('omw-1.4')
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

import numpy as np
import json
import random
import tensorflow as tf
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import SGD
import pickle

from time import sleep
seed = 42
np.random.seed(seed=seed)
tf.random.set_seed(seed=seed)

import warnings
warnings.filterwarnings('ignore')

with open(r'C:\Users\Ghadeer\Desktop\github like\ChatBot-Deep-Neural-Network\dataset\intents.json') as file:
    intents = json.load(file)

try:
    with open(r'C:\Users\Ghadeer\Desktop\github like\ChatBot-Deep-Neural-Network\model\data.pickle', 'rb') as f:
        words, classes, train_x, train_y = pickle.load(f)
except:      
    words=[]
    classes = []
    documents = []
    ignore_words = ['?', '!']
    
    
    for intent in intents['intents']:
        for pattern in intent['patterns']:
            #tokenize each word
            w = nltk.word_tokenize(pattern)
            words.extend(w)# add each elements into list
            #combination between patterns and intents
            documents.append((w, intent['tag']))#add single element into end of list
            # add to tag in our classes list
            if intent['tag'] not in classes:
                classes.append(intent['tag'])
    
    # lemmatize, lower each word and remove duplicates
    words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
    words = sorted(list(set(words)))
    # sort classes
    classes = sorted(list(set(classes)))
    
    # create our training data
    training = []
    # create an empty array for our output
    output_empty = [0] * len(classes)
    # training set, bag of words for each sentence
    for doc in documents:
        # initialize our bag of words
        bag = []
        # list of tokenized words
        pattern_words = doc[0]
        # convert pattern_words in lower case
        pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]
        # create bag of words array,if word match found in current pattern then put 1 otherwise 0.[row * colm(263)]
        for w in words:
            bag.append(1) if w in pattern_words else bag.append(0)
        
        # in output array 0 value for each tag ang 1 value for matched tag.[row * colm(8)]
        output_row = list(output_empty)
        output_row[classes.index(doc[1])] = 1
        
        training.append([bag, output_row])
    
    # shuffle training and turn into a list of lists
    random.shuffle(training)
    training = list(training)  # Keep it as a list of lists for flexibility
    
    # Separate the input (X) and output (Y)
    train_x = np.array([np.array(item[0]) for item in training])
    train_y = np.array([np.array(item[1]) for item in training])
    print("Training data created")
    
    with open(r'C:\Users\Ghadeer\Desktop\github like\ChatBot-Deep-Neural-Network\model\data.pickle', 'wb') as f:
        pickle.dump((words, classes, train_x, train_y), f)
        
tf.keras.backend.clear_session()
    
    # Create model - 3 layers. First layer 128 neurons, second layer 64 neurons and 3rd output layer contains number of neurons
# equal to number of intents to predict output intent with softmax
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))
# print("First layer:",model.layers[0].get_weights()[0])
# Compile model. Stochastic gradient descent with Nesterov accelerated gradient gives good results for this model
# sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


try:
    model = load_model(r'C:\Users\Ghadeer\Desktop\github like\ChatBot-Deep-Neural-Network\model\chatbot_model.h5')
    print("model loaded")
except:
    #fitting and saving the model 
    hist = model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)
    model.save(r'C:\Users\Ghadeer\Desktop\github like\ChatBot-Deep-Neural-Network\model\chatbot_model.h5', hist)
    
    print("model created")
    
    
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
