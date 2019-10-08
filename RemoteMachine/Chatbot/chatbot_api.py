import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

from flask import Flask, jsonify, request
from flask_cors import CORS, cross_origin
import random
import pickle
import pandas as pd
import numpy as np
import tensorflow as tf
import threading 
import os 
import subprocess
# In[12]:


data = pickle.load( open( "chatbot-data.pkl", "rb" ) )
words = data['words']
classes = data['classes']


# In[13]:


def clean_up_sentence(sentence):
    # tokenize the pattern
    sentence_words = nltk.word_tokenize(sentence)
    # stem each word
    sentence_words = [stemmer.stem(word.lower()) for word in sentence_words]
    return sentence_words

# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence
def bow(sentence, words, show_details=True):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words
    bag = [0]*len(words)  
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s: 
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % w)

    return(np.array(bag))


# In[14]:


p = bow("Load bood pessure for patient", words)
print (p)
print (classes)


# In[15]:
import json
with open('intents.json') as json_data:
    intents = json.load(json_data)

# Use pickle to load in the pre-trained model
global graph
graph = tf.get_default_graph()

with open(f'chatbot-model.pkl', 'rb') as f:
    model = pickle.load(f)


# In[16]:


def classify_local(sentence):
    ERROR_THRESHOLD = 0.25
    
    # generate probabilities from the model
    input_data = pd.DataFrame([bow(sentence, words)], dtype=float, index=['input'])
    results = model.predict([input_data])[0]
    # filter out predictions below a threshold, and provide intent index
    results = [[i,r] for i,r in enumerate(results) if r>ERROR_THRESHOLD]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append((classes[r[0]], str(r[1])))
    # return tuple of intent and probability
    
    return return_list





# In[ ]:
def findmytest(inputtag):
  print("Processing...")
  print(inputtag)
  with open('Configration.json') as json_d:
         config = json.load(json_d)
  print(config)
  script=config[inputtag]
  path="E:/CodingPractice/ChatbotDrivenTesting/RemoteMachine/Tests/"+script
  subprocess.call(['python', path])
	
app = Flask(__name__)
CORS(app)

@app.route("/assistant", methods=['POST'])
def classify():
    ERROR_THRESHOLD = 0.25
    content = request.data
    content=str(content)
    content.replace("/b","")
    content.replace("\b","")
    print(content)
    print(type(content))
    sentence = content
    # generate probabilities from the model
    input_data = pd.DataFrame([bow(sentence, words)], dtype=float, index=['input'])
    results = model.predict([input_data])[0]
    # filter out predictions below a threshold
    results = [[i,r] for i,r in enumerate(results) if r>ERROR_THRESHOLD]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    # return tuple of intent and probability
    result=""
    #response = jsonify(return_list)
    #print(response)
     
    for val in intents['intents']:
       if str(val['tag']) in str(return_list):
          print(val['responses'])
          result_list=val['responses']
          result=result_list[random.randrange(0, len(result_list), 1)]
          if val['tag'] not in ["greeting","goodbye","thanks","noanswer","options"]:
             t1 = threading.Thread(target=findmytest,args=(val['tag'],), name='t1')
             t1.start()
          
	
    print(result)
    return result

# running REST interface, port=5000 for direct test, port=5001 for deployment from PM2
if __name__ == "__main__":
    app.run(port=5001)