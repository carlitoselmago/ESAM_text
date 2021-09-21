# IMPORT LIBRARIES ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
import pandas as pd
import numpy as np
import re
import sys
import csv
import os.path
import time
import os

import tensorflow as tf
from tensorflow import keras
from nltk import word_tokenize
from keras.models import Sequential, Model
from keras.layers.embeddings import Embedding
from keras.models import model_from_json
from keras.layers import Input, Activation, Dense, Dropout
from keras.layers import LSTM, Bidirectional, GRU

import nltk
nltk.download('punkt')

dataCached=False

csv_file = "quotes.csv"
savedmodel_file="model"

#TWITTER API :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

if not os.path.isfile(csv_file):

    import tweepy

    #API USER
    consumer_key='sqCZ4EKznfmKwxKw5y9DLexFD'
    consumer_secret='tIRaumL8cHX3R6d3CRAHxWePIfENfl87Uj0mn570mUTOp2zQPv'
    access_token='1436664768425758720-oaT9rdnSWPmddZVR2vFUifUUIVb17i'
    access_token_secret='UkiFinhCS152lwbsyOQeRipwkdWHH2VQCmUF43PZ3COdJ'

    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)

    api = tweepy.API(auth)

    ### TWEET SETTINGS #########################
    twitterUsers=[
      'GreatestQuotes',
      'TooPositiveMind',
      'ip_quotes',

    ]
    minimumLength=40
    maxTweets=5000 #per user
    ###############################################

    rawquotes=[]
    maxPrint=20
    for u,username in enumerate(twitterUsers):
      print("")
      print("Extracting data from @",username,":::::::::::")
      print("")
      rawquotes.append([])
      for i,status in enumerate(tweepy.Cursor(api.user_timeline, screen_name='@'+username, tweet_mode="extended").items()):
        tweet=status.full_text
        if len(tweet)>minimumLength:
          if not status.retweeted and ('RT @' not in tweet):
            if i<maxTweets:
              rawquotes[u].append(tweet + '\n')
              #quotes.append(tweet)
              if i<maxPrint:
                print(tweet)
         #time.sleep(0.1)
      print("user total tweets ",len(rawquotes[u]))
      time.sleep(2)


#CLEAN UP DATA :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

if not os.path.isfile(csv_file):

    allquotes=[]

    for u,quotes in enumerate(rawquotes):
      print("")
      print("quotes #",(u+1),":::::::::::::::::::::::::::::::::::::")
      print("")
      for i,_ in enumerate(quotes):

        #remove author
        if "- " in quotes[i]:
          qparts=quotes[i].split("- ")
          quotes[i]=qparts[0]

        if "~" in quotes[i]:
          qparts=quotes[i].split("~")
          quotes[i]=qparts[0]

        #remove urls
        quotes[i]=re.sub(r"http\S+", "", quotes[i])

        #remove hashtags
        quotes[i] = re.sub("#[A-Za-z0-9_]+","", quotes[i])

        #remove user mentions
        quotes[i] = re.sub("@[A-Za-z0-9_]+","", quotes[i])

        #remove unwanted characters
        removeList=['"','#', '$', '%', '(', ')', '=', ';' ,':',  '*', '+', '£' , '—','’']

        for r in removeList:
          if r in quotes[i]:
            quotes[i]=quotes[i].replace(r, '')

        #remove whitespace
        quotes[i]=quotes[i].strip()

        #if quotes are longer than 10 characters
        if len(quotes[i])>10:

          if i<maxPrint:
            print(quotes[i])

          allquotes.append(quotes[i])

else:
    #load data from csv file
    csv.field_size_limit(100000000)
    with open(csv_file, 'r',encoding='utf-8', newline='\r\n') as f:
        reader = csv.reader(f)
        allquotes = list(reader)[0]
        dataCached=True


#save data to csv file
if (dataCached==False):
    try:
        with open(csv_file, 'w',newline="",encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerows([allquotes])
    except IOError:
        print("I/O error")

print("")
print("all clean quotes total:",len(allquotes),":::")

#FORMAT X (break up texts into pieces) :::::::::::::::::::::::::::::::::::::::::
quotes_cleaned=allquotes

text = ' '.join(quotes_cleaned)
chars = sorted(list(set(text)))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

maxlen = 17
step = 6
sentences = []
next_chars = []

for quote in quotes_cleaned:
  for i in range(0, len(quote) - maxlen, step):
    try:
      sentences.append(quote[i: i + maxlen])
      next_chars.append(quote[i + maxlen])
      sentences.append(quote[-maxlen:])
      next_chars.append(quote[-1])
    except:
      pass
print('nb sequences:', len(sentences))
print(sentences[:20])

#VECTORIZE (convert text to numbers) :::::::::::::::::::::::::::::::::::::::::::
print('Vectorization...')
x = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        x[i, t, char_indices[char]] = 1
    y[i, char_indices[next_chars[i]]] = 1

np.seterr(divide = 'ignore')


# DEFINE GENERATOR :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
two_first_words = [bigram for bigram in [' '.join(word_tokenize(quote)[:2]) for quote in allquotes] if len(bigram) <= maxlen]

def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

def generate_quote(sentence = None, temperature = 0.8):

    if not sentence: ## if input is null then sample two first word from dataset
        random_index = np.random.randint(0, len(two_first_words))
        sentence = two_first_words[random_index]

    if len(sentence) > maxlen:
        sentence = sentence[-maxlen:]
    elif len(sentence) < maxlen:
        sentence = ' '*(maxlen - len(sentence)) + sentence

    generated = ''
    generated += sentence
    predicted=generated
    next_char = generated
    total_word = 0
    max_word = 15

    while ((next_char not in ['\n', '.']) & (total_word <= 500)):
        x_pred = np.zeros((1, maxlen, len(chars)))
        for t, char in enumerate(sentence):
            x_pred[0, t, char_indices[char]] = 1.

        preds = model.predict(x_pred, verbose=0)[0]
        next_index = sample(preds, temperature)
        next_char = indices_char[next_index]

        if next_char == ' ':
           total_word += 1
        generated += next_char
        sentence = sentence[1:] + next_char
        predicted+=next_char

    return predicted

# define your custom callback for prediction
class PredictionCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    y_pred = generate_quote("Life is")

    print(' ')
    print(' ')
    print(y_pred)

#THE NEURAL NET MODEL:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
if os.path.isfile(savedmodel_file+".h5"):
    #model = keras.models.load_model(savedmodel_file)
    json_file = open(savedmodel_file+'.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    # load weights into new model
    model.load_weights(savedmodel_file+".h5")
    print("Loaded model from disk")
else:

    input_sequences = Input((maxlen, len(chars)) , name="input_sequences")
    lstm = Bidirectional(GRU(200, return_sequences= True, input_shape=(maxlen, len(chars))), name = 'bidirectional')(input_sequences)
    lstm = Dropout(0.1, name = 'dropout_bidirectional_lstm')(lstm)
    lstm = GRU(64, input_shape=(maxlen, len(chars)), name = 'lstm')(lstm)
    lstm = Dropout(0.1,  name = 'drop_out_lstm')(lstm)

    dense = Dense(15 * len(chars), name = 'first_dense')(lstm)
    dense = Dropout(0.1,  name = 'drop_out_first_dense')(dense)
    dense = Dense(5 * len(chars), name = 'second_dense')(dense)
    dense = Dropout(0.1,  name = 'drop_out_second_dense')(dense)
    dense = Dense(len(chars), name = 'last_dense')(dense)

    next_char = Activation('softmax', name = 'activation')(dense)

    model = Model([input_sequences], next_char)
    model.compile(optimizer='adam', loss='categorical_crossentropy')

    model.summary()

    #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

    #TRAIN MODEL, target loss is something around 0.02 less is worse beacause overfitting :::::::::::::::::::::::::::::::::::::::::::::::::::::::
    # TIP: if you don't reach something below 0.9 in epoch 100 your train data is probably too small
    model.fit([x], y,
             batch_size=100,
              epochs= 100,
              callbacks=[PredictionCallback()]

             )

#GENERATE QUOTE ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
print (generate_quote("People are ",0.8))

# SAVE MODEL (to reuse later) ::::::::::::::::::::::::::::::::::::::::::::::::::
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")
