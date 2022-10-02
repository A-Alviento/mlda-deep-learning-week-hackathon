from wordcloud import WordCloud
import gensim
import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import pickle
import preprocess_kgptalkie as ps
import re
import seaborn as sns

# tokenizer used to tokenize data
from tensorflow.keras.preprocessing.text import Tokenizer
# pad sequence used to pad datasets which are not long enough, need to pad for constant len input
from tensorflow.keras.preprocessing.sequence import pad_sequences
# sequential model to feeding our model layers
from tensorflow.keras.models import Sequential
# layers
from tensorflow.keras.layers import Dense, Embedding, LSTM, Conv1D, MaxPool1D
# train test split is to split data sets into training set and test set
from sklearn.model_selection import train_test_split
# accuracy report and accuracy score used to test model performance
from sklearn.metrics import classification_report, accuracy_score

# read spam dataset into a frame data
data = pd.read_csv("./data/spam.csv", encoding = "ISO-8859-1") 

## Generating word cloud
# make two arrays, one to indicate spam and another to store the sms
spam_arr = data['v1'].tolist()
data_list = data['v2'].tolist()

# separate the data set into spam and ham
spam = []
ham = []

for i in range(len(spam_arr)):
  if spam_arr[i] == 'ham':
    ham.append(data_list[i])
  else:
    spam.append(data_list[i])

text = ' '.join(spam)
# now we can use wordcloud
wordcloud = WordCloud(width=1920, height=1080).generate(text)
wordcloud.to_file("./data/spam.png")
text = ' '.join(ham)
wordcloud = WordCloud(width=1920, height=1080).generate(text)
wordcloud.to_file("./data/ham.png")

# remove special characters
data['v2'] = data['v2'].apply(lambda x: ps.remove_special_chars(x))

# we need to label the spam and the ham
list_of_cols_to_change = ['v1']
data["v1"] = data["v1"].apply(lambda x: 1 if x == "spam" else 0)

# we only need v1 and v2, drop the rest
data = data[['v1', 'v2']]


# get array of class
y = data['v1'].values
# data['text'].tolist turns text data into a sequence of list, which we need to convert into a list of words
x = [i.split() for i in data['v2'].tolist()]

# each word is converted into a seq of 100 vectors
dim = 100
# creat a gensim model
# sentences are the list of the list, x, the size is dim, window shows us how many words are connected together, min_count means even if there is only 1 word it generates a vecor for that
w2v = gensim.models.Word2Vec(sentences=x, vector_size=dim, window=10, min_count=1)

# now our text is converted into vectors
# we can feed these vectors as initial weight in machine learning model, and then use the machine learning to recreate this weights again

# create tokenizer 
tokenizer = Tokenizer()
tokenizer.fit_on_texts(x)

pickle.dump(tokenizer, open("./data/tokenizer.pickle", "wb"))

# after tokenisation, text data is converted into a set of sequence
x = tokenizer.texts_to_sequences(x)

# we create a threshold of word limit
maxlen = 100
# when sequence is more than 100, truncate, if less than 100 pad it
x = pad_sequences(x, maxlen=maxlen)

# we put +1 as there are many words that may not come in the tokens
# for those words, we consider those as unknown words which creates another sequence
vocab_len = len(tokenizer.word_index) + 1
# voc takes on the tokenizer.word_index which represents the word to index conversion
voc = tokenizer.word_index

# feed vectors as the initial weight in the machine learning model and then we use the machine learning model to retrain this weight to find the maximum accuracy
def get_weight_matrix(model):
  # weight matrix for the words
  weight_matrix = np.zeros((vocab_len, dim))
  for word, i in voc.items():
    # assign the weight
    try:
        weight_matrix[i] = model.wv[word]
    except:
       pass 

  return weight_matrix

weight_matrix = get_weight_matrix(w2v)

# here we create our machine learning model; we create an instance of the Sequential Class
model = Sequential()

# model layers are subsequently added to it

# here we add an embedding layer - first arg is number of distinct words in training set; second arg is size of embedding vector (dim of vectorisation); 
# third arg is just our array of weights for the words; fourth arg is the size of each input sequence; last arg indicates if we want our weight_matrix
# to be trainable, i.e. to adapt according to the machine learning
model.add(Embedding(vocab_len, output_dim=dim, weights = [weight_matrix], input_length=maxlen, trainable=True))

# here we add a long short-term memory layer. lstm is a special version of RNN which solves short term memory problem
# unit refers to the dimensionality of the output space, which is the dimension of the hidden state vector a that is the
# state output from RNN cell
model.add(LSTM(units=128))

# adding a dense layer. 
# first arg represents the number of units and it affects the output layer; second arg represents the activation function
# here we are using sigmoid as the activation function and it guarantees that the output of this unit will always be between 0 and 1
model.add(Dense(1, activation='sigmoid'))

# compile the model since we already defined it
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])

# summary of model
model.summary()

# split data sets into training and testing sets
# recall x is the text and y is the class 
x_train, x_test, y_train, y_test = train_test_split(x,y)

# training
model.fit(x_train, y_train, validation_split=0.3, epochs=6)

test_loss, test_acc = model.evaluate(x_test, y_test)
print(test_loss, test_acc)

model.save("./data")