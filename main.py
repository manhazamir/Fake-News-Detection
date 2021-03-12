
!pip install plotly
!pip install --upgrade nbformat
!pip install nltk
!pip install spacy # spaCy is an open-source software library for advanced natural language processing
!pip install WordCloud
!pip install gensim # Gensim is an open-source library for unsupervised topic modeling and natural language processing
import nltk
nltk.download('punkt')

import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS
import nltk
import re
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from tensorflow.keras.preprocessing.text import one_hot, Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Embedding, Input, LSTM, Conv1D, MaxPool1D, Bidirectional
from tensorflow.keras.models import Model
from jupyterthemes import jtplot
jtplot.style(theme='monokai', context='notebook', ticks=True, grid=False)
# setting the style of the notebook to be monokai theme
# this line of code is important to ensure that we are able to see the x and y axes clearly
# If you don't run this code line, you will notice that the xlabel and ylabel on any plot is black on black and it will be hard to see them.


# load the data
df_true = pd.read_csv("True.csv")
df_fake = pd.read_csv("Fake.csv")


df_fake
df_true

df_true.isnull().sum()
df_fake.isnull().sum()

df_true.info()
df_fake.info()

# add a target class column to indicate whether the news is real or fake
df_true['isfake'] = 1
df_true.head()

df_fake['isfake'] = 0
df_fake.head()

# Concatenate Real and Fake News
df = pd.concat([df_true, df_fake]).reset_index(drop = True)
df

#reset_index is used to index the concatenated rows from 0 to (sum of df_true+df_fake)
#concatenated into one dataframe to see true and fake results in one


#drop redundant column

df.drop(columns = ['date'], inplace ='TRUE')

#inplace = FALSE only drops the column here in notebook (code)
#inplace = TRUE will drop the column in memory as well (Permanent change)

#combining the columns; title and text to consider it as 1 column in df

df['original'] = df['title'] + ' ' + df['text']

df['original'][0]  #original column x first row result

#Now we need to clean our data i.e. remove stop words from the data


#--------TASK 4 -----DATA CLEANING----------

nltk.download("stopwords")   #download package stopwords

#import stopwords from nltk (NLP)

# Obtain additional stopwords from nltk
from nltk.corpus import stopwords
stop_words = stopwords.words('english')   #extract english text stopwords
stop_words.extend(['from', 'subject', 're', 'edu', 'use'])

#We can extend by adding some words to the stopwords we want to drop from our text


stop_words #Results all the stopwords


# Remove stopwords and remove words with 2 or less characters
def preprocess(text):  #function called preprocess with column text
    result = []
    for token in gensim.utils.simple_preprocess(text):  #gensim is a library used for nlp
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3 and token not in stop_words:
            result.append(token)

    return result

#Starting result as 0, check for a token (variable) if its length is >3 and it is not in the stopwords, then append it to my results


# Apply the function to the dataframe
df['clean'] = df['original'].apply(preprocess)  #Adds new column in df that is without the stopwords

df['original'][0]  #Check the original text for comparison

print(df['clean'][0])   #Print the clear column x first row

#This gives us the unique data that we need to feed to our LSTM RNN by converting them to numbers (tokenization)

# Obtain the total words present in the dataset
list_of_words = []
for i in df.clean:
    for j in i:
        list_of_words.append(j)


list_of_words  #show words in dfclean in entire dataframe

len(list_of_words)  #9276947

# Obtain the total number of unique words
total_words = len(list(set(list_of_words)))   #Applying 'set' gives unique words only
total_words        #total unique words = 108704

# join the words into a string
df['clean_joined'] = df['clean'].apply(lambda x: " ".join(x))  #Join all words separated by space

df

df['clean_joined'][0]

#-----------TASK-5-------VISUALIZE DATA-------

# plot the number of samples in 'subject'
plt.figure(figsize = (8, 8))
sns.countplot(y = "subject", data = df)  #Seaborn library to visualize the column subject

#y axis has subject column and data is on the x axis

#Print out WordCloud; WordCloud is actually really powerful visualization specifically for text data

#shows importance of each word in text (The keywords)

# plot the word cloud for text that is Fake
plt.figure(figsize = (20,20))
wc = WordCloud(max_words = 2000 , width = 1600 , height = 800 , stopwords = stop_words).generate(" ".join(df[df.isfake == 0].clean_joined))
plt.imshow(wc, interpolation = 'bilinear')

#maxwords is the maximum words you want to print out on screen
# plot the word cloud for text that is Real
plt.figure(figsize = (20,20))
wc = WordCloud(max_words = 2000 , width = 1600 , height = 800 , stopwords = stop_words).generate(" ".join(df[df.isfake == 1].clean_joined))
plt.imshow(wc, interpolation = 'bilinear')

# length of maximum document will be needed to create word embeddings
maxlen = -1
for doc in df.clean_joined:
    tokens = nltk.word_tokenize(doc)
    if(maxlen<len(tokens)):
        maxlen = len(tokens)
print("The maximum number of words in any document is =", maxlen)

#nltk.word_tokenize(df['clean_joined'][0])  would tokenize each word in a string eg ['hi', 'hello', 'new']

# visualize the distribution of number of words in a text
import plotly.express as px
fig = px.histogram(x = [len(nltk.word_tokenize(x)) for x in df.clean_joined], nbins = 100)
fig.show()

#-------TASK 6------- PREPARE DATA USING TOKENIZATION AND PADDING


#Split data into train and test using sklearn

# split data into test and train
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(df.clean_joined, df.isfake, test_size = 0.2)

#Parameters include input data as df.clean joined and output as df.isfake, the test size is 20% here meaning
#testing data = 20% and training data = 80%

from nltk import word_tokenize

# Create a tokenizer to tokenize the words and create sequences of tokenized words
tokenizer = Tokenizer(num_words = total_words)
tokenizer.fit_on_texts(x_train)  #Stack overflow - Vocabulary
train_sequences = tokenizer.texts_to_sequences(x_train)  #Transformation here to integers
test_sequences = tokenizer.texts_to_sequences(x_test)

len(train_sequences)

print("The encoding for document\n",df.clean_joined[0],"\n is : ",train_sequences[0])

#All the different news should have the same length, for that we use padding

# Add padding can either be maxlen = 4406 or smaller number maxlen = 40 seems to work well based on results
padded_train = pad_sequences(train_sequences,maxlen = 40, padding = 'post', truncating = 'post')
padded_test = pad_sequences(test_sequences,maxlen = 40, truncating = 'post')

#can take 4406 too
#Zeros are added at the end of each sample to make all samples of same length
#for loop to check padded sequences in first two rows
for i,doc in enumerate(padded_train[:2]):
     print("The padded encoding for document",i+1," is : ",doc)



# Sequential Model
model = Sequential()

# embeddidng layer
model.add(Embedding(total_words, output_dim = 128))   #can change the output dimension
# model.add(Embedding(total_words, output_dim = 240))


# Bi-Directional RNN and LSTM
model.add(Bidirectional(LSTM(128)))  #can change the number of nuerons to add

# Dense layers
model.add(Dense(128, activation = 'relu'))  #128 layers and relu activation
model.add(Dense(1,activation= 'sigmoid'))  #sigmoid layer, 1 neuron (output)  which represents 0 or 1
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])  #metrics = accuracy
model.summary()

#check .compile etc functions

y_train = np.asarray(y_train) # converting to array, imp step before feeding along the model

# train the model
model.fit(padded_train, y_train, batch_size = 64, validation_split = 0.1, epochs = 2)

#input = padded_train, output = (array) y_train
#validation spilt = 10% for cross validation and 90 for training the model (i.e. splitting training data)
#we apply cross validation to check the model is not over fitting the data. After every epoch, the 10%..
#..will run through the model and see if error of validation is going down or not. If down, then good.
#If the error on training and validation is going down , then good
#If the error on training data is going down but the error on validation is going up, then over fitting, hence fix

#Result for two epoch after running
#Train on 32326 samples, validate on 3592 samples
#Epoch 1/2
#32326/32326 [==============================] - 321s 10ms/sample - loss: 0.0421 - acc: 0.9815 - val_loss: 0.0073 - val_acc: 0.9992
#Epoch 2/2
#32326/32326 [==============================] - 316s 10ms/sample - loss: 0.0016 - acc: 0.9997 - val_loss: 0.0096 - val_acc: 0.9981

#Here above, the result shows accuracy of 99%


#----------TASK 9---------ASSESS TRAINED MODEL  (On testing data)

# make prediction
pred = model.predict(padded_test)  #feed testing data

# if the predicted value is >0.5 it is real else it is fake
#Applying threshold for our output (which in our case is a sigmoid function i.e (i.e. either 0 or 1))

prediction = []
for i in range(len(pred)):
    if pred[i].item() > 0.5:   #So, here threshold = 0.5
        prediction.append(1)  #put into class real
    else:
        prediction.append(0)

# getting the accuracy
#model predictions vs the actual
from sklearn.metrics import accuracy_score

accuracy = accuracy_score(list(y_test), prediction)

print("Model Accuracy : ", accuracy)   # we get 0.99

# get the confusion matrix
#i.e. visualising the predictions vs actual
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(list(y_test), prediction)
plt.figure(figsize = (25, 25))
sns.heatmap(cm, annot = True)

#this tells us the misclassified samples visually

# category dict
category = { 0: 'Fake News', 1 : "Real News"}  #final labels

