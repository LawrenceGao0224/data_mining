import tensorflow as tf
import keras
import matplotlib.pyplot as plt
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers.core import Dense,Dropout,Activation,Flatten
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import SimpleRNN
from keras.layers.recurrent import LSTM
import numpy as np
filename = 'training_label.txt'

train_text1 = []
with open(filename,encoding = 'utf8') as file:
   for line in file:
      line = line.strip().split('+++$+++')
      train_text1.append(line)
train_text1 = np.array(train_text1)
train_text = train_text1[:,1]
y_train = train_text1[:,0]
filename2 = 'testing_label.txt'

test_text1 = []
with open(filename2,encoding = 'utf8') as file:
   for line2 in file:
      line2 = line2.strip().split('#####')
      if len(line2)!=1:
          test_text1.append(line2)
test_text1 = np.array(test_text1)
test_text = test_text1[:,1]
y_test = test_text1[:,0]
token = Tokenizer(num_words = 2000)
token.fit_on_texts(test_text)
x_train_seq = token.texts_to_sequences(train_text)
x_test_seq = token.texts_to_sequences(test_text)
x_train = sequence.pad_sequences(x_train_seq,maxlen = 40)
x_test = sequence.pad_sequences(x_test_seq,maxlen = 40)
model = Sequential()
model.add(Embedding(output_dim = 32,input_dim = 2000,input_length = 40))
#model.add(Dropout(0.7))
#model.add(Flatten())
model.add(Dense(units=256,activation = 'relu'))
#model.add(Dropout(0.7))
model.add(SimpleRNN(units =16))
#model.add(LSTM(units =32))
model.add(Dense(units=1,activation='sigmoid'))
model.compile(loss='binary_crossentropy',optimizer = 'adam',metrics=['accuracy'])
train_history = model.fit(x_train,y_train,batch_size = 100,epochs =10,verbose = 2,validation_split =0.2)
plt.plot(train_history.history['acc'])  
plt.plot(train_history.history['val_acc'])  
plt.title('Train History')  
plt.ylabel('acc')  
plt.xlabel('Epoch')  
plt.legend(['acc', 'val_acc'], loc='upper left')  
plt.show() 
scores = model.evaluate(x_test,y_test,verbose =1)