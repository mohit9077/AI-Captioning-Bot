#!/usr/bin/env python
# coding: utf-8

# In[51]:


import json
import cv2
import matplotlib.pyplot as plt
import re
import collections
import pandas as pd
import numpy as np
from keras.applications.resnet50 import ResNet50,preprocess_input,decode_predictions
from keras.models import Model,load_model
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.utils import to_categorical
from keras.layers import Input,Dense,Dropout,Embedding,LSTM
from keras.layers.merge import add
from time import time
import pickle
import string
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Input,Dense,LSTM,Dropout,Embedding
import pickle
import tensorflow as tf
import keras

# In[52]:
config = tf.ConfigProto(
    device_count={'GPU': 1},
    intra_op_parallelism_threads=1,
    allow_soft_placement=True
)

config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.6

session = tf.Session(config=config)

keras.backend.set_session(session)




# In[61]:


model=load_model("./model_weights-20200508T020532Z-001/model_weights/model_9.h5")
model._make_predict_function()

# In[62]:


model_temp=ResNet50(weights="imagenet",input_shape=(224,224,3))


# In[63]:


model_resnet=Model(model_temp.input,model_temp.layers[-2].output)
model_resnet._make_predict_function()

# In[64]:


def preprocess_image(img):
    img= image.load_img(img,target_size=(224,224))
    img= image.img_to_array(img)    ### when you feed an image to resnet you can not feed single image you have to feed by batch
    #print(img.shape)                              #### which creates a 4d tensor   
    img= np.expand_dims(img,axis=0)  ## it creates 4d tensor
    
    ### normalisation
    img=preprocess_input(img)  ## preprocessing done by keras by subtracting mean (see documentation)
    
    return img


# In[65]:


def encode_img(img):

    try:
        with session.as_default():
            with session.graph.as_default():
                img=preprocess_image(img)
                feature_vector=model_resnet.predict(img,verbose=1)
                feature_vector=feature_vector.reshape(1,feature_vector.shape[1])
                #print(feature_vector.shape)
                return feature_vector

    except Exception as ex:
        log.log('Seatbelt Prediction Error', ex, ex.__traceback__.tb_lineno)

    


# In[66]:


with open('./storage/word_to_idx.pkl','rb') as w2i:
    word_to_idx=pickle.load(w2i)
    
with open('./storage/idx_to_word.pkl','rb') as i2w:
    idx_to_word=pickle.load(i2w)


def predict_caption(photo):
    
    in_text = "startseq"
    maxlen=35
    for i in range(maxlen):
        sequence = [word_to_idx[w] for w in in_text.split() if w in word_to_idx]
        sequence = pad_sequences([sequence],maxlen=maxlen,padding='post')
        try:

            with session.as_default():
                with session.graph.as_default():

                    ypred = model.predict([photo,sequence])
                    ypred = ypred.argmax() #WOrd with max prob always - Greedy Sampling
                    word = idx_to_word[ypred]
                    in_text += (' ' + word)
                    
                    if word == "endseq":
                        break
        except Exception as ex:
            log.log('Seatbelt Prediction Error', ex, ex.__traceback__.tb_lineno)    
    
    final_caption = in_text.split()[1:-1]
    final_caption = ' '.join(final_caption)
    return final_caption


# In[67]:

def caption_this_image(image):
    enc=encode_img(image)
    caption=predict_caption(enc)
    
    return caption






