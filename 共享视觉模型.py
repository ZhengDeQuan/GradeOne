#!/usr/bin/python
# -*- coding:UTF-8 -*-

import keras
from keras.layers import Conv2D, MaxPooling2D, Input, Dense, Flatten
from keras.models import Model

# First, define the vision modules
digit_input = Input(shape=( 27, 27 ,1))
x = Conv2D(64, (3, 3))(digit_input)
x = Conv2D(64, (3, 3))(x)
x = MaxPooling2D((2, 2))(x)
out = Flatten()(x)

vision_model = Model(digit_input, out)

# Then define the tell-digits-apart model
digit_a = Input(shape=(27, 27 , 1))
digit_b = Input(shape=(27, 27 , 1))

# The vision model will be shared, weights and all
out_a = vision_model(digit_a)
out_b = vision_model(digit_b)

concatenated = keras.layers.concatenate([out_a, out_b])
out = Dense(1, activation='sigmoid')(concatenated)
print("out = ",out)
print("con = ",concatenated)
#classification_model = Model([digit_a, digit_b], out)
model = Model([digit_a,digit_b],out)
model.compile(optimizer='sgd',loss='binary_crossentropy')
import numpy as np
a = np.array(range(27*27*3)).reshape(3,27,27,1)
b = np.array(range(27*27*3)).reshape(3,27,27,1)
c = np.array([1,0,1])
model.fit(x=[a,b],y=c)
model.predict([a,b])