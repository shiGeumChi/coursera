In this exercise you'll try to build a neural network that predicts the price of a house according to a simple formula.

So, imagine if house pricing was as easy as a house costs 50k + 50k per bedroom, so that a 1 bedroom house costs 100k, a 2 bedroom house costs 150k etc.

How would you create a neural network that learns this relationship so that it would predict a 7 bedroom house as costing close to 400k etc.

Hint: Your network might work better if you scale the house price down. You don't have to give the answer 400...it might be better to create something that predicts the number 4, and then your answer is in the 'hundreds of thousands' etc.

import tensorflow as tf
import numpy as np
from tensorflow import keras
# GRADED FUNCTION: house_model
def house_model(y_new):
    xs = np.array([1.0,2.0,3.0,4.0,5.0,6.0],dtype=float) # Your Code Here#
    ys = np.array([1.0,1.5,2.0,2.5,3.0,3.5],dtype=float)# Your Code Here#
    model = keras.Sequential([keras.layers.Dense(units=1,input_shape=[1])])# Your Code Here#
    model.compile(optimizer='sgd', loss = 'mean_squared_error')# Your Code Here#
    model.fit(xs,ys,epochs=500)# Your Code here#
    return model.predict(y_new)[0]
prediction = house_model([7.0])
print(prediction)
Epoch 1/500
6/6 [==============================] - 0s 28ms/sample - loss: 0.0651
Epoch 2/500
6/6 [==============================] - 0s 286us/sample - loss: 0.0548
Epoch 3/500
6/6 [==============================] - 0s 226us/sample - loss: 0.0499
........
Epoch 497/500
6/6 [==============================] - 0s 212us/sample - loss: 0.0012
Epoch 498/500
6/6 [==============================] - 0s 226us/sample - loss: 0.0012
Epoch 499/500
6/6 [==============================] - 0s 254us/sample - loss: 0.0012
Epoch 500/500
6/6 [==============================] - 0s 263us/sample - loss: 0.0012
[4.0502462]
# Now click the 'Submit Assignment' button above.
# Once that is complete, please run the following two cells to save your work and close the notebook
%%javascript
<!-- Save the notebook -->
IPython.notebook.save_checkpoint();
%%javascript
IPython.notebook.session.delete();
window.onbeforeunload = null
setTimeout(function() { window.close(); }, 1000);
