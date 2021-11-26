Exercise 2
In the course you learned how to do classificaiton using Fashion MNIST, a data set containing items of clothing. There's another, similar dataset called MNIST which has items of handwriting -- the digits 0 through 9.

Write an MNIST classifier that trains to 99% accuracy or above, and does it without a fixed number of epochs -- i.e. you should stop training once you reach that level of accuracy.

Some notes:

It should succeed in less than 10 epochs, so it is okay to change epochs= to 10, but nothing larger
When it reaches 99% or greater it should print out the string "Reached 99% accuracy so cancelling training!"
If you add any additional variables, make sure you use the same names as the ones used in the class
I've started the code for you below -- how would you finish it?

import tensorflow as tf
from os import path, getcwd, chdir
​
# DO NOT CHANGE THE LINE BELOW. If you are developing in a local
# environment, then grab mnist.npz from the Coursera Jupyter Notebook
# and place it inside a local folder and edit the path to that location
path = f"{getcwd()}/../tmp2/mnist.npz"
# GRADED FUNCTION: train_mnist
def train_mnist():
    # Please write your code only where you are indicated.
    # please do not remove # model fitting inline comments.
​
    # YOUR CODE SHOULD START HERE
    class myCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self,epoch,logs={}):
            if(logs.get('acc') > 0.99):
                print("Reached 99% accuracy so cancelling training!")
                self.model.stop_training = True
                
    callbacks = myCallback()
    # YOUR CODE SHOULD END HERE
    
    mnist = tf.keras.datasets.mnist
​
    (x_train, y_train),(x_test, y_test) = mnist.load_data(path=path)
    # YOUR CODE SHOULD START HERE
    x_train, x_test = x_train/255.0, x_test/255.0
    # YOUR CODE SHOULD END HERE
    model = tf.keras.models.Sequential([
        # YOUR CODE SHOULD START HERE
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(1024,activation=tf.nn.relu),
        tf.keras.layers.Dense(10,activation=tf.nn.softmax)
        # YOUR CODE SHOULD END HERE
    ])
​
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    # model fitting
    history = model.fit(# YOUR CODE SHOULD START HERE
        x_train,y_train,epochs=10,callbacks=[callbacks]
              # YOUR CODE SHOULD END HERE
    )
    # model fitting
    return history.epoch, history.history['acc'][-1]
train_mnist()
Epoch 1/10
60000/60000 [==============================] - 9s 150us/sample - loss: 0.1858 - acc: 0.9440
Epoch 2/10
60000/60000 [==============================] - 9s 142us/sample - loss: 0.0752 - acc: 0.9766
Epoch 3/10
60000/60000 [==============================] - 9s 147us/sample - loss: 0.0491 - acc: 0.9846
Epoch 4/10
60000/60000 [==============================] - 9s 143us/sample - loss: 0.0351 - acc: 0.9886
Epoch 5/10
59328/60000 [============================>.] - ETA: 0s - loss: 0.0263 - acc: 0.9910Reached 99% accuracy so cancelling training!
60000/60000 [==============================] - 9s 142us/sample - loss: 0.0265 - acc: 0.9910
([0, 1, 2, 3, 4], 0.9909833)
# Now click the 'Submit Assignment' button above.
# Once that is complete, please run the following two cells to save your work and close the notebook
%%javascript
<!-- Save the notebook -->
IPython.notebook.save_checkpoint();
%%javascript
IPython.notebook.session.delete();
window.onbeforeunload = null
setTimeout(function() { window.close(); }, 1000);
