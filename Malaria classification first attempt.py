#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import tensorflow as tf


# In[2]:


import tensorflow as tf


# In[3]:


print(tf.__version__)



# In[4]:


np.random.seed(1000)


# In[5]:


import numpy as np  # Import the NumPy library

# Now you can use np.random.seed(1000)
np.random.seed(1000)


# In[6]:


import matplotlib.pyplot as plt


# In[7]:


import os
import cv2


# In[8]:


import keras


# In[9]:


os.environ['KERAS_BACKEND'] = 'tensorflow' # Added to set the backend as Tensorflow
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
#We can also set it to Theano if we want.


# In[10]:


#Iterate through all images in Parasitized folder, resize to 64 x 64
#Then save as numpy array with name 'dataset'
#Set the label to this as 0
image_directory = 'C:\\Users\\Nayee\\OneDrive\\Desktop\\KIICE conference\\Malaria\\cell_images'

SIZE = 128
dataset = []  #Many ways to handle data, you can use pandas. Here, we are using a list format.  
label = []  #Place holders to define add labels. We will add 0 to all parasitized images and 1 to uninfected.


# In[11]:


import os

# Define the image directory with a separator at the end
image_directory = 'C:/Users/Nayee/OneDrive/Desktop/KIICE conference/Malaria/cell_images/'

# Now, list the files in the 'Parasitized' subdirectory
parasitized_images = os.listdir(image_directory + 'Parasitized/')

# Rest of your code...


# In[12]:


import os
import cv2
from PIL import Image  # Import the Image module from PIL

# Define the image directory with a separator at the end
image_directory = 'C:/Users/Nayee/OneDrive/Desktop/KIICE conference/Malaria/cell_images/'

# Now, list the files in the 'Parasitized' subdirectory
parasitized_images = os.listdir(image_directory + 'Parasitized/')

# Rest of your code...


# In[21]:


# import os
# import cv2
# from PIL import Image
# import numpy as np
# import matplotlib.pyplot as plt  # Import Matplotlib for plotting

# # Rest of the code...


# In[13]:


import numpy as np
import tensorflow as tf


# In[14]:


parasitized_images = os.listdir(image_directory + 'Parasitized/')
for i, image_name in enumerate(parasitized_images):    #Remember enumerate method adds a counter and returns the enumerate object
    
    if (image_name.split('.')[1] == 'png'):
        image = cv2.imread(image_directory + 'Parasitized/' + image_name)
        image = Image.fromarray(image, 'RGB')
        image = image.resize((SIZE, SIZE))
        dataset.append(np.array(image))
        label.append(0)

#Iterate through all images in Uninfected folder, resize to 64 x 64
#Then save into the same numpy array 'dataset' but with label 1

uninfected_images = os.listdir(image_directory + 'Uninfected/')
for i, image_name in enumerate(uninfected_images):
    if (image_name.split('.')[1] == 'png'):
        image = cv2.imread(image_directory + 'Uninfected/' + image_name)
        image = Image.fromarray(image, 'RGB')
        image = image.resize((SIZE, SIZE))
        dataset.append(np.array(image))
        label.append(1)


# In[15]:


from keras.layers import Convolution2D, MaxPooling2D, Flatten, Dense, BatchNormalization, Dropout
from keras.models import Sequential
model = None
model = Sequential()

model.add(Convolution2D(32, (3, 3), input_shape = (SIZE, SIZE, 3), activation = 'relu', data_format='channels_last'))
model.add(MaxPooling2D(pool_size = (2, 2), data_format="channels_last"))
model.add(BatchNormalization(axis = -1))
model.add(Dropout(0.2))

model.add(Convolution2D(32, (3, 3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2), data_format="channels_last"))
model.add(BatchNormalization(axis = -1))
model.add(Dropout(0.2))


model.add(Flatten())

model.add(Dense(activation = 'relu', units=512))
model.add(BatchNormalization(axis = -1))
model.add(Dropout(0.2))

model.add(Dense(activation = 'relu', units=256))
model.add(BatchNormalization(axis = -1))
model.add(Dropout(0.2))


model.add(Dense(activation = 'sigmoid', units=2))
model.compile(optimizer = 'Adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
print(model.summary())


# In[16]:


### Split the dataset
# 
# I split the dataset into training and testing dataset.
# 1. Training data: 80%
# 2. Testing data: 20%
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical

X_train, X_test, y_train, y_test = train_test_split(dataset, to_categorical(np.array(label)), test_size = 0.20, random_state = 9)

# When training with Keras's Model.fit(), adding the tf.keras.callback.TensorBoard callback 
# ensures that logs are created and stored. Additionally, enable histogram computation 
#every epoch with histogram_freq=1 (this is off by default)
#Place the logs in a timestamped subdirectory to allow easy selection of different training runs.
#import datetime

#log_dir="logs/fit/" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + "/"
#tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)


# In[17]:


# ### Training the model
# As the training data is now ready, I will use it to train the model.   

#Fit the model
history = model.fit(np.array(X_train), 
                         y_train, 
                         batch_size = 64, 
                         verbose = 1, 
                         epochs = 15,      
                         validation_split = 0.1,
                         shuffle = True
                      #   callbacks=callbacks
                     )


# In[18]:


# ## Accuracy calculation
# 
# I'll now calculate the accuracy on the test data.

print("Test_Accuracy: {:.2f}%".format(model.evaluate(np.array(X_test), np.array(y_test))[1]*100))

f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
t = f.suptitle('Model Performance', fontsize=12)
f.subplots_adjust(top=0.85, wspace=0.3)

max_epoch = len(history.history['accuracy'])+1
epoch_list = list(range(1,max_epoch))
ax1.plot(epoch_list, history.history['accuracy'], label='Train Accuracy')
ax1.plot(epoch_list, history.history['val_accuracy'], label='Validation Accuracy')
ax1.set_xticks(np.arange(1, max_epoch, 5))
ax1.set_ylabel('Accuracy Value')
ax1.set_xlabel('Epoch')
ax1.set_title('Accuracy')
l1 = ax1.legend(loc="best")
ax2.plot(epoch_list, history.history['loss'], label='Train Loss')
ax2.plot(epoch_list, history.history['val_loss'], label='Validation Loss')
ax2.set_xticks(np.arange(1, max_epoch, 5))
ax2.set_ylabel('Loss Value')
ax2.set_xlabel('Epoch')
ax2.set_title('Loss')
l2 = ax2.legend(loc="best")


# In[19]:


# Save the model to your local drive
model.save('C:\\Users\\Nayee\\OneDrive\\Desktop\\KIICE conference\\Malaria\\output')


# In[20]:


predictions = model.predict(np.array(X_test)) 


# In[21]:


classes = ['Parasitized', 'Uninfected']


# In[22]:


import itertools

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig((title +'.png'), format='png', dpi=600) # saving img as png


# In[23]:


pip install scikit-learn


# In[24]:


from sklearn.metrics import confusion_matrix

def make_confusion_matrix(y_test, predictions):
    cm = confusion_matrix(y_true=np.argmax(y_test, -1), y_pred=predictions.argmax(axis=1))
    plot_confusion_matrix(cm=cm, classes=classes)


# In[25]:


make_confusion_matrix(y_test, predictions)


# In[29]:


from sklearn.metrics import confusion_matrix, f1_score, precision_score

def make_confusion_matrix(y_test, predictions):
    # Calculate F1 score
    f1 = f1_score(y_true=np.argmax(y_test, -1), y_pred=np.argmax(predictions, axis=1), average='weighted')
    
    # Calculate precision
    precision = precision_score(y_true=np.argmax(y_test, -1), y_pred=np.argmax(predictions, axis=1), average='weighted')
    
    # Compute confusion matrix
    cm = confusion_matrix(y_true=np.argmax(y_test, -1), y_pred=np.argmax(predictions, axis=1))
    
    # plot_confusion_matrix(cm=cm, classes=classes)  # Assuming you have a function to plot confusion matrix
    
    return f1, precision


f1, precision = make_confusion_matrix(y_test, predictions)


print("F1 Score:", f1)
print("Precision:", precision)

