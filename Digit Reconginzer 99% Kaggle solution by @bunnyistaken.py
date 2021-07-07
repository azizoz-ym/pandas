# %% [code] {"execution":{"iopub.status.busy":"2021-07-07T10:41:13.071518Z","iopub.execute_input":"2021-07-07T10:41:13.071882Z","iopub.status.idle":"2021-07-07T10:41:13.077884Z","shell.execute_reply.started":"2021-07-07T10:41:13.071852Z","shell.execute_reply":"2021-07-07T10:41:13.076693Z"}}
# Importing Required Libraries
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, Conv2D, MaxPooling2D, Flatten, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
import cv2
import seaborn as sns
import matplotlib.pyplot as plt

# %% [code] {"execution":{"iopub.status.busy":"2021-07-07T10:41:21.297325Z","iopub.execute_input":"2021-07-07T10:41:21.297661Z","iopub.status.idle":"2021-07-07T10:41:24.369532Z","shell.execute_reply.started":"2021-07-07T10:41:21.297633Z","shell.execute_reply":"2021-07-07T10:41:24.368411Z"}}
# Reading Data from provided .csv file
data = pd.read_csv('../input/digit-recognizer/train.csv')
print(data.info())
data.head()

# %% [code] {"execution":{"iopub.status.busy":"2021-07-07T10:41:36.936820Z","iopub.execute_input":"2021-07-07T10:41:36.937211Z","iopub.status.idle":"2021-07-07T10:41:36.945478Z","shell.execute_reply.started":"2021-07-07T10:41:36.937180Z","shell.execute_reply":"2021-07-07T10:41:36.944761Z"}}
# Retrieving labels from Data
labels = data['label']

# Converting into One-Hot Vectors for Classification
labels = pd.get_dummies(labels)
labels.shape

# %% [code] {"execution":{"iopub.status.busy":"2021-07-07T10:41:49.167472Z","iopub.execute_input":"2021-07-07T10:41:49.168051Z","iopub.status.idle":"2021-07-07T10:41:49.174205Z","shell.execute_reply.started":"2021-07-07T10:41:49.168015Z","shell.execute_reply":"2021-07-07T10:41:49.173418Z"}}
# Retrieving all pixel columns from Data
data = data.iloc[:,1:]
data.shape 

# %% [code] {"execution":{"iopub.status.busy":"2021-07-07T10:42:15.189048Z","iopub.execute_input":"2021-07-07T10:42:15.189667Z","iopub.status.idle":"2021-07-07T10:42:15.196263Z","shell.execute_reply.started":"2021-07-07T10:42:15.189632Z","shell.execute_reply":"2021-07-07T10:42:15.195174Z"}}
# Taking 40000 images to Train the model
train_data = data[2000:]
train_labels = labels[2000:]

train_data.shape, train_labels.shape

# %% [code] {"execution":{"iopub.status.busy":"2021-07-07T10:42:29.609089Z","iopub.execute_input":"2021-07-07T10:42:29.609487Z","iopub.status.idle":"2021-07-07T10:42:29.615638Z","shell.execute_reply.started":"2021-07-07T10:42:29.609457Z","shell.execute_reply":"2021-07-07T10:42:29.614793Z"}}
# Taking 2000 images for Cross Validation
test_data = data[:2000]
test_labels = labels[:2000]

test_data.shape, test_labels.shape

# %% [code] {"execution":{"iopub.status.busy":"2021-07-07T10:42:58.997202Z","iopub.execute_input":"2021-07-07T10:42:58.997744Z","iopub.status.idle":"2021-07-07T10:42:59.051248Z","shell.execute_reply.started":"2021-07-07T10:42:58.997692Z","shell.execute_reply":"2021-07-07T10:42:59.050199Z"}}
# Standard loc and scale values for kernel initializer 
def initialize_weights(shape, dtype=None):
    
    return np.random.normal(loc = 0.0, scale = 1e-2, size = shape)

# Standard loc and scale values for bias initializer
def initialize_bias(shape, dtype=None):
    
    return np.random.normal(loc = 0.5, scale = 1e-2, size = shape)

def DeepLearningModel(input_shape):
    model = Sequential()
    
    model.add(Input(input_shape))
    
    model.add(Dense(64, activation='relu', kernel_initializer=initialize_weights, bias_initializer=initialize_bias, kernel_regularizer=l2(3e-3)))
    
    model.add(Dense(128, activation='relu', kernel_initializer=initialize_weights, bias_initializer=initialize_bias, kernel_regularizer=l2(3e-3)))

    model.add(Dense(256, activation='relu', kernel_initializer=initialize_weights, bias_initializer=initialize_bias, kernel_regularizer=l2(3e-3)))
                
    model.add(Dense(10, activation='softmax', kernel_initializer=initialize_weights, bias_initializer=initialize_bias, kernel_regularizer=l2(3e-3)))    
    
    return model


model = DeepLearningModel((784))
model.summary()

# %% [code] {"execution":{"iopub.status.busy":"2021-07-07T10:43:13.876995Z","iopub.execute_input":"2021-07-07T10:43:13.877394Z","iopub.status.idle":"2021-07-07T10:44:10.453056Z","shell.execute_reply.started":"2021-07-07T10:43:13.877361Z","shell.execute_reply":"2021-07-07T10:44:10.451543Z"}}
# Hyperparameters
lr = 0.0003
epochs = 50
batch_size = 128

optimizer = Adam(lr)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(train_data, train_labels, epochs=epochs, batch_size=batch_size)

# %% [code] {"execution":{"iopub.status.busy":"2021-07-07T10:44:39.163355Z","iopub.execute_input":"2021-07-07T10:44:39.163696Z","iopub.status.idle":"2021-07-07T10:44:39.495945Z","shell.execute_reply.started":"2021-07-07T10:44:39.163668Z","shell.execute_reply":"2021-07-07T10:44:39.495132Z"}}
# Evaluating model using Validation Data
model.evaluate(test_data, test_labels)

# %% [code] {"execution":{"iopub.status.busy":"2021-07-07T10:44:52.386066Z","iopub.execute_input":"2021-07-07T10:44:52.386452Z","iopub.status.idle":"2021-07-07T10:44:54.765210Z","shell.execute_reply.started":"2021-07-07T10:44:52.386420Z","shell.execute_reply":"2021-07-07T10:44:54.764139Z"}}
# Reading Test Dataset
test = pd.read_csv('../input/digit-recognizer/test.csv')
test.info()

# %% [code] {"execution":{"iopub.status.busy":"2021-07-07T10:45:55.076607Z","iopub.execute_input":"2021-07-07T10:45:55.076967Z","iopub.status.idle":"2021-07-07T10:45:56.308912Z","shell.execute_reply.started":"2021-07-07T10:45:55.076937Z","shell.execute_reply":"2021-07-07T10:45:56.308017Z"}}
# Making Predictions on existing model.
prediction = model.predict(test)

# Rounding the prediction and converting float64 output to int64 output.
prediction = np.array(np.round(prediction), dtype='int64')

# Converting one-hot encoding into label encoding
prediction = (np.argmax(prediction, axis=1)).reshape(-1, 1)

# Creating a DataFrame similar to Example Submission
out = [{'ImageId': i+1, 'Label': prediction[i][0]} for i in range(len(prediction))]

# Creating a .csv file from the DataFrame
pd.DataFrame(out).to_csv('./submission.csv', index=False)

# %% [code] {"execution":{"iopub.status.busy":"2021-07-07T10:46:07.959309Z","iopub.execute_input":"2021-07-07T10:46:07.959681Z","iopub.status.idle":"2021-07-07T10:46:08.064712Z","shell.execute_reply.started":"2021-07-07T10:46:07.959648Z","shell.execute_reply":"2021-07-07T10:46:08.063686Z"}}
imgdata = np.reshape(np.array(data),(data.shape[0],28,28,1))
imglabels = labels

imgdata.shape, imglabels.shape

# %% [code] {"execution":{"iopub.status.busy":"2021-07-07T10:46:19.727265Z","iopub.execute_input":"2021-07-07T10:46:19.727623Z","iopub.status.idle":"2021-07-07T10:46:19.735475Z","shell.execute_reply.started":"2021-07-07T10:46:19.727591Z","shell.execute_reply":"2021-07-07T10:46:19.734561Z"}}
# Taking 40000 images to Train the model
img_train_data = imgdata[2000:]
img_train_labels = imglabels[2000:]

img_train_data.shape, img_train_labels.shape

# %% [code] {"execution":{"iopub.status.busy":"2021-07-07T10:46:27.590493Z","iopub.execute_input":"2021-07-07T10:46:27.590834Z","iopub.status.idle":"2021-07-07T10:46:27.597862Z","shell.execute_reply.started":"2021-07-07T10:46:27.590803Z","shell.execute_reply":"2021-07-07T10:46:27.596845Z"}}
# Taking 2000 images for Cross Validation
img_test_data = imgdata[:2000]
img_test_labels = imglabels[:2000]

img_test_data.shape, img_test_labels.shape

# %% [code] {"execution":{"iopub.status.busy":"2021-07-07T10:46:39.237661Z","iopub.execute_input":"2021-07-07T10:46:39.237999Z","iopub.status.idle":"2021-07-07T10:46:41.935951Z","shell.execute_reply.started":"2021-07-07T10:46:39.237970Z","shell.execute_reply":"2021-07-07T10:46:41.934947Z"}}
# Plotting Dataset

plt.figure(figsize=(6, 6))
for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img_train_data[i])
    plt.tight_layout()
plt.show()

# %% [code] {"execution":{"iopub.status.busy":"2021-07-07T10:47:29.264485Z","iopub.execute_input":"2021-07-07T10:47:29.264963Z","iopub.status.idle":"2021-07-07T10:47:29.275143Z","shell.execute_reply.started":"2021-07-07T10:47:29.264931Z","shell.execute_reply":"2021-07-07T10:47:29.273738Z"}}
# Building an Image Dataset for a Convolutional Neural Network.
def convolutional_model(input_shape):
    model = Sequential()
    
    model.add(Conv2D(32, (3,3), activation='relu', input_shape=input_shape,
                   kernel_initializer=initialize_weights, kernel_regularizer=l2(2e-4)))
    
    model.add(MaxPooling2D())
    
    model.add(Dropout(0.25))
    
    model.add(Conv2D(32, (3,3), activation='relu', kernel_initializer=initialize_weights,
                     bias_initializer=initialize_bias, kernel_regularizer=l2(2e-4)))
    
    model.add(MaxPooling2D())
    
    model.add(Dropout(0.25))
    
    model.add(Conv2D(32, (3,3), activation='relu', kernel_initializer=initialize_weights,
                     bias_initializer=initialize_bias, kernel_regularizer=l2(2e-4)))
    
    model.add(Flatten())
    
    model.add(Dense(1024, activation='relu',
                   kernel_regularizer=l2(1e-3),
                   kernel_initializer=initialize_weights,bias_initializer=initialize_bias))
    
    model.add(Dense(10, activation='softmax',
                   kernel_regularizer=l2(1e-3),
                   kernel_initializer=initialize_weights,bias_initializer=initialize_bias))
    return model


# %% [code] {"execution":{"iopub.status.busy":"2021-07-07T10:47:35.686742Z","iopub.execute_input":"2021-07-07T10:47:35.687139Z","iopub.status.idle":"2021-07-07T10:47:35.783592Z","shell.execute_reply.started":"2021-07-07T10:47:35.687089Z","shell.execute_reply":"2021-07-07T10:47:35.782645Z"}}
model = convolutional_model((28,28,1))
model.summary()

# %% [code] {"execution":{"iopub.status.busy":"2021-07-07T10:47:47.954048Z","iopub.execute_input":"2021-07-07T10:47:47.954956Z","iopub.status.idle":"2021-07-07T10:50:28.280030Z","shell.execute_reply.started":"2021-07-07T10:47:47.954908Z","shell.execute_reply":"2021-07-07T10:50:28.279209Z"}}
# Hyperparameters
lr = 0.0003
epochs = 10
batch_size = 128

optimizer = Adam(lr)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(img_train_data, img_train_labels, epochs=epochs, batch_size=batch_size)

# %% [code] {"execution":{"iopub.status.busy":"2021-07-07T10:50:37.546083Z","iopub.execute_input":"2021-07-07T10:50:37.546495Z","iopub.status.idle":"2021-07-07T10:50:38.082872Z","shell.execute_reply.started":"2021-07-07T10:50:37.546457Z","shell.execute_reply":"2021-07-07T10:50:38.082131Z"}}
model.evaluate(img_test_data, img_test_labels)

# %% [code] {"execution":{"iopub.status.busy":"2021-07-07T10:50:52.758841Z","iopub.execute_input":"2021-07-07T10:50:52.759223Z","iopub.status.idle":"2021-07-07T10:50:54.610560Z","shell.execute_reply.started":"2021-07-07T10:50:52.759186Z","shell.execute_reply":"2021-07-07T10:50:54.609459Z"}}
# Reading Test Dataset
test = pd.read_csv('../input/digit-recognizer/test.csv')
test = np.reshape(np.array(test), (test.shape[0], 28, 28, 1))

# %% [code] {"execution":{"iopub.status.busy":"2021-07-07T10:51:11.254308Z","iopub.execute_input":"2021-07-07T10:51:11.254666Z","iopub.status.idle":"2021-07-07T10:51:15.580642Z","shell.execute_reply.started":"2021-07-07T10:51:11.254636Z","shell.execute_reply":"2021-07-07T10:51:15.579535Z"}}
# Making Predictions on existing model.
prediction = model.predict(test)

# Rounding the prediction and converting float64 output to int64 output.
prediction = np.array(np.round(prediction), dtype='int64')

# Converting one-hot encoding into label encoding
prediction = (np.argmax(prediction, axis=1)).reshape(-1, 1)

# Creating a DataFrame similar to Example Submission
out = [{'ImageId': i+1, 'Label': prediction[i][0]} for i in range(len(prediction))]

# Creating a .csv file from the DataFrame
pd.DataFrame(out).to_csv('submission_v2.csv', index=False)

# %% [code]


# %% [code]


# %% [code]


# %% [code]


# %% [code]


# %% [code]


# %% [code]


# %% [code]
