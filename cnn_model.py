# Import required packages
import numpy as np
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Input, Dense
from keras.layers import Dense, Dropout, Flatten, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
import data_parser

# Hyperparameters should be a list that contains sequentially the layers in the CNN model
# Layers supported include: conv2d','maxpooling','dropout','flatten','dense'
# Note:
## 1. The supported parameters of each layer type are given in the below example
## 2. The sequential model is used for all CNN models and is automatically added
## 3. The final layer must be a dense layer with a softmax activation function to 47 classes, and is also automaticalled added as well
# 
# For example one example of a CNN model can be:
# [
# {'type':'conv2d','filters':32, 'kernel_size':3,'strides':(1,1),'padding':'same','activation':'relu'}, # Full list of activation functions available can be found at https://keras.io/activations/
# {'type':'maxpooling','pool_size':(2,2)},
# {'type':'dropout','dropout_ratio':0.25},
# {'type':'batchnormalization'},
# {'type':'flatten'},
# {'type':'dense','dense_layer_size':128,'activation':'relu'},
# {'type':'dropout','dropout_ratio':0.25}
# ]
#
#
# Training_parameters should be a dictionary that contains the following items:
# 'optimizer','verbose','train_batch_size','train_epochs'
#
# {
# 'optimizer':'adam',  # Full list of optimizers can be found at https://keras.io/optimizers/
# 'verbose':1,         # Verbose: either 1 or 0
# 'train_batch_size':32,
# 'train_epochs':10
# }


class CNN_model (object):
    def __init__ (self,data=None,hyperparameters=None,training_parameters=None,load_model= None):
        if load_model == None:
            self.__train_X = data[0]
            self.__train_Y = data[1]
            self.__val_X = data[2]
            self.__val_Y = data[3]
            self.__test_X = data[4]
            self.__test_Y = data[5]

            self.__hyperparameters = hyperparameters
            self.__training_parameters = training_parameters

            self.__input_shape = (self.__train_X.shape[1],self.__train_X.shape[2],1)
            self.__conv2d_count = 0

            # Construct CNN Model based on the given hyperparameters
            self.__model = Sequential()
            for layer in self.__hyperparameters:
                self.__add_layer(layer)
            self.__model.add(Dense(47,activation='softmax'))

            # Compile model
            self.__model.compile(loss='categorical_crossentropy',
                                optimizer=self.__training_parameters.get('optimizer'),
                                metrics=['accuracy'])

            # Show summary (if specified)
            if self.__training_parameters.get('verbose') == 1:
                self.__model.summary()
            
            # Train model
            self.__model.fit(self.__train_X,
                            self.__train_Y,
                            batch_size=self.__training_parameters.get('train_batch_size'),
                            epochs=self.__training_parameters.get('train_epochs'),
                            verbose=self.__training_parameters.get('verbose'))
        else:
            self.__model = load_model

    def __add_layer(self,layer):
        # Add convolution layer based on specified hyperparameters
        # Note: Channels_First selected for data_format: (batch, height, width, channels)
        if layer.get('type') == 'conv2d':
            if self.__conv2d_count == 0:
                self.__model.add(Conv2D(filters=layer.get('filters'),
                                        kernel_size=layer.get('kernel_size'),
                                        strides=layer.get('strides'),
                                        padding=layer.get('padding'),
                                        activation=layer.get('relu'),
                                        data_format="channels_last", 
                                        input_shape=self.__input_shape))
            else:
                self.__model.add(Conv2D(filters=layer.get('filters'),
                                        kernel_size=layer.get('kernel_size'),
                                        strides=layer.get('strides'),
                                        padding=layer.get('padding'),
                                        activation=layer.get('relu'),
                                        data_format="channels_last"))
            self.__conv2d_count += 1

        # Add max pooling layer based on specified hyperparameters
        elif layer.get('type') == 'maxpooling':
            self.__model.add(MaxPooling2D(pool_size=layer.get('pool_size')))

        # Add dropout layer based on specified hyperparameters
        elif layer.get('type') == 'dropout':
            self.__model.add(Dropout(rate=layer.get('dropout_ratio')))

        # Add flatten layer based on specified hyperparameters
        # Flatten refers the conversion of the matrices to a single vector for input into a dense/feed-forward layer
        elif layer.get('type') == 'flatten':
            self.__model.add(Flatten())

        # Add dropout layer based on specified hyperparameters
        elif layer.get('type') == 'dense':
            self.__model.add(Dense(units=layer.get('dense_layer_size'),activation=layer.get('activation')))

        # Add batch normalization layer
        elif layer.get('type') == 'batchnormalization':
            self.__model.add(BatchNormalization())

    def get_test_score (self):
        if not hasattr(self,'__test_score'):
            # Evaluate performance of model on test set
            self.__test_score = self.__model.evaluate(self.__test_X,
                                                        self.__test_Y,
                                                        verbose=0)
        return self.__test_score

    def get_val_score (self):
        if not hasattr(self,'__val_score'):
            # Evaluate performance of model on validation set
            self.__val_score = self.__model.evaluate(self.__val_X,
                                                    self.__val_Y,
                                                    verbose=0)
        return self.__val_score

    def save_cnn_model(self,filepath):
        self.__model.save(filepath)

    def get_model(self):
        return self.__model

    def get_training_parameters(self):
        return self.__training_parameters

    def predict_image(self,X_array,mapping_table,adjustment_list):
        # Returns the prediction of the given image array
        # X_array is the image array to be predicted from the image generator
        # mapping_table is the mapping table that converts the predicted class to the ascii number of the character
        input_shape = X_array.shape
        array_shape = (1,input_shape[0],input_shape[1],input_shape[2])
        X_array = X_array.reshape(array_shape)

        predictions = self.__model.predict(X_array,verbose=0)

        predicted_class = np.argmax(predictions[0])
        return data_parser.target_class_displayer(predicted_class,mapping_table,adjustment_list).target_class_to_actual()