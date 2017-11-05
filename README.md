# emnist_image_generator_predictor

## Overview
This piece of code is an attempt to the assessment task assigned by BNP Paribas for the position of AI/Digital Analyst.

The selected task is: An English word image generators, then feed it to machine learning model [preferably neural network] to recognize the word from the image.

## Method
For demonstration purposes, a character handwriting recognizer (instead of an English word image recognizer) is implemented.  This character handwriting recognizer composes of two separate sections:
1. Image generator
    - A. Random selection from holdout data
    - B. Generating an image using Generative Adversarial Networks trained using samples of the respective character
2. Recognition model - built using CNN neural networks with different hyperparameters

## Models Built for the (2) Recognition Model
1.

## Model Selection
The model with the best performance on the validation set will be selected, and the performance on the testing set would be reported as the final model performance.  Since we care about how accurate the model can predict the correct answer, our primary performance measurement is precision.

## Dataset
Both models take handwritten images from the EMNIST dataset[1] as training and validation data.  For simplification purposes, the EMNIST Balanced Dataset will be used.

According to [1], the Balanced Dataset contains 47 classes, including both upper and lower case alphabets.  It should be noted that since characters such as s C, I, J, K, L, M, O, P, S, U, V, W, X, Y and Z have relatively similar upper and lower case letters, therefore the samples of these characters are merged into one single class.  As a result, only 47 classes (including 10 digits) are available.  The focus of this demonstration is on English characters and hence the digit classes will be removed.

The EMNIST dataset, by default, comes in two separate training and testing sets.  However, since we are removing the digit classes, and their distribution among the default sets are unknown, we shall:
1. Consolidate all the training samples
2. Remove samples relating to digit classes
3. Split data into 4 portions:
    - Training Data (80% of total)
        - Training data for (2) (80% of Training Data or 64% of Total)
        - Validation data for (2) (20% of Training Data or 16% of Total)
    - Testing Data (20% of Total)
        - Testing dta for (2) (50% of Testing Data or 10% of Total)
        - Training data for (1A) (50% of Testing Data or 10% of Total)

Note that in the usual data splitting practice, testing data accounts for 20% of the data (subject to size of dataset).  However, since the nature of the data required by (1A) is similar to that of testing, we shall take the samples from that dataset.


## Packages Used
- Keras (High level execution of deep learning models)
- Tensorflow (Engine for deep learning models)
- Sckit-learn (For data pre-processing)
- Scipy (For loading dataset)

- Written using Anaconda 5.0.0 (Python 3.6) in Ubuntu 16.04.3 (64-bit)

## Results
Results on the model performances on the validation dataset is as follows:

Model 1: Score: 0.8683035714285714
Model 2: Score: 0.85253609422492405
Model 3: Score: 0.88188639817629177
Model 4: Score: 0.85600303951367784
Model 5: Score: 0.85804521276595747
Model 6: Score: 0.84237272036474165
Model 7: Score: 0.88207636778115506
Model 8: Score: 0.86369680851063835
Model 9: Score: 0.88231382978723405
Model 10: Score: 0.86127469604863227

The best result is Model 9 (88.23%), and compared to the hyperparameters of the base model, only the train epochs has increased.  This increase in model performance may therefore be due to overfitting.  As the next best alternative, Model 7 gives an accuracy of 88.21%.  This is achieved by increasing the dense layer size from 128 to 256.  We should take this model as our final model.

The accuracy on the test set of this model is 87.93%.



## Reference
- [1] Cohen, G., Afshar, S., Tapson, J., & van Schaik, A. (2017). EMNIST: an extension of MNIST to handwritten letters. Retrieved from http://arxiv.org/abs/1702.05373

This piece of program is built from scratch, however, similar pieces were later identified on the internet.  These include:
- [2] https://github.com/Coopss/EMNIST/blob/master/training.py
- [3] https://github.com/fchollet/keras/blob/master/examples/mnist_cnn.py
- [4] https://github.com/fchollet/keras/blob/master/examples/cifar10_cnn.py

Building GAN using Keras
- [5] https://github.com/eriklindernoren/Keras-GAN/blob/master/gan/gan.py
