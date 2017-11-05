# emnist_image_generator_predictor

## Overview
This piece of code is an attempt to the assessment task assigned by BNP Paribas for the position of AI/Digital Analyst.

The selected task is: An English word image generators, then feed it to machine learning model [preferably neural network] to recognize the word from the image.

## Method
For demonstration purposes, a character handwriting recognizer (instead of an English word image recognizer) is implemented.  This character handwriting recognizer composes of two separate sections:
1. Image generator
    - A. Random selection from holdout data
    - B. Generating an image using Generative Adversarial Networks trained using samples of the respective character
        - Implementation only uses the Keras package
        - Implementation uses Keras and Keras-Adversarial packages
2. Recognition model - built using CNN neural networks with different hyperparameters

## Dataset
Both models take handwritten images from the EMNIST dataset[1] as training and validation data.  For simplification purposes, the EMNIST Balanced Dataset will be used, and therefore adjusting weights would not be required.

The full dataset can be found at http://www.itl.nist.gov/iaui/vip/cs_links/EMNIST/matlab.zip.  As an alternative, the emnist-balanced-small.mat file under the dataset folder can be used.  However, note this is a direct shrink of the EMNIST Balanced Dataset, and therefore classes will become imbalanced.

According to [1], the Balanced Dataset contains 47 classes, including both upper and lower case alphabets.  It should be noted that since characters such as s C, I, J, K, L, M, O, P, S, U, V, W, X, Y and Z have relatively similar upper and lower case letters, therefore the samples of these characters are merged into one single class (Only the Upper case class is available).  As a result, only 47 classes (including 10 digits) are available.  The focus of this demonstration is on English characters and hence the digit classes will be removed.

The EMNIST dataset, by default, comes in two separate training and testing sets.  However, since we are removing the digit classes, and their distribution among the default sets are unknown, we shall:
1. Consolidate all the training samples
2. Remove samples relating to digit classes
3. Split data into 4 portions:
    - Training Data (80% of total)
        - Training data for (2) (80% of Training Data or 64% of Total)
        - Validation data for (2) (20% of Training Data or 16% of Total)
    - Testing Data (20% of Total)
        - Testing data for (2) (50% of Testing Data or 10% of Total)
        - Training data for (1A) (50% of Testing Data or 10% of Total)
        
Note that in the usual data splitting practice, testing data accounts for 20% of the data (subject to size of dataset).  However, since the nature of the data required by (1A) is similar to that of testing, we shall take the samples from that dataset.

Note: Another data split available in the code, but require manual uncommenting.  This split focuses on giving the GAN models more samples to train from.  The split is as follows:
    - Data for (1B) (50% of total)
    - Data for (2) (50% of total)
        - Training Data for (2) (80% of 50%)
            - Training data for (2) (80% of Training Data)
            - Validation data for (2) (20% of Training Data)
        - Testing Data for (2) (20% of 50%)

## Models Built for the (1B) Image Generator Model
1. Keras Only Implementation
    - Generator
        - 3 feed forward layers 256, 512, 1024
    - Discriminator
        - 2 feed foward layers 512, 256
2. Keras and Keras-Adversarial Implementation
    - Generator
        - 3 feed forward layers 256, 512, 1024
    - Discriminator
        - 3 feed forward layers 1024, 512, 256

## Models Built for the (2) Recognition Model
Note: All models use dropout, batch normalization and relu optimization.
For details, please refer to model_hyperparameters.txt.

1. Model 0
    - 1 layer CNN and max pooling layer with 32 filters
    - 1 layer feed forward layer with 64 neurons
    - Dropout of 25%
2. Model 1
    - 1 layer CNN and max pooling layer with 64 filters
    - 1 layer feed forward layer with 64 neurons
    - Dropout of 25%
3. Model 2
    - 2 layers CNN and max pooling layer with 64 then 32 filters
    - 1 layer feed forward layer with 64 neurons
    - Dropout of 25%
4. Model 3
    - 2 layers CNN and max pooling layer with both 64 filters
    - 1 layer feed forward layer with 64 neurons
    - Dropout of 25%
5. Model 4
    - 2 layers CNN and max pooling layer with both 32 filters
    - 1 layer feed forward layer with 64 neurons
    - Dropout of 25%
6. Model 5
    - 2 layers CNN and max pooling layer with both 32 filters
    - 1 layer feed forward layer with 64 neurons
    - Dropout of varying ratios at different layers (75%, 50%, 25%)
7. Model 6
    - 2 layers CNN and max pooling layer with both 32 filters
    - 1 layer feed forward layer with 64 neurons
    - Dropout of varying ratios at different layers (75%, 75%, 50%)
8. Model 7
    - 2 layers CNN and max pooling layer with both 32 filters
    - 2 layer feed forward layer with 128 then 64 neurons
    - Dropout of varying ratios at different layers (75%, 75%, 50%, 50%)
9. Model 8
    - 2 layers CNN and max pooling layer with 64 then 32 filters
    - 2 layer feed forward layer with 128 then 64 neurons
    - Dropout of varying ratios at different layers (75%, 75%, 50%, 50%)
10. Model 9
    - 2 layers CNN and max pooling layer with both 64 filters
    - 2 layer feed forward layer with 128 then 64 neurons
    - Dropout of varying ratios at different layers (75%, 75%, 50%, 50%)
11. Model 10
    - 2 layers CNN and max pooling layer with both 64 filters
    - 2 layer feed forward layer with 256 then 128 neurons
    - Dropout of varying ratios at different layers (75%, 75%, 50%, 50%)

## Model Selection
The model with the best performance on the validation set will be selected, and the performance on the testing set would be reported as the final model performance.

## Results
Results on the model performances on the validation dataset is as follows:

- Model 0: Score: 51.17%
- Model 1: Score: 50.78%
- Model 2: Score: 39.42%
- Model 3: Score: 41.52%
- Model 4: Score: 40.50%
- Model 5: Score: 53.79%
- Model 6: Score: 77.15%
- Model 7: Score: 83.62%
- Model 8: Score: 76.51%
- Model 9: Score: 71.24%
- Model 10: Score: 57.59%

The best result comes from Model 7 (83.62%).  The accuracy on the test set of this model is 83.58%.

## Packages Used
- Keras (High level execution of deep learning models)
- Keras-Adversarial (High level execution of deep learning models) - installation files downloaded from github instead of pypi
- Tensorflow (Engine for deep learning models)
- Sckit-learn (For data pre-processing)
- Scipy (For processing dataset)
- Numpy (For loading dataset)
- H5py (For saving Keras models)
- Matplotlib (Optional) (For displaying image)
- Written using Python 3.6 (64-bit)

## Reference
- [1] Cohen, G., Afshar, S., Tapson, J., & van Schaik, A. (2017). EMNIST: an extension of MNIST to handwritten letters. Retrieved from http://arxiv.org/abs/1702.05373

This piece of program is built from scratch, however, similar pieces were later identified on the internet.  These include:
- [2] https://github.com/Coopss/EMNIST/blob/master/training.py
- [3] https://github.com/fchollet/keras/blob/master/examples/mnist_cnn.py
- [4] https://github.com/fchollet/keras/blob/master/examples/cifar10_cnn.py

Building GAN using Keras
- [5] https://github.com/eriklindernoren/Keras-GAN/blob/master/gan/gan.py

Building GAN using Keras-Adversarial
- [6] https://github.com/bstriner/keras-adversarial/blob/master/examples/example_gan.py
