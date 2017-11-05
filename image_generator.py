# Import required packages
import random
import data_parser
import os

# For Keras-Adversarial implementation
import numpy as np
from keras.layers import Reshape, Flatten, Activation
from keras.models import Sequential
from keras.optimizers import Adam
from keras_adversarial import AdversarialModel, simple_gan, gan_targets
from keras_adversarial import normal_latent_sampling, AdversarialOptimizerAlternating
from keras_adversarial.legacy import l1l2, Dense, fit

# For Keras implementation
from keras.layers import Input, Dense, Dropout
from keras.layers import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.models import Model
from keras.models import load_model

# For image displayer
import data_parser

# There are 3 image generators available:
# 1. basic_image_generator
# 2. gan_image_generator_keras_adversarial
# 3. gan_image_generator_keras

class image_generator (object):
    # The image_generator object is the parent class for basic_image_generator and gan_image_generator
    def __init__(self,data,mapping_table,adjustment_list):
        self.__X = data [6]
        self.__Y = data [7]
        self.__mapping_table = mapping_table
        self.__adjustment_list = adjustment_list

        # Place all entries into a dictionary
        self.collection = {}
        for i in range(len(self.__Y)):
            target_class = data_parser.target_class_displayer(self.__Y[i].argmax(),self.__mapping_table,self.__adjustment_list).target_class_to_actual()
            if self.collection.get(target_class):
                self.collection.get(target_class).append(self.__X[i])
            else:
                self.collection[target_class] = [self.__X[i]]

class basic_image_generator(image_generator):
    # The basic_image_generator class is a random image selector
    # Image data from pre-processing is reserved for this random image selector
    # When given a certain valid character, the image generator returns an image array from the hold out set
    def __init__(self,data,mapping_table,adjustment_list):
        image_generator.__init__(self,data,mapping_table,adjustment_list)

    def get_image(self,character,seed=42):
        # Return a random image from the set as a the selected image 
        image_set = self.collection.get(character)
        random.seed(seed)
        if image_set:
            return random.choice(image_set)
        print("ERROR: This is not a valid character\n")
        return None

##########################################################################################

class gan_image_generator_keras_adversarial(image_generator):
    # The GAN_image_generator class builds a collection of Generative Adversarial Network (GAN) models
    # Each model is a generator for a character class
    def __init__(self,data,mapping_table,adjustment_list,saving_folder = './gan_collection_keras_adversarial'):
        image_generator.__init__(self,data,mapping_table,adjustment_list)
        self.__gan_collection = {}

        # Load the GAN models if pre-built
        if os.path.exists(saving_folder):
            for file in os.listdir(saving_folder):
                name = file[:-3]
                filepath = os.path.join(saving_folder,file)
                model = load_model(filepath)
                self.__gan_collection[name] =  model

    def get_image(self,character,seed=42):
        # Initialize seed
        np.random.seed(seed)
        
        # Get the character dataset, if there is none, then the character is invalid
        dataset = self.collection.get(character,None)
        if dataset == None:
            print("ERROR: This is not a valid character\n")
            return None

        # Check if a model is existent in gan collection
        # If the model is not existent, then build one 
        if character not in self.__gan_collection:
            model = gan_model_keras_adversarial(dataset)
            self.__gan_collection[char] = model

        # Return the prediction of model as the generated image
        model = self.__gan_collection.get(character)
        generated_image = model.predict()
        return generated_image

class gan_model_keras_adversarial(object):
    # The gan model is adopted from bstriner/keras-adversarial/examples/example_gan.py on Github
    # https://github.com/bstriner/keras-adversarial/blob/master/examples/example_gan.py
    def __init__(self,data):
        # Reshape data such that it is suitable for input
        data = np.asarray(data)
        data = data.reshape(data.shape[0],data.shape[1],data.shape[2])

        # Initialize parameters
        self.latent_dim = 100
        self.input_shape = (28, 28)

        # Initialize models 
        generator = self.model_generator(self.latent_dim , self.input_shape)
        discriminator = self.model_discriminator(self.input_shape)

        # Train models
        self.generator_trained = self.example_gan(data,
                                                AdversarialOptimizerAlternating(),
                                                opt_g=Adam(),
                                                opt_d=Adam(),
                                                nb_epoch=20, generator=generator, discriminator=discriminator,
                                                latent_dim=self.latent_dim)

    def model_generator(self,latent_dim, input_shape, hidden_dim=1024, reg=lambda: l1l2(1e-5, 1e-5)):
        # This function returns a generator model
        return Sequential([
                Dense(int(hidden_dim / 4), input_dim=self.latent_dim,activation='relu'),
                Dense(int(hidden_dim / 2), activation='relu'),
                Dense(hidden_dim,activation='relu'),
                Dense(np.prod(self.input_shape),activation='sigmoid'),
                Reshape(self.input_shape)])


    def model_discriminator(self,input_shape, hidden_dim=1024, reg=lambda: l1l2(1e-5, 1e-5), output_activation="sigmoid"):
        # This function returns a discriminator model
        return Sequential([
                Flatten(input_shape=self.input_shape),
                Dense(hidden_dim,activation='relu'),
                Dense(int(hidden_dim / 2),activation='relu'),
                Dense(int(hidden_dim / 4),activation='relu'),
                Dense(1,activation='sigmoid')])

    def example_gan(self,data,adversarial_optimizer, opt_g, opt_d, nb_epoch, generator, discriminator, latent_dim,
                    targets=gan_targets, loss='binary_crossentropy'):
        # Create gan model
        gan = simple_gan(generator, discriminator, normal_latent_sampling((latent_dim,)))

        # Print summary of models
        generator.summary()
        discriminator.summary()
        gan.summary()

        # Place gan model into adversarial model
        model = AdversarialModel(base_model=gan,
                                 player_params=[generator.trainable_weights, discriminator.trainable_weights])
        model.adversarial_compile(adversarial_optimizer=adversarial_optimizer,
                                  player_optimizers=[opt_g, opt_d],
                                  loss=loss)

        # Train model
        xtrain = data
        y = targets(xtrain.shape[0])
        history = fit(model, x=xtrain, y=y,nb_epoch=nb_epoch,batch_size=32)

        # Return generator model
        return generator

    def generate_photo (self):
        # Generate image
        zsamples = np.random.normal(size=(1, self.latent_dim))
        return self.generator_trained.predict(zsamples).reshape((28, 28))

################################################################################################

class gan_image_generator_keras(image_generator):
    # The GAN_image_generator class builds a collection of Generative Adversarial Network (GAN) models
    # Each model is a generator for a character class
    def __init__(self,data,mapping_table,adjustment_list,saving_folder = './gan_collection_keras'):
        image_generator.__init__(self,data,mapping_table,adjustment_list)
        self.__gan_collection = {}

        # Load the GAN models if pre-built
        if os.path.exists(saving_folder):
            for file in os.listdir(saving_folder):
                name = file[:-3]
                filepath = os.path.join(saving_folder,file)
                model = load_model(filepath)

                self.__gan_collection[name] =  model

    def get_image(self,character,seed=42):
        np.random.seed(seed)
        dataset = self.collection.get(character,None)
        if dataset == None:
            print("ERROR: This is not a valid character\n")
            return None

        if character not in self.__gan_collection:
            model = gan_model_keras()
            model.train(dataset,epochs=10000, batch_size=32)
            self.__gan_collection[character] = model.generator

        model = self.__gan_collection.get(character)
        input_noise = np.random.normal(0, 1, (1, 100))
        generated_image = model.predict(input_noise)
        return generated_image

class gan_model_keras(object):
    # The gan model is adopted from eriklindernoren/Keras-GAN/gan/gan.py on Github
    def __init__(self):
        # Build and compile the discriminator
        self.img_shape = (28, 28, 1)
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy', 
                                    optimizer='adam',
                                    metrics=['accuracy'])

        # Build and compile the generator
        self.generator = self.build_generator()
        self.generator.compile(loss='binary_crossentropy', optimizer='adam')

        # The generator takes noise as input and generated imgs
        z = Input(shape=(100,))
        img = self.generator(z)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The valid takes generated images as input and determines validity
        valid = self.discriminator(img)

        # The combined model  (stacked generator and discriminator) takes
        # noise as input => generates images => determines validity 
        self.combined = Model(z, valid)
        self.combined.compile(loss='binary_crossentropy', optimizer='adam')

    def build_generator(self):
        noise_shape = (100,)

        # Set generator to be with 4 layers of dense layers
        model = Sequential()
        model.add(Dense(256, input_shape=noise_shape,activation='relu'))
        model.add(BatchNormalization())
        model.add(Dense(512,activation='relu'))
        model.add(BatchNormalization())
        model.add(Dense(1024,activation='relu'))
        model.add(BatchNormalization())
        model.add(Dense(np.prod(self.img_shape), activation='tanh'))
        model.add(Reshape(self.img_shape))

        # Display summary of generator
        model.summary()

        # Determine input and output
        noise = Input(shape=noise_shape)
        img = model(noise)
        return Model(noise, img)

    def build_discriminator(self):
        # Set discriminator to be with 3 layers of dense layers      
        model = Sequential()
        model.add(Flatten(input_shape=self.img_shape))
        model.add(Dense(512,activation='relu'))
        model.add(Dense(256,activation='relu'))
        model.add(Dense(1, activation='sigmoid'))

        # Display summary of discriminator
        model.summary()

        # Determine input and output
        img = Input(shape=self.img_shape)
        validity = model(img)
        return Model(img, validity)

    def train(self, char,X_train, epochs, batch_size=128,display_size =1000 ):
        # Initialize parameters
        X_train = np.asarray(X_train)
        half_batch = int(batch_size / 2)

        for epoch in range(epochs):
            # Train discriminator first
            # Select a random half batch of images
            idx = np.random.randint(0, X_train.shape[0], half_batch)
            imgs = X_train[idx]

            # Generate fake pictures for the discriminator to perform forward and backward propagation
            noise = np.random.normal(0, 1, (half_batch, 100))
            gen_imgs = self.generator.predict(noise)
            d_loss_real = self.discriminator.train_on_batch(imgs, np.ones((half_batch, 1)))
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, np.zeros((half_batch, 1)))
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # Train generator
            # Use noise as input and let the generator perform forward and backward propagation
            noise = np.random.normal(0, 1, (batch_size, 100))

            # Set the truth values to all 1, since the objective for the generator is to trick the discriminator into believing the fake images are real
            valid_y = np.array([1] * batch_size)

            # Train the generator
            g_loss = self.combined.train_on_batch(noise, valid_y)

            # Plot the progress
            print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))

            if epoch%display_size == 0:
                print(gen_imgs[0].shape)
                data_parser.image_displayer(gen_imgs[0]).display_image()
