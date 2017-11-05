# Import required packages
from scipy.io import loadmat
# from matplotlib import pyplot as plt  # Uncomment if programme run in iPython (Gives a better display of image)
import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils import np_utils
import pickle

class image_displayer (object):
    # Image displayer is used for displaying a processed image array in graphics
    # Image class is not built since it would be incompatibale with various dataset level data manipulation under scikit-learn or numpy
    def __init__(self,image_array):
        if image_array.shape == (28,28,1):
            self.__image_array = image_array.reshape(28,28,)
        else:
            print("Please provide a processed image array")
            raise

    def display_image (self):
        # Display image of processed image matrix

        # Uncomment below if programme run in iPython (gives a better display of image)
        #plt.imshow(self.__image_array, cmap='gray', interpolation='nearest', vmin=0, vmax=1)
        #plt.show()

        # Display image in command line
        print("Preview of selected image:\n")
        for i in self.__image_array:
            for j in i:
                if j >= 0.5:
                    print ("***",end="")
                else:
                    print("   ",end="")
            print("\n")
        print("\n")

class target_class_displayer (object):
    # Target class displayer converts the integer from the target class in the dataset into the actual character
    def __init__(self,target_class,mapping_table,adjustment_list):
        self.__target_class = target_class
        self.__mapping_table = mapping_table
        self.__adjustment_list = adjustment_list
    
    def target_class_to_actual(self):
        # Return the character represented by the category number
        return chr(self.__mapping_table.get(self.__adjustment_list[self.__target_class]))

class data_parser(object):
    def __init__(self,filepath,seed=42):
        self.__filepath = filepath
        np.random.seed(seed)

        if self.__filepath.endswith(".mat"):
            # Re-process data if file is a matlab file

            try:
                # Load data from matlab file
                self.__dataset = self.__load_matlab_file()

            except:
                print ("Please provide a valid matlab file.")
                raise
                
            # Parse mapping table from dataset
            self.__mapping_table = self.__load_mapping_table ()

            # Perform all required preprocessing of image data
            ## 1. Combine data from original training and testing sets
            ## 2. Remove digit classes
            self.__X, self.__Y = self.__consol_data_from_dataset ()
            self.__X, self.__Y = self.__remove_digit_classes ()

            ## 3. Adjust the prediction classes due to the removal of the digit classes
            self.__adjustment_table, self.__adjustment_list, self.__Y = self.__adjust_Y()

            ## 4. Convert data to format required by Keras
            self.__X, self.__Y = np.asarray(self.__X), np.asarray(self.__Y)
            self.__X = self.__X.reshape(self.__X.shape[0],self.__X.shape[1],self.__X.shape[2],1)
            self.__Y = self.__Y.reshape(self.__Y.shape[0],1)
            self.__Y = np_utils.to_categorical(self.__Y,len(self.__mapping_table))

            ## 5. Split X and Y
            self.__train_val_test_gen_X_Y = self.__split_data ()



        elif self.__filepath.endswith(".pickle"):
            # Load data from previously processed image datasets and mapping table

            try:
                data = pickle.load(open(self.__filepath, 'rb'))

            except:
                print ("Please provide a valid pickle file.")
                raise

            self.__train_val_test_gen_X_Y = data['data']
            self.__mapping_table = data['mapping_table']

        else:
            # Neither matlab or pickle file is provided
            print ("Please provide a valid matlab or pickle file.")
            raise

    def __load_matlab_file (self):
        # Load data from matlab file (Balanced Dataset)
        return loadmat(self.__filepath).get('dataset')[0][0]

    def __load_mapping_table (self):
        # Target values in dataset are stored as numbers 0 to 46 for 47 classes
        # The respective value is stored as the third value in the dataset array
        # The function converts the mapping table to a python dictionary for later use
        mapping_table = {}
        for code, ascii_no in self.__dataset[2]:
            mapping_table[code] = ascii_no
            
        return mapping_table

    def __process_image_vector (self,image_vector):
        # Perform the following processing:
        # 1. Reshape image vectors to 28 x 28
        # 2. Transpose image
        # 3. Convert data to float 32
        # 4. Normalize data to between 0 and 1
        image_vector = image_vector.reshape((28,28)).transpose().astype(np.float32)
        image_vector /= 255.
        
        return image_vector

    def __consol_data_from_dataset (self):
        # Load from data set all samples
        # Consolidate all original training and testing sets as one set of data
        X = []
        Y = []
        
        training_set = self.__dataset[0][0][0]
        testing_set = self.__dataset[1][0][0]
        
        for image in training_set[0]:
            X.append(self.__process_image_vector(image))
        for target in training_set[1]:
            Y.append(target[0])
        for image in testing_set[0]:
            X.append(self.__process_image_vector(image))
        for target in testing_set[1]:
            Y.append(target[0])

        return X, Y

    def __remove_digit_classes (self):
        # Remove data that belongs to a digit class
       
        # Digits 0 to 9 are of Ascii values 48 to 57
        digit_classes = [j for j in range(48,58)]

        # New list is created to avoid messing up of indices in the original list
        non_digit_X = []
        non_digit_Y = []

        for i in range(len(self.__X)):
            if self.__Y[i] not in digit_classes:
                non_digit_X.append(self.__X[i])
                non_digit_Y.append(self.__Y[i])
                
        return non_digit_X, non_digit_Y

    def __adjust_Y(self):
        # Adjust Y prediction classes since digit classes have been removed
        # The adjustment table is returned in order to map the orginal classes to the new classes
        # The adjustment list is on the other hand for mapping the new classes to the original classes
        # A new array is also returned for Y

        adjustment_table = {}
        adjustment_list = []
        count = 0
        Y = []
        for i in range(len(self.__Y)):
            if self.__Y[i] not in adjustment_table:
                adjustment_table [self.__Y[i]] = count
                adjustment_list.append(self.__Y[i])
                count += 1
            Y.append(adjustment_table.get(self.__Y[i]))
        
        return adjustment_table, adjustment_list, np.array(Y)

    def __split_data (self):
        """
        # This splitting is for producing better gan_image_generator results

        # Split data according to specified proportions
        # Training set: 80% of 80% of 50%
        # Validation set: 20% of 80% of 50%
        # Testing set: 20% of 50%
        # Image Generator set: 50% of Total
        rem_X, gen_X, rem_Y, gen_Y = train_test_split (self.__X,self.__Y,test_size=0.5)
        train_X, test_X, train_Y, test_Y = train_test_split (rem_X,rem_Y,test_size=0.2)
        train_X, val_X, train_Y, val_Y = train_test_split (train_X,train_Y,test_size=0.2)
        """
        
        # This splitting is for basic_image_generator

        # Split data according to specified proportions
        # Training set: 80% of 80%
        # Validation set: 20% of 80%
        # Testing set: 50% of 20%
        # Image Generator set: 50% of 20%
        train_X, test_X, train_Y, test_Y = train_test_split (self.__X,self.__Y,test_size=0.2)
        train_X, val_X, train_Y, val_Y = train_test_split (train_X,train_Y,test_size=0.2)
        test_X, gen_X, test_Y, gen_Y = train_test_split (test_X,test_Y,test_size=0.5)

        return np.array([train_X, train_Y, val_X, val_Y, test_X, test_Y, gen_X, gen_Y])

    def get_mapping_table (self):
        # Get the mapping table from the data_parser object
        return self.__mapping_table

    def get_train_val_test_gen_X_Y (self):
        # Get the training, validation, testing, image generator datasets from the data_parser object
        return self.__train_val_test_gen_X_Y

    def get_adjustment_table(self):
        # Get the adjustment table that maps the original Y classes to the adjusted Y classes
        return self.__adjustment_table 

    def get_adjustment_list(self):
        # Get the adjustment list that maps the adjusted Y classes to the original Y classes
        return self.__adjustment_list

    def save_data (self,filepath):
        # Save the training, validation, testing, image generator datasets from the data_parser object to a file
        data = {'data':self.__train_val_test_gen_X_Y,'mapping_table':self.__mapping_table}
        pickle.dump(data, open(filepath, 'wb'))
        print("File saved to {0}.".format(filepath))