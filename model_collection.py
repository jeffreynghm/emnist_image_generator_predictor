import cnn_model
import ast
import pickle
from operator import itemgetter
from os.path import join, isdir,exists
from os import makedirs
from keras.models import load_model

class model_collection(object):
	def __init__(self,folder=None):
		
		# If folder to existing model_collection is given
		# Then load the model_collection
		if folder != None:
			self.load_model_collection(folder)
		else:
			self.__model_collection = {}

	def __convert_hyperparameter_list(self,hyperparameter_list):
		# Standardize the list of CNN layers and convert them into a standardized tuple for key in model dictionary 
		key = []
		for hyperparameter in hyperparameter_list:
			key.append(self.__convert_hyperparameter(hyperparameter))
		return tuple(key)

	def __convert_hyperparameter (self,hyperparameter):
		# Convert CNN layer into a standardized tuple for key in model dictionary
		# Called by self.__convert_hyperparameter_list for element wise standardization
		if hyperparameter.get('type') == 'conv2d':
			return (hyperparameter.get('type'),
					hyperparameter.get('filters'),
					hyperparameter.get('kernel_size'),
					hyperparameter.get('strides'),
					hyperparameter.get('padding'),
					hyperparameter.get('activation'))

		elif hyperparameter.get('type') == 'maxpooling':
			return (hyperparameter.get('type'),
					hyperparameter.get('pool_size'))

		elif hyperparameter.get('type') == 'dropout':
			return (hyperparameter.get('type'),
					hyperparameter.get('dropout_ratio'))

		elif hyperparameter.get('type') == 'batchnormalization':
			return (hyperparameter.get('type'))

		elif hyperparameter.get('type') == 'flatten':
			return (hyperparameter.get('type'))

		elif hyperparameter.get('type') == 'dense':
			return (hyperparameter.get('type'),
					hyperparameter.get('dense_layer_size'),
					hyperparameter.get('activation'))

		else:
			print("Please provide a valid hyperparameter or layer")
			raise

	def train_models(self, data,hyperparameter_layer_filepath, training_parameters_filepath=None,seed=42,retrain=False):
		# Train models using given hyperparameter list
		# Get user input model hyperparameters from text file
		with open(hyperparameter_layer_filepath,'rt') as f:
			self.__hyperparameter_layers = ast.literal_eval(f.read())

		# Load training_parameters from file if file specified
		# Otherwise, load default parameters
		self.__training_parameters = {'optimizer':'adam',
										'verbose':1,
										'train_batch_size':32,
										'train_epochs':10}

		if training_parameters_filepath != None:
			with open(training_parameters_filepath,'rt') as f:
				self.__training_parameters = ast.literal_eval(f.read())

		# Either create a valid data set if the data_parser_filepath given is a matlab file
		# or load the previously processed data set if the data_parser_filepath given is a pickle file
		self.__data_collection = data

		# Train the CNN models using the given layers of hyperparameters
		# then save the best model based on the performance on the validation set

		for hyperparameter_layer in self.__hyperparameter_layers:
			# Get key for this particular model
			key = self.__convert_hyperparameter_list(hyperparameter_layer)
			
			# If this key exists then skip this model
			# since a model with the specified hyperparameter settings has already been trained

			# Can be overridden by the retrain parameter
			if key in self.__model_collection:
				if retrain == False:
					continue

			# Train the CNN model using the specified hyperparameter settings
			current_model = cnn_model.CNN_model(self.__data_collection.get_train_val_test_gen_X_Y(),hyperparameter_layer,self.__training_parameters)
			current_model_val_score = current_model.get_val_score()[0]
			current_model_test_score = current_model.get_test_score()[0]

			# Add trained model in collection of models 
			self.__model_collection[key] = (current_model,current_model_val_score,current_model_test_score)

			self.save_model_collection("./model/backup")

	def save_model_collection(self,folder):
		# Save model collection to a folder
		mapping_table = {}
		counter = 0

		# Check if folder exist, if not create folder
		if not isdir(folder):
			makedirs(folder)

		# Save each model as a separate h5 file
		for key, value in self.__model_collection.items():
			model, val_score, test_score = value

			filepath = join(folder,"{0}.h5".format(str(counter)))
			model.save_cnn_model(filepath)
			mapping_table[counter] = (key,val_score,test_score)

			counter += 1

		# Store mapping table to a pickle file
		pickle.dump(mapping_table,open(join(folder,"collection_map_table.pickle"),'wb'))

	def load_model_collection(self,folder):
		self.__model_collection = {}
		try:
			# Load the model collection from folder
			mapping_table = pickle.load(open(join(folder,"collection_map_table.pickle"),'rb'))
			for counter, value in mapping_table.items():
				key, val_score, test_score = value
				model = load_model(join(folder,"{0}.h5".format(counter)))
				self.__model_collection[key] = (cnn_model.CNN_model(load_model = model),val_score,test_score)
			print("\nModel collection file loaded.\n")
		except:
			print ("\nPlease provide a valid folder.  Model collection is NOT loaded.  An empty model collection is returned.\n")


	def get_results(self,show_hyperparameter=True):
		# Display the results (and model hyperparameters)
		counter = 0
		for key, value in self.__model_collection.items():
			model, val_score, test_score = value
			
			print("\nModel {0}".format(str(counter)))
			if show_hyperparameter:
				for layer in key:
					print(layer)
			print("    Validation Score - {0:.2f}%\n".format(val_score*100))
			counter += 1

	def get_best_model (self):
		# Get the best model from the collection for prediction purposes
		return max(self.__model_collection.items(),key=lambda x: x[1][1])[1]