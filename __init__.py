import scipy # Imported before tensorflow to avoid crashing
import numpy # Imported before tensorflow to avoid crashing
import argparse
import trainer_predictor

class init(object):
	def __init__(self,args):
		# Load data from command line
		self.__image_data_filepath = args.image_data_filepath
		try:
			self.__seed = args.seed
		except:
			self.__seed = 42
		try:
			self.__model_collection_path = args.model_collection_path
		except:
			self.__model_collection_path = None

		# Create trainer_predictor object
		self.__trainer_predictor_obj = trainer_predictor.trainer_predictor(self.__image_data_filepath,self.__seed,self.__model_collection_path)
		
		# Initialize parameters for later use
		self.__hyperparameter_layer_filepath = None
		self.__training_parameters_filepath = None
		self.__retrain = None
		self.__output_filepath = None
		self.__load_folder = None

		while True:
				user_input = input("\nWould you like to Train a Model, Generate and Predict a Image, Save Model Collection, Load Model Collection, Display Results, or Exit\n[Train/Predict/Save/Load/Results/Exit]\n")
				
				# User selects to train the models
				if user_input == "Train":
					# Get relevant filepaths from users
					self.__hyperparameter_layer_filepath = self.__get_response_path(self.__hyperparameter_layer_filepath,
																					"Would you like to use previous hyperparameter_layer_filepath? [Y/N]",
																					"Please input the hyperparameter_layer_filepath.")
					self.__training_parameters_filepath = self.__get_response_path(self.__training_parameters_filepath,
																					"Would you like to use previous training_parameters_filepath? [Y/N]",
																					"Please input the training_parameters_filepath.")					
					self.__retrain = self.__get_response_yes_no("Please input whether you like to retrain the models if the same hyperparameter settings are given. [Y/N]")
					
					# Try to train the models with the specified filepaths
					try:
						self.__trainer_predictor_obj.train(self.__hyperparameter_layer_filepath,self.__training_parameters_filepath,self.__retrain)
					except:
						print("\nPlease input valid filepaths.\n")
						continue

				# User selects to save the current model collection
				elif user_input == "Save":
					# Ask user for the folder filepath to save to
					self.__output_filepath = self.__get_response_path(self.__output_filepath,
																		"Would you like to use previous output_filepath? [Y/N]",
																		"Please input the output folder path.")
					# Save file to directory
					try:
						self.__trainer_predictor_obj.save_model_collection(self.__output_filepath)
						print("\nFiles saved to {0}.\n".format(self.__output_filepath))
					except:
						print("\nError occured in saving.  File not saved.\n")

				# User selects to generate a photo and predict the character using the best model available
				elif user_input == "Predict":
					# Ask user which image generator should be used
					self.__image_generator = self.__get_response_image_generator("Which image generator should be used? [basic/keras/keras_adversarial]")
					
					# Ask user what character image to generate
					char = input("\nWhat character would you like to generate image on and make a prediction?"\
								"\nNote that not all characters are available for prediction."\
								"\nUnavailable characters are: c, i, j, k, l, m, o, p, s, u, v, w, x, y and z.\n")
					
					# Give prediction
					prediction = self.__trainer_predictor_obj.predict(char)
					if prediction != None:
						print("\nThis character image is predicted to be:\n{0}".format(prediction))
					else:
						print("\nPlease select a valid character.  Prediction aborted.\n")

				# User selects to load a previously saved model collection
				elif user_input == "Load":
					# Ask user for folder to load the file from
					self.__load_folder = self.__get_response_path(self.__load_folder,
																	"Would you like to use previous output_filepath? [Y/N]",
																	"Please input the folder path to load the model collection from.")
					self.__trainer_predictor_obj.load_model_collection(self.__load_folder)

				# User selects to display current built models
				elif user_input == "Results":
					# Ask user if hyperparameters should be displayed
					self.__show_hyperparameter = self.__get_response_yes_no("Display hyperparameters? [Y/N]")
					self.__trainer_predictor_obj.get_results(self.__show_hyperparameter)

				# User selects to exit the programme
				elif user_input == "Exit":
					break

				else:
					print("\nPlease input a valid command.\n")

	def __get_response_path(self,parameter,question_reuse,question_reuse_no):
		# If path previously exists, prompt user whether or not it should used
		# Otherwise, ask user for a new filepath
		if parameter != None and parameter != "":

			while True:
				filepath_reply = input("\n{0}\n".format(question_reuse))
				if filepath_reply == "N":
					return input("\n{0}\n".format(question_reuse_no))
				elif filepath_reply == "Y":
					return None

		else:
			return input("\n{0}\n".format(question_reuse_no))


	def __get_response_yes_no (self,question):
		# Ask user for input on whether models should be retrained if hyperparameters are the same as a previously trained model
		while True:
			response = input("\n{0}\n".format(question))
			if response == 'Y':
				return True
			elif response == 'N':
				return False

	def __get_response_image_generator (self,question):
		# Ask user for input on whether models should be retrained if hyperparameters are the same as a previously trained model
		while True:
			response = input("\n{0}\n".format(question))
			if response == 'basic' or response == 'keras' or response == 'keras_adversarial' :
				return response

if __name__ == "__main__":
	# Get inputs from user on image data filepath, seed, and model_collection path
	# If no value is received, give the default value
	parser = argparse.ArgumentParser(description='Predict the English character from the given generated character image.')
	parser.add_argument('image_data_filepath',help="This can either be \n 1) Path to EMNIST balanced dataset, or\n2) Path to stored pre-processed image dataset with Training, Validation, Testing, Image Generator sets, and Mapping Table.")
	parser.add_argument('-s','--seed',help="Seed for image generator, data parser and models.")
	parser.add_argument('-m','--model_collection_path',help="Path to a previously trained model collection.")
	args = parser.parse_args()

	init(args)