import image_generator
import model_collection
import data_parser

class trainer_predictor(object):
	# Trainer predictor class is a high level abstraction for performing model training and image prediction
	def __init__(self,image_data_filepath,seed=42,model_collection_path=None):
		# Get required info from user
		self.__data = data_parser.data_parser(image_data_filepath,seed)
		self.__mapping_table = self.__data.get_mapping_table()
		self.__adjustment_list = self.__data.get_adjustment_list()
		self.__seed =seed

		# Create model_collection instances
		self.__model_collection_obj = model_collection.model_collection(model_collection_path)
		
	def train(self,hyperparameter_layer_filepath,training_parameters_filepath,retrain=False):
		# Train the entire batch of models in one go
		# Based on the hyperparameters in the hyperparameter file
		self.__model_collection_obj.train_models(self.__data,hyperparameter_layer_filepath, training_parameters_filepath,self.__seed,retrain)

	def predict(self,char,image_generator_input="basic"):
		# Create image generator object
		if not hasattr(self, 'image_generator'):
			if image_generator_input == "basic":
				self.image_generator = image_generator.basic_image_generator(self.__data.get_train_val_test_gen_X_Y(),self.__mapping_table,self.__adjustment_list)

			elif image_generator_input == "keras_adversarial":
				self.image_generator = image_generator.gan_image_generator_keras_adversarial(self.__data.get_train_val_test_gen_X_Y(),self.__mapping_table,self.__adjustment_list)

			elif image_generator_input == "keras":
				self.image_generator = image_generator.gan_image_generator_keras(self.__data.get_train_val_test_gen_X_Y(),self.__mapping_table,self.__adjustment_list)

		# Generate image to be predicted
		image = self.image_generator.get_image(char,self.__seed)
		
		# Display image
		try:
			data_parser.image_displayer(image).display_image()
		except:
			# Invalid character is selected
			return

		# Perform prediction given a specified character to predict
		self.__best_model = self.__model_collection_obj.get_best_model()
		try:
			self.__best_model = self.__model_collection_obj.get_best_model()
		except:
			print("You must at least train one model before predicting.")
			return

		print("\nThe performance of this model:\nValidation Score - {0:.2f}%\nTest Set Score - {1:.2f}%".format(self.__best_model[1]*100,self.__best_model[2]*100))

		# Predict the character image
		return self.__best_model[0].predict_image(image,self.__mapping_table,self.__adjustment_list)

	def save_model_collection(self,folder):
		# Save the trained models from the model collection
		self.__model_collection_obj.save_model_collection(folder)

	def load_model_collection(self,folder):
		# Load the trained models from a given folder
		self.__model_collection_obj.load_model_collection(folder)

	def get_results(self,show_hyperparameter):
		# Display results of the trained models in model collection
		self.__model_collection_obj.get_results(show_hyperparameter)
