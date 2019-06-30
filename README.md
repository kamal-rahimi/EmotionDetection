# Facial Emotion Detection using Convolutional Neaural Networks (CNN)

## Model Description
A face detection/croption layer followed by Convolutional Neaural Networks classifier is used to build the Emotion Reconition model. 

The face area in an input image is first detected using HAAR Classifier. Then, the cropped face area is fed to Convolutional Neural Network.

The Convolutional Neural Network is composed of two blocks of convoution-max_poll-convoution-max_poll with a shortcut convolution. The shortcur convultion is to speed up the training of the network. The first block is followed by a dropout layer and the second block. The output of the second block is followed by a dropout layer, two fully connected layers and a softmax layer.

The activation function of each layer are Exponential Linear Unit (ELU). 

Cross entrhopy is used to measure classification loss and model is trained using Adam optimizer.

The train data is 784 face images from Cohn-Kanade dataset. The test data is another set of 197 face images Cohn-Kanade dataset. Both train and test data are balanced for emotion using oversamling.

The model can predict facial emotion in test data with accuracy of 85%.

The network stucture is depicted below. The notaion (X, Y x Y, S) denotes that the layer has X filters and uses Y x Y kernel with stride S.  

						      Input image
					      		   |
					      |-------------------------|
					      | face detection/croption |
					      |  using HAAR Classifier  |
					      |-------------------------|
							   |
	                     |------------------- Face image (64x64x1) 
			     |				   |
			     |			 |-------------------|
			     |			 | conv2D (16,2x2,1) | 
			     |			 | ACT: ELU          |
	        	     |			 |-------------------|
	        	     |	       			   |
	          	     |			  |------------------|
	           	     |			  | max_poll (2x2,2) |
	           |-------------------|          |------------------|
	           | conv2D (16,4x4,4) | 	       	   |	
	           | ACT: ELU          | 	 |-------------------|
	           |-------------------|	 | conv2D (16,2x2,1) | 
			     |			 | ACT: ELU          |
			     |			 |-------------------|
			     |				   |
			     |			  |------------------|
			     |			  | max_poll (2x2,2) |
			     |			  |------------------|
			     |				   |
			     |			   |----------------|
			     |---------------------|       +        |
						   |----------------|
							   |
						    |-------------|
						    | dropuot 30% |
						    |-------------|
							   |
			     |-----------------------------|  
			     |				   |
			     |			 |-------------------|
			     |			 | conv2D (32,2x2,1) | 
			     |			 | ACT: ELU          |
	        	     |			 |-------------------|
	        	     |	       			   |
	          	     |			  |------------------|
	           	     |			  | max_poll (2x2,2) |
	           |-------------------|          |------------------|
	           | conv2D (64,4x4,4) | 	       	   |	
	           | ACT: ELU          | 	 |-------------------|
	           |-------------------|	 | conv2D (64,2x2,1) | 
			     |			 | ACT: ELU          |
			     |			 |-------------------|
			     |				   |
			     |			  |------------------|
			     |			  | max_poll (2x2,2) |
			     |			  |------------------|
			     |				   |
			     |			   |----------------|
			     |---------------------|       +        |
						   |----------------|
							   |
						    |-------------|
						    | dropuot 50% |
						    |-------------|
							   |
					       |------------------------|
					       | fully-connected (1024) |
					       | ACT: ELU               |
					       |------------------------|
							   |
						    |-------------|
						    | dropuot 50% |
						    |-------------|
							   |
					        |----------------------|
					        | fully-connected (8)  |
					        |----------------------|
							   |
			         |----------------------------------------------------------|
			         |               	softmax                             |
	                         |----------------------------------------------------------|
	                             |     |       |       |      |    |      |       |
				  neutral anger contempt disgust fear happy sadness surprise


## How to use the model

### train_gender.py
Creats a model and trains it based on the Cohn-Kanade image dataset to detect facial emotion in an image.

Example usage:
```
$ python3 train_emotion.py
```
Note a trained model based on 784 face images from Cohn-Kanade dataset is icluded in this repository.


### predict.py
Predict the facial emotion in an input image.

Example usage:
```
$ python3 predict.py -p "./data/test/image.jpg"
```
When no image path is provided, facial emotion is detected for each image file in "./data/test/" directory.

The recognized emotion and model confidence (probabity of the predicted emotion) will be shown in the image.


