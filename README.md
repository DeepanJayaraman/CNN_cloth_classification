# CNN_cloth_classification
Classifying cloths using CNN in Tensor flow
The code starts by importing tensorflow_datasets module to load the Fashion MNIST dataset. The dataset is divided into training and test sets. The labels for each of the 10 classes of the images are printed.

The data is preprocessed by normalizing each image by dividing it by 255. This is followed by exploring the data by plotting a few images using matplotlib.

Next, a sequential model is built using the Keras API. The model has an input layer, a hidden layer, and an output layer. The input layer is flattened, and the hidden layer has 128 units with the ReLU activation function. The output layer has 10 units with the softmax activation function. The model is compiled with the Adam optimizer and SparseCategoricalCrossentropy loss function.

The model is trained using the fit method and the training data with 5 epochs and a batch size of 32. The accuracy of the model is checked on the test data.

Lastly, a CNN model is built using Keras API. The model consists of two convolution layers with max-pooling layers in between, a flattened layer, a dense layer with 128 units, and an output layer with the softmax activation function. The model is then compiled and trained on the data similar to the previous model.
