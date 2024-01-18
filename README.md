# CatAndDogImageClassifier

Project Description:

This project is a machine learning project focused on creating an image classifier using TensorFlow 2.0 and Keras. The goal of the project is to build a convolutional neural network (CNN) that can correctly classify images of cats and dogs. The dataset used for this project contains labeled images of cats and dogs for training and validation, and unlabeled test images for final evaluation.

Objectives:
The specific objectives of this project include:

1.Importing the necessary libraries for machine learning and image processing.

2.Setting up data generators for the training, validation, and test datasets using ImageDataGenerator. This includes converting images to floating-point tensors and rescaling their values to be between 0 and 1.

3.Augmenting the training data by applying random transformations to reduce the risk of overfitting.

4.Creating a CNN model using Keras Sequential API that consists of convolutional layers, max-pooling layers, and a fully connected layer with ReLU activation.

5.Compiling the model by specifying the optimizer, loss function, and accuracy as a metric.

6.Training the model using the fit method on the training data and validating it on the validation data.

7.Visualizing the training and validation accuracy and loss to assess model performance.

8.Using the trained model to make predictions on the test dataset and calculating the probabilities of images being cats or dogs.

9.Plotting the test images along with their predicted probabilities.

Expected Outcome:

The expected outcome of this project is a trained convolutional neural network (CNN) model that can classify images of cats and dogs with an accuracy of at least 63%. The final cell in the notebook will show whether the challenge has been passed, indicating if the accuracy goal has been met. The model's performance can be further improved for extra credit by achieving a 70% or higher accuracy rate.

