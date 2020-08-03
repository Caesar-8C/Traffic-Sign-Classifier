# Traffic Sign Classifier

![random sign image][image0]

This project is a neural network that performs classification on a dataset of
German traffic signs.

[image0]: ./imgs/14.png
[image1]: ./imgs/CNN.png
[image2]: ./imgs/12563.png
[image3]: ./imgs/21.png
[image4]: ./imgs/30.png
[image5]: ./imgs/10273.png
[image6]: ./imgs/38.png

## Prerequisites

* python3
* numpy
* glob
* tensorflow
* pickle
* scipy
* cv2

## Project Structure

* augmenter.py: augments training data
* main.py: creates, trains and saves the model
* test.py: analyzes model performance using test data

trafsignsDataset/classes: contains examples of all signs presented in the dataset
trafsignsDataset/myImgs: images I took myself to test the model
models/: contains trained model that was tested

## Project Description

### Data Augmentation

To train a neural network one needs a large dataset and no dataset is large enough.
In order to increase the number of training samples we use data augmentation.

It is done by adding for each sample from the dataset two modified samples:
* randomized saturation, using `tf.image.random_saturation`
* randomized brightness, using `tf.image.random_brightness`

When training model on augmented data resulting accuracy is approximately 0.5% higher.

### Data Preparation

For the network to work stable with images, 0 to 255 integer values for color were
rescaled to be floats from -1 to 1.

### Model Structure

Layers:
* Input Layer
* Convolution Layer with 32 filters and kernel size 5
* MaxPooling Layer with pool size 2x2
* Convolution Layer with 32 filters and kernel size 3
* MaxPooling Layer with pool size 2x2
* Dense Layer with 512 neurons
* Dense Layer with 256 neurons
* Output Dense Layer with 43 neurons

![model schematic][image1]

Trial and error method was used to come up with this model.
It showed the highest accuracy on the test set.

### Hyperparameters

Following parameters were chosen:
* Learning rate: 0.0001
* Number of epochs: 30
* Batch size: 32
* Activation function: RELU
* Loss: Sparse Categorical Crossentropy

### Testing

Example of wrong classification:

![image 12563][image2]
![class 21][image3]
![class 30][image4]

```
===================================
i:  12563 , label:  30, guess:  21
1 th guess is 21, prob: 0.47
2 th guess is 30, prob: 0.36
3 th guess is 11, prob: 0.07
4 th guess is 23, prob: 0.07
5 th guess is 19, prob: 0.02
===================================
```
Example of correct classification:

![image 10273][image5]
![class 38][image6]

```
===================================
i:  10273 , label:  38, guess:  38
1 th guess is 38, prob: 1.0
2 th guess is 34, prob: 0.0
3 th guess is 36, prob: 0.0
4 th guess is 13, prob: 0.0
5 th guess is  9, prob: 0.0
===================================
```

Total accuracy:
```
Test accuracy:  93.14 %
```

Precision and Recall for all classes:
```
Class   0  P =  0.96 R =  0.77
Class   1  P =  0.89 R =  0.98
Class   2  P =  0.91 R =  0.98
Class   3  P =  0.94 R =  0.94
Class   4  P =  0.96 R =  0.95
Class   5  P =  0.93 R =  0.9
Class   6  P =  0.99 R =  0.78
Class   7  P =  0.97 R =  0.83
Class   8  P =  0.9  R =  0.92
Class   9  P =  0.98 R =  0.97
Class  10  P =  0.97 R =  0.99
Class  11  P =  0.86 R =  0.94
Class  12  P =  1.0  R =  0.99
Class  13  P =  0.98 R =  0.99
Class  14  P =  1.0  R =  0.92
Class  15  P =  0.99 R =  0.93
Class  16  P =  0.99 R =  0.99
Class  17  P =  0.99 R =  0.99
Class  18  P =  0.9  R =  0.83
Class  19  P =  0.8  R =  0.98
Class  20  P =  0.94 R =  0.92
Class  21  P =  0.75 R =  0.66
Class  22  P =  0.92 R =  0.88
Class  23  P =  0.82 R =  0.95
Class  24  P =  0.71 R =  0.68
Class  25  P =  0.91 R =  0.89
Class  26  P =  0.77 R =  0.8
Class  27  P =  0.58 R =  0.5
Class  28  P =  0.97 R =  0.89
Class  29  P =  0.83 R =  0.97
Class  30  P =  0.98 R =  0.52
Class  31  P =  0.9  R =  0.96
Class  32  P =  0.67 R =  1.0
Class  33  P =  0.96 R =  0.97
Class  34  P =  0.77 R =  0.99
Class  35  P =  1.0  R =  0.95
Class  36  P =  0.97 R =  0.96
Class  37  P =  1.0  R =  0.98
Class  38  P =  0.97 R =  0.97
Class  39  P =  0.93 R =  0.96
Class  40  P =  0.89 R =  0.94
Class  41  P =  0.82 R =  0.9
Class  42  P =  0.99 R =  0.88
```
Precision and recall show that class 27 (a person in a triangle) was the hardest to classify correctly.

Total accuracy on own images:
```
Test accuracy:  80.0 %
```
Bicycle on image 02 and 5 on image 09 were not classified correctly.


### Problems and Solutions

* The network does not analyze traffic signs, it analyzes 32x32 images of signs.
Which means the prediction is affected by the background. A solution could be to use
another network for sighn detection that would find patterns like circles, squares and
triangles and send only the region of interest to the classification network.