# **Behavioral Cloning**
## Author: Paul Chun
## Date: 9/13/2017


---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/center.jpg "Center"
[image3]: ./examples/recover1.jpg "Recovery Image"
[image4]: ./examples/recover2.jpg "Recovery Image"
[image5]: ./examples/recover3.jpg "Recovery Image"
[image6]: ./examples/before_flipped.jpg "Normal Image"
[image7]: ./examples/flipped.jpg "Flipped Image"


---
My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 5x5 and 3x3 filter sizes and depths between 24 and 64 (model.py lines 49-54)

The model includes RELU layers to introduce nonlinearity (code line 49,51,52,53,54), and the data is normalized in the model using a Keras lambda layer (code line 47).

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 50).

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 10-16). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 61).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road.

For details about how I created the training data, see the next section.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to generate the lowest training and validation loss possible.

In preprocessing step, I normalized the intensity of image using x/255.0 - 0.5 equation. Then I cropped out non-road parts of the image. Images were cropped from pixel 0 to 70 and 125 to 150, which are background scenery of the track and front bonnet of the car that are not necessary for image training.

My first step was to use a convolution neural network model similar to the network that is published by Nvidia's research team. I thought this model might be appropriate because it did not required many epochs. Using GPU instance from EC2, I found that using LeNet network takes longer and requires more epochs than the Nvidia's network architecture.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting.

To combat the overfitting, I modified the model so that it reduces the gap between the validation and training loss. I set keeping ratio to 50% on Dropout function. As a result, the validation loss globally decreased as the program runs more epochs. Although the validation loss was still higher than the training loss, the loss values were small (around 0.02-0.07) for both of them, and overfitting was no longer an issue.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track. When I was driving the car for training, I drove from edge to center repeatedly to improve the driving behavior in these cases.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road for both track 1 and 2.

#### 2. Final Model Architecture

The final model architecture (model.py lines 46-59) consisted of a convolution neural network with the following layers and layer sizes.

Here is a visualization of the architecture.

My final model consisted of the following layers:

| Layer         		|     Description	        					|
|:---------------------:|:---------------------------------------------:|
| Input         		| 150x320x3 RGB image   							|
| Lambda         		| input/255.0 - 0.5 (normalize)  							|
| Cropping         		| Crop vertically:0-70, 125-150, Output: 55x320x3  		|
| Convolution 5x5     	| 2x2 stride 	|
| RELU					|												|
| Drop out					|	Keep prob = 0.5			  						|
| Convolution 5x5	    | 2x2 stride	|
| RELU					|												|
| Convolution 5x5	    | 2x2 stride 	|
| RELU					|												|
| Convolution 3x3	    | 1x1 stride 	|
| RELU					|												|
| Convolution 3x3	    | 1x1 stride 	|
| RELU					|												|
| Flatten					|												|
| Dense		| outputs 100       									|
| Dense		| outputs 50       									|
| Dense		| outputs 10       									|
| Dense		| outputs 1       									|


#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to recover in situation where it goes off the track. These images show what a recovery looks like starting from left to right:
### 1
![alt text][image3]
### 2
![alt text][image4]
### 3
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images thinking that this would add more variety in steering angle. Track 1 has mostly left turn corner, so it gets only trained for left turning situation. Adding flipped image for training would add training for right turning, which will improve the result. For example, here is an image that has then been flipped:

### Before
![alt text][image6]
### After
![alt text][image7]



After the collection process, I had 22233 number of data points. I then preprocessed this data by normalization and cropping the image.

I finally randomly shuffled the data set and put 20% of the data into a validation set.

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 5 as evidenced by the training loss and validation loss output. They were staying around same values after 5th epoch. I used an adam optimizer so that manually training the learning rate wasn't necessary.
