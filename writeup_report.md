# **Behavioral Cloning**
---

### Overview

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

[//]: # (Image References)

[image1]: ./examples/architecture.jpg "Model Visualization"
[image2]: ./examples/structure.jpg "Final architecture"
[image3]: ./examples/center.png "Center image"
[image4]: ./examples/center_flip.png "flip image"
[image5]: ./examples/crop.png "cropped image"
---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network
* writeup_report.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

---
### Model Architecture and Training Strategy

#### 1. Model architecture

The model architecture that was used in this project is based on the model developed by NVIDIA described [here](https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf).
This network consists of a normalizing layer followed by 5 convolution layers and 3 fully connected layers.

![alt text][image1]

In order to gauge how well the model was working, The image and steering angle data was split into 85% training and 15% validation set. To reduce the overfitting and also to increase the robustness of the model, a dropout of 30% was used after each fully connected layer. The model uses an Adam optimizer, so the learning rate was not tuned manually. The final architecture of the model looked like below

![alt text][image2]

The model was trained using a mix of data provided by Udacity and data collected by driving the car in the simulator. The data was preprocessed for training process and is described in next section. An EPOCH of 5 was used resulting in mean squared error loss of around 0.021. However it was observed that that loss did not reduce significantly after 3rd EPOCH.
At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 3. Creation of the Training Set & Training Process

The dataset provided by Udacity was tried to begin with and it seemed quite sufficient for training the model to drive the car in required path. However the car was tending to steer late in some corners and thus the dataset was complimented by collecting additional data from simulator focusing steering around the corners.

The dataset contains images from three cameras mounted on the car (center, left and right). Since only steering angle associated to the image from center camera is recorded, a correction factor of 0.2deg was added to center angle to approximately calculate steering angle associated to left and right cameras.

To augment the data set, the images and angles were flipped. For example, here is an image that has then been flipped:

![alt text][image3]
![alt text][image4]

The images from the dataset were preprocessed before using to train the model. The top 70 pixels and bottom 20 pixels of the images were cropped to ignore the unnecessary parts like trees and sky for the model. Here is an example of how cropped image looks

![alt text][image5]

The dataset was finally split to training and validation set and then shuffled randomly while feeding to the CNN model.
