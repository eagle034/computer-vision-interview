# computer-vision-interview
Interview questions to ponder related to computer vision.

## Imaging Systems

* rough overview of a camera calibration process 
  
  Geometric camera calibration estimates the parameters of a lens and image sensor of an image or video camera

  Camera parameters include intrinsics, extrinsics, and distortion coefficients. To estimate the camera parameters, you need to have 3-D world points and their       corresponding 2-D image points. You can get these correspondences using multiple images of a calibration pattern, such as a checkerboard. Using the correspondences, you can solve for the camera parameters. 
  
  ![image](https://user-images.githubusercontent.com/47903307/199170111-3ff5b71c-cb0c-4304-972b-fb07a7d3c61a.png)

  After you calibrate a camera, to evaluate the accuracy of the estimated parameters, you can:
  
    Plot the relative locations of the camera and the calibration pattern
    
    Calculate the reprojection errors.
    
    Calculate the parameter estimation errors.
    https://www.mathworks.com/help/vision/ug/camera-calibration.html
* say you have acquired a number of constraints for example during camera calibration process - how do you solve the system of the constraints (e.g. describe gradient method in 3-4 sentences)
* Describe the sources of noise in an image sensor.
* How does a camera see in color?
* Draw the schematic of a 3T pixel architecture.
* List out all of the lens parameters you can think of.

  R1, R2, thickness (t), refractive index (n)
  
* You've got a camera that powers on and takes pictures, but the images it captures are very dim. What could be the potential problems?
* What are the typical voltage levels required to operate an image sensor? What power domains do these voltages supply to?
* Your image is blurry. What are the potential issues?

  Aberrations
  
  Monochromatic - Spherical, Coma, Astigmatism, Field Curvature, Distortion, Defocus, Tilt
  
  Chromatic aberrations
  
  https://www.edmundoptics.com/knowledge-center/application-notes/optics/comparison-of-optical-aberrations/
  https://www.edmundoptics.com/knowledge-center/application-notes/optics/chromatic-and-monochromatic-optical-aberrations/
  
* Explain fixed pattern noise (FPN). Where does it come from? How do you correct for FPN?
* How do you configure an image sensor?
* How does data get out of an image sensor? What are some standards you know of?
* Suppose you have a flash LED that can deliver a high lumen output for a fraction of a second, but no more. How would you synchronize the LED with the start of a frame?

## Mathematics
* what are eigenvalues and eigenvectors
* Implement SQRT(const double & x) without using any special functions, just fundamental arithmetic.


## 3D Image Processing
* what is the camera matrix and what does it consist of *** A camera is a mapping between the 3D world and a 2D image. Camera matrix consists of focal lengths (fx,fy), principal point offset (x0,y0), axis skew(s)  
![image](https://user-images.githubusercontent.com/47903307/199159571-d3c7479c-806d-4512-9182-63bd6734869e.png)
https://ksimek.github.io/2013/08/13/intrinsic/ 
* how to convert the point for 3D world coordinates into 2d image coordinates ? *** x = PX where x = (x,y), X = (X,Y,Z) and generalized pinhole camera model, P = K[R|t]
  
P is required to properly relate world points to image points.
![image](https://user-images.githubusercontent.com/47903307/199160555-cb6dffd0-4a8c-4baf-8eb5-f434130d34ea.png)
https://www.cs.cmu.edu/~16385/s17/Slides/11.1_Camera_matrix.pdf

* How to eliminate corners in a bunch of 3D point cloud?
* How to find shape/object correspondences in 3D point cloud?
* Assuming you have an image and depth points for every pixel around an object, how would you create a 3D model of that object?
* Given n correspondences between n images taken from cameras with approximately known poses, find the position of the corresponding 3D feature point.
* Given a set of coordinates of joints, perform a rotation such that the hips are perpendicular to the camera

## 2D Image Processing
* rough overview of optical flow estimation (again, just 3-4 sentences)
* what are the necessary assumptions during optical flow estimation (answer: constant image brightness, no lens/shutter distortion)
* How would you code up a custom rectangle detector?
* How would you send an encoded message in an image. First encode it, then write code to decode it.
* Write a function to produce the convolution of an image with a filter.
* Implement Canny edge detection using graphs.
* How would you implement a connected components labeling in a binary image
* How would you remove outliers when trying to estimate a flat plane from noisy samples?
* Content-based image retrieval (CBIR), also known as query by image content (QBIC) is implemented how?
* Describe how convolution works. What about if your inputs are grayscale vs RGB imagery? What determines the shape of the next layer?
* How do you rotate an image 90 degrees most efficiently if you don't know anything about the cache of the system you're working on?
* Implement voronoi clustering
* Given a MxN matrix where each element can either be 0 or 1. We need to find the shortest path between a given source cell to a destination cell (connected components).
* Make a histogram of 2 variables.

## Color Science
* Stuff about colorspace transformations and color-based segmentation (esp. talking about YUV/Lab/HSV/etc).
* What is metamerism?
* Why do we have different color spaces?
* What is the color space your eyes see in?

## Detection and Tracking
* How does image registration work? Sparse vs. dense optical flow and so on.
* You're in deep space with a star atlas and a calibrated camera. Find your orientation.
* How do you implement loop closure detection in SLAM?

## Linear Algebra
* x belongs to A's null space then, what does that mean? how to calculate covariance given a set of data?
* Transpose a Matrix. First do it using extra memory, then do it in place.
* Implement a sparse matrix class in C++? Implement a dot-product method on the class.
* Create a function to compute an integral image, and create another function to get area sums from the integral image.

## Detection/Features
* what visual features do I know (SIFT, SURF, ORB, etc)
* how does the detector of the feature X work
* What is a difference between SIFT and SURF?
* How can you detect car through CCTV Camera?
* Write a function that fills a buffer with a rolling set of image object segmentation results
* Implement non maximal suppression as efficiently as you can.

## Metrics
* Why does one use MSE as a measure of quality. What is the scientific/mathematical reason for the same?

## String Manipulation
* write the c function for the atoi function.

## Machine Learning
* describe the functionality and usage of LRN normalization layer in AlexNet
* What is a neural network?
* From a set of loss functions like L2 loss or hinge loss, explain which ones are rotationally invariant, sensitive to rescalings of the input features, and which are convex
* Given a set of standard random forest trees and a set of gradient boosted trees, explain how the model's predictions would change if you remove the first tree, the last tree, or make other changes after training
* Describe back propagation.
* How can you tell if you've underfit or overfit?
* What are Sigmoid and ReLU activations? How do vanishing and exploding gradients factor into both of these activations?
* Given stride and kernel sizes for each layer of a (1-dimensional) CNN, create a function to compute the receptive field of a particular node in the network. This is just finding how many input nodes actually connect through to a neuron in a CNN.

## Clustering
* write code for K nearest neighbor algorithm

  The k-nearest neighbors algorithm, also known as KNN or k-NN, is a non-parametric, supervised learning classifier, which uses proximity to make classifications or   predictions about the grouping of an individual data point. While it can be used for either regression or classification problems, it is typically used as a classification algorithm, working off the assumption that similar points can be found near one another.
  
  Step 1: Calculate Euclidean Distance.
  Step 2: Get Nearest Neighbors.
  Step 3: Make Predictions.
  
## Operating Systems
* What is meant by virtual memory?

## Data Structures
* Write a function to delete a duplicate node within a linked list
* Time complexity of insertion in Linked List vs. Array
* Write a function to create a list for every level within a binary tree
* How can you find 2nd highest number in an array in least time
* add a node to a linked list

## Design Patterns
* What is a singleton? How and when is it used?
*** Singleton pattern is a design pattern which restricts a class to instantiate its multiple objects. It is nothing but a way of defining a class. Class is defined in such a way that only one instance of the class is created in the complete execution of the program. Singleton classes are used for logging, driver objects, caching and thread pool, database connections.   
![image](https://user-images.githubusercontent.com/47903307/199155679-acad0563-d779-41e3-9cbb-a1285ca55b5b.png)

## Software Engineering
* What is the difference between debug and release mode?
* What is the purpose of revision/version control?
* Design a web service that provides computer vision computations on input images. How would you test it? How would you deploy it?
* What is a monkey patch?
* Given an algorithm describe your process to turn it into production worthy code.

## Pandas
* Given a pandas DataFrame with some numeric columns, how to produce counts of data for different category IDs, including with or without extra processing to remove noisy, missing, or outlier data points

## Object Oriented Programming
* What is inheritance? Polymorphism?
** In inheritance, the child class inherits the methods from the parent class. In polymorphism, it is possible to define methods in the child class that have the same name as the methods in the parents class. It is possible to modify in the child class that has been inherited from the parent class.

## Algorithms
* Check whether two words are anagram with each other
* Reverse a bit string
* Reverse a linked list in place.
* There is a building with 100 floors. You are given 2 identical eggs. How do you use 2 eggs to find the threshold floor, where the egg will definitely break from any floor above floor N, including floor N itself.
* How do you get the count of each letter in a sentence?
* Given a list A of objects and another list B which is identical to A except that one element is removed, find the removed element.
* Given a list of integers (positive & negative), write an algorithm to find whether there’s at least a pair of integers that sum up to zero. How would you improve your algorithm’s performance?
* Write a function which handles the placement of Tetris blocks in a tetris game.

## Statistics
* You randomly draw a coin from 100 coins — 1 unfair coin (head-head), 99 fair coins (head-tail) and roll it 10 times. If the result is 10 heads, what is the probability that the coin is unfair?
* You’re about to get on a plane to Seattle. You want to know if you should bring an umbrella. You call 3 random friends of yours who live there and ask each independently if it’s raining. Each of your friends has a 2/3 chance of telling you the truth and a 1/3 chance of messing with you by lying. All 3 friends tell you that “Yes” it is raining. What is the probability that it’s actually raining in Seattle?
* Consider a game with 2 players, A and B. Player A has 8 stones, player B has 6. Game proceeds as follows. First, A rolls a fair 6-sided die, and the number on the die determines how many stones A takes over from B. Next, B rolls the same die, and the exact same thing happens in reverse. This concludes the round. Whoever has more stones at the end of the round wins and the game is over. If players end up with equal # of stones at the end of the round, it is a tie and another round ensues. What is the probability that B wins in 1, 2, …, n rounds?
* You’re at a casino with two dice, if you roll a 5 you win, and get paid $10. What is your expected payout? If you play until you win (however long that takes) then stop, what is your expected payout?

## CUDA
* Implement Viola-Jones in CUDA

## General Programming
* How do you average two integers without overflow?

## Multithreaded Programming
* How do you create concurrent programs that operate on the same data without the use of locks?

## Frameworks/Language-Specific
* Know how to use Open Neural Network Exchange Format. (ONNX)
* Use a python built-in package to manipulate ‘csv’ data.

## Databases
* Given a KPI, choose the right metric, perform ETL. (using SQL/Code)
