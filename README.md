## DL-Digit-comparison
# EE559-Deep Learning (EPFL) - Project 1: "Classification, weight sharing, auxiliary losses"

This project was created alongside Philip Ngo and David Khatanassian.

We explored how different network architectures affect the classification of handwritten digits from MNIST dataset. The learning task was to predict for pairs of images if the first digit is lesser or equal to the second. To accomplish this task we created two main architectures called JointNet and SplitNet. Both architectures have dropout and batch normalization implemented as optional features.  The deep networks are trained on 1k pairs of digits and the goal was to achieve a ~15% error rate. As it is a classification task, we opted for either Cross Entropy Loss or BCE loss for multiple outputs or single output respectively.

### Files (alphabetical order)
* **architectures/** - Contains the 4 networks: WSN, WOSN, ClassifyNet and CompareNet. 
* **data** - Contains MNIST data (http://yann.lecun.com/exdb/mnist/).
* dlc_practical_prologue.py - Create pairs from MNIST data. Provided by François Fleuret (https://fleuret.org/ee559-2018/dlc/).
* _errors.py_ - Contains functions for evaluating the accuracy of the test runs.
* **images** - Contains images used in this file. 
* _main.py_
* _test.py_ - An executable with all functions and modules, simplified and condensed into a single file. Used to obtain one run of the entire project using default values. Useful for quick tests or demonstrations.  
* _train.py_ - Contains functions for training the networks and counting errors.

### Project overview

With the JointNet architecture two networks were created:
* WSN: Consists of two stages, digit classification and digit comparison. An auxiliary loss is obtained in the digit classification, which compliments the training. The first stage is a Siamese network and the second stage is a fully connected layer. 
![Image 1](https://github.com/jpruzcuen/DL-Digit-comparison/blob/main/Images/WSN.png)

* WOSN: Both digits enter the CNN at the same time, each with its own weights (weight sharing). Since there is no intermediary digit classification, this network is not benefited by an auxiliary loss. 
![Image 2](https://github.com/jpruzcuen/DL-Digit-comparison/blob/main/Images/WOSN.png)

Lastly, as its name suggets, Splitnet is an architecture where the task is split in two disconnected networks (the output from the classification task is detached from the computational graph). This results in two separate and independent backpropagations, where the losses from the classification network (ClassifyNet) and comparison network (CompareNet) don't influence each other. 

![Image 3](https://github.com/jpruzcuen/DL-Digit-comparison/blob/main/Images/Split.png)






