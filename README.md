## DL-Digit-comparison
# EE559-Deep Learning (EPFL) - Project 1: "Classification, weight sharing, auxiliary losses"

This project is co-authored by Philip Ngo and David Khatanassian

We explored how different network architectures affect the classification of handwritten digits from MNIST dataset. The learning task was to predict for pairs of images if the first digit is lesser or equal to the second. To accomplish this task we created two main architectures called JointNet and SplitNet. With the JointNet architecture two networks were created:
* WSN
* WOSN


The deep networks are trained on 1k pairs of digits and the goal was to achieve a ~15% error rate. As it is a classification task, we opted for either Cross Entropy Loss or BCE loss for multiple outputs or single output respectively.


Files:
* test.py - Includes all functions and modules, simplified and condensed into a single file. Can be executed to obtain one run of the entire project using default   
* dlc_practical_prologue.py - Create pairs from MNIST data. Provided by François Fleuret (https://fleuret.org/ee559-2018/dlc/)



