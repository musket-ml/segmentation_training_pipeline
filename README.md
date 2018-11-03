# Segmentation Training Pipeline
Research Pipeline for image masking/segmentation in Keras

Idea for this project came from my first attempts to participate in Kaggle competitions. My programmers heart was painfully damaged by looking on my own code as well as on other people kernels. Code was highly repetitive, suffering from numerous reimplementation of same or almost same things through the kernels, model/experiment configuration was oftenly mixed with models code, in other words from programmer perspective it all looked horrible. 

So I decided to extract repetitive things into framework that will work at least for me, and that will follow to this statements: 
 - experiment configurations should be cleanly separated from model definitions.
 - experiment configuration files should be easy to compare, and should fully describe experiment that is being performed except of the dataset
- common blocks like an architecture, callbacks, storing model metrics, visualizing network predictions, should be written one and to be a part of common library
