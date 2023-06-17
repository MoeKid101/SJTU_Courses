"cifar-10-batches-py" and "data" are intentionally left empty. If you would like to test my code please copy the CIFAR dataset into folder "cifar-10-batches-py".

"models.py" contain the model architectures I used.

"utils.py" contain all the necessary functions including functions for training, testing and generating required data.

"script.py" can be directly runned, generating needed data, train a model and test its performance. The trained models and the confusion matrix will be stored in "temp" folder.

"script_ROS.py" implements random over sampling. It won't run without "script.py" run at least once to generate files in "data" folder.