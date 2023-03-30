# Greedy Algorithm Code Repository

This repository contains code implementing the algorithm described in http://arxiv.org/abs/2212.05706.  We added code for creating the synthetic data set, code for the greedy algorithm (DSA), code for training the VAE decoder, code for training the faster_rcnn.  The structure of the repo is as follows:




`/shared/':

	contains all created datasets, the trained decoder and the trained faster_rcnn.
	Synthetic data has formats as follows:
		train_data_xxx.npy - 200x200 images with pairs of objects
		train_gt_xxx.npy - Ground truth for these images (boxes and classes and occlusion ordering.
		VAE_train_xxx.npy - Images of single objects extracted from train_data_xxx.npy to train decoder.
		test_data_xxx.npy - 200x200 test data images with multiple objects
		test_gt_xxx.npy - Ground truth labels for test data
		test_boxes_gt_xxx.npy - Ground truth boxes for test data.
		

/scripts/greedy_algorithm/:
	synthetic_dataset.py`: script for creating the synthetic dataset.  Parameter file: ARGS/args_synth.conf. 
				The parameter --train_data_suffix provides the `xxx' in the generated data (see above.)

	greedy_algorithm.py: main greedy algorithm script. Parameter file: ARGS/args_10_300_nonlin.conf contains 
			architecture of decoder, and other parameters for training the decoder and running the greedy_algorithm. The output from greedy_algorithm runs are written in subfolder output/greedy_algo_[timestamp]

	utils.py: utilities used by the greedy algorithm script

	netw.py: create pytorch network model based on lines read from an argument file, used both for VAE and greedy_algortihm.

	train_faster_rcnn.py: train a faster_rcnn with or without occlusion. Parameter file: ARGS/args_frcnn.conf

	mytorchvision/*: Modifications to faster_rcnn to accommodate additional occlusion branch in output

	output/*: output location for greedy algorithm script - saves results to output.

	output/images/* : with option draw=1 greedy algorithm creates images for each step of the algorithm.

	train_vae.py : srcipt to train VAE decoder: Parameter file:ARGS/args_10_300_nonlin.conf

	lib.py: functions for use in VAE training

	ARGS': Parameter files:
		args_default.conf : Some default parameter values (always gets read into any of the scripts)
		args_conv.conf: Parameters for convolutional decoder.	
		args_10_300_nonlin.conf: Parameters for fully connected decoder with 10 dim latent variables and 300 hidden layer.
		args_synth.conf: Parameters for creating synthetic dataset.
		args_frcnn: Parameters for faster_rcnn - number of training and validation data.

	example_run.sh: example shell to 

		create dataset, 
		train VAE decoder, 
		train faster RCNN, 
		run detections with only soft non-maximum suppression, 
		run detections with non-maximum suppression and greedy selection algorithm, 
		run experiment on slightly rotated data.



	


