#!/bin/bash
set -x
# Create the synthetic training set for faster_rcnn with pairs of objects 
# and then use the individual objects from that set for a training set for training the decoder. i
# Also create the detetction test set with 5-7 objects 
#and the validation detection set with 3-4 objects.
python3 synthetic_dataset.py --config_file=args_synth.conf --train_data_suffix=_aoc > out_synth.txt
# Train the decoder on data created above using the network described in args_10_300_nonlin.conf
# Write model parameters to aoc_10_300_nonlin.pt
python3 train_vae.py --config_file=args_10_300_nonlin.conf --train_data_suffix=_aoc --model_name=aoc_10_300_nonlin.pt > out_vae.txt
# Train faster_rcnn with occlusion on data created above -
# write out parameters to fasterrcnn_occ_aoc_5000
python3 train_faster_rcnn_occ.py --config_file=args_frcnn.conf --run_gpu=1 --cnn_num_train=4000 --cnn_num_valid=1000 --train_data_suffix=_aoc --n_epoch=3 > out_frcnn.txt
# Test trained faster_rcnn trained above with just soft_nms - output written to output/greedy*
# Detection threshold is .6
python3 greedy_algorithm.py --config_file=args_10_300_nonlin.conf --faster_rcnn=fasterrcnn_occ_aoc_5000 --soft_nms=1 --detect_thresh=.6 --run_parallel=4 --test_data_suffix=_aoc --process_likelihoods=0 --vae_decoder=aoc_10_300_nonlin.pt > out_gr.txt
# Test trained faster_rcnn trained above with soft_nms 
# and detection selection algorithm - output written to output/greedy*
# Lower detection threshold to avoid false negatives and to show how DSA recovers correct detections.
python3 greedy_algorithm.py --config_file=args_10_300_nonlin.conf --faster_rcnn=fasterrcnn_occ_aoc_5000 --soft_nms=1 --detect_thresh=.25 --run_parallel=4 --test_data_suffix=_aoc --process_likelihoods=1 --vae_decoder=aoc_10_300_nonlin.pt > out_gr_pl.txt
# Implement small rotation on image and a search for rotation parameter in the greedy algorithm also check class 8 when you get class 9
python3 greedy_algorithm.py --config_file=args_10_300_nonlin.conf --faster_rcnn=fasterrcnn_occ_aoc_5000 --soft_nms=1 --detect_thresh=.25 --run_parallel=4 --test_data_suffix=_aoc --process_likelihoods=1 --angle=10 --adjust_rotate=1 --recheck=9,8 --vae_decoder=aoc_10_300_nonlin.pt > out_gr_pl_rot.txt
