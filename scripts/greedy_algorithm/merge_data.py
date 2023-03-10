import numpy as np
import os
max_num_objects=7
import sys


test_data_suffix=sys.argv[1]

dn = os.path.dirname(os.path.realpath(__file__))
predir=os.path.join(dn[0:dn.find('research')+len('research')],'shared')
test_gt = np.load(os.path.join(predir, 'test_gt' + test_data_suffix + '.npy'))
test_boxes_gt = np.load(os.path.join(predir, 'test_boxes_gt' + test_data_suffix + '.npy'))


tt_gt=test_gt[:,1:].reshape(-1,7,1)
tt=np.concatenate((test_boxes_gt,tt_gt),axis=2)

np.save(os.path.join(predir,'train_gt'+test_data_suffix),tt)
aa=np.load(os.path.join(predir,'test_data'+test_data_suffix+'.npy'))
np.save(os.path.join(predir,'train_data'+test_data_suffix+'.npy'),aa)
print("hello")