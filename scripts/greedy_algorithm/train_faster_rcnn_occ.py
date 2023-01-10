import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import sys
from configparser import ExtendedInterpolation
import configparser

import torch
import time
import numpy as np
import random
from utils import get_args
from torch.utils.data import DataLoader, Dataset
from collections import Counter
import warnings
warnings.filterwarnings("ignore", category=UserWarning)


# We assume the training data has pairs of objects
class Faster_Rcnn_Dataset(Dataset):
    def __init__(self, img_data, img_gt_boxes, max_num_boxes=2):
        super().__init__()
        self.images = img_data  # data_frame['image_id'].unique()
        self.img_gt_boxes = img_gt_boxes
        self.max_num_boxes=max_num_boxes
        self.transforms = False  # get_transforms(phase)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        im_data = self.images[idx, :]
        gt_boxes = self.img_gt_boxes[idx, :]

        area = (gt_boxes[:, 3] - gt_boxes[:, 1]) * (gt_boxes[:, 2] - gt_boxes[:, 0])
        area = torch.as_tensor(area, dtype=torch.float32)

        # there is only one class
        labels = gt_boxes[:, 4]

        # suppose all instances are not crowd
        iscrowd = torch.zeros((gt_boxes[:, 4].shape[0],), dtype=torch.int64)

        target = {}
        temp_boxes = gt_boxes[:, 0:4]
        temp_labels = np.concatenate((labels.reshape(-1,1),np.array(range(len(gt_boxes))).reshape(-1,1)),1)
       #temp_labelsa = np.concatenate((labels.reshape(2,1), np.array([[0], [1]])), 1)
        # if np.random.randint(0,2,1)[0]==1:
        #     temp_boxes = np.concatenate((temp_boxes[1:2,:], temp_boxes[0:1,:]), 0)
        #     temp_labels = np.concatenate((temp_labels[1:2,:], temp_labels[0:1,:]), 0)
        ii=np.array(range(len(temp_labels)))
        np.random.shuffle(ii)
        temp_boxes=temp_boxes[ii]
        temp_labels=temp_labels[ii]
        target['boxes'] = temp_boxes
        target['labels'] = temp_labels
        target['image_id'] = torch.tensor(idx)
        target['area'] = area
        target['iscrowd'] = iscrowd

        if self.transforms:
            sample = {
                'image': im_data,
                'bboxes': target['boxes'],
                'labels': target['labels']
            }
            sample = self.transforms(**sample)
            image = sample['image']


        return im_data.astype(np.float32).transpose((2, 0, 1)), target, idx

def evaluation(model,dataloader):
    total_loss = 0
    for images, targets, image_ids in dataloader:
        images = list(image.to(device) for image in images)
        targets = [{k: v[i].to(device) for k, v in targets.items()} for i in range(batch_size)]
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        total_loss += losses.item()
    return total_loss


if __name__ == "__main__":

    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    batch_size = 1
    total_train_loss = []
    patience = 10
    trigger_times = 0

    args = get_args()


    if args.run_gpu>=0 and torch.cuda.device_count()>0:
        world_size = 1
        device = torch.device("cuda:"+str(args.run_gpu))
    else:
        device=torch.device("cpu")

    print('Loading ', os.path.join(args.predir, 'train_data' + args.train_data_suffix + '.npy'))
    train_dat, train_gt_boxes = np.load(os.path.join(args.predir, 'train_data'+args.train_data_suffix+'.npy')) / 255, np.load(
        os.path.join(args.predir, 'train_gt'+args.train_data_suffix+'.npy'))

    print('Train data shape:',train_dat.shape, train_gt_boxes.shape)
    train_gt_boxes[:, :, 4][train_gt_boxes[:, :, 4] == 0] = 10
    print('Numbers of each class:', Counter(train_gt_boxes[:, :, 4].reshape(-1)))

    train_num, valid_num = args.cnn_num_train, args.cnn_num_valid
    train_dat, train_gt_boxes, valid_dat, valid_gt_boxes = train_dat[:train_num, :], train_gt_boxes[:train_num, :, :], \
        train_dat[train_num:(train_num + valid_num),:], train_gt_boxes[train_num:(train_num + valid_num),:, :]

    print('train and valid data:', train_dat.shape, train_gt_boxes.shape, valid_dat.shape, valid_gt_boxes.shape)

    traindata = Faster_Rcnn_Dataset(train_dat.reshape(-1, args.test_image_size, args.test_image_size, 3),
                                    train_gt_boxes)
    trainloader = torch.utils.data.DataLoader(traindata, batch_size=batch_size,
                                              shuffle=False) #, num_workers=2)
    validdata = Faster_Rcnn_Dataset(valid_dat.reshape(-1, args.test_image_size, args.test_image_size, 3),
                                    valid_gt_boxes)
    validloader = torch.utils.data.DataLoader(validdata, batch_size=batch_size,
                                              shuffle=False) #, num_workers=2)

    if args.cnn_type == 'occ':
        from mytorchvision.faster_rcnn import FastRCNNPredictor
        from mytorchvision import fasterrcnn_resnet50_fpn
    else:
        from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
        from torchvision.models.detection import fasterrcnn_resnet50_fpn

    print(device)

    model = fasterrcnn_resnet50_fpn(pretrained=False)
    num_classes = args.num_class+1

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    print('number of input features',in_features)

    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    model.to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

    num_epochs = args.n_epochs

    checkpoint_path = os.path.join(args.predir,'fasterrcnn'+'_occ'+args.train_data_suffix+'_'+ str(train_num + valid_num))
    if args.cont_training:
        print('Continuing to train',checkpoint_path)
        model.load_state_dict(torch.load(checkpoint_path,map_location=device)['state_dict'])
    current_valid_loss = evaluation(model, validloader)
    print('validation set current_valid_loss:', current_valid_loss)
    best_valid_loss = current_valid_loss

    for epoch in range(num_epochs):
        print(f'Epoch :{epoch + 1}')
        start_time = time.time()
        train_loss = []
        model.train()
        for images, targets, image_ids in trainloader:

            images = list(image.to(device) for image in images)
            targets = [{k: v[i].to(device) for k, v in targets.items()} for i in range(batch_size)]
            loss_dict = model(images, targets)

            losses = sum(loss for loss in loss_dict.values())
            train_loss.append(losses.item())
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
        epoch_train_loss = np.mean(train_loss)
        total_train_loss.append(epoch_train_loss)
        print(f'Epoch train loss is {epoch_train_loss}')

        # create checkpoint variable and add important data
        checkpoint = {
            'epoch': epoch + 1,
            'train_loss_min': epoch_train_loss,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }

        time_elapsed = time.time() - start_time
        print('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

        # early stopping
        current_valid_loss = evaluation(model, validloader)
        print('validation set current_valid_loss:', current_valid_loss)
        if current_valid_loss > best_valid_loss:
            trigger_times += 1
            print('Trigger Times:', trigger_times)
            if trigger_times >= patience:
                print('Early stopping')
                break
        else:
            best_valid_loss = current_valid_loss
            print('trigger times: 0')
            trigger_times = 0
            # save checkpoint
            torch.save(checkpoint, checkpoint_path)


