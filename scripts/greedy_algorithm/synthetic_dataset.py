import os
import sys
import platform
import pyvista as pv

if platform.system()=='Linux':
    pv.start_xvfb(3)
else:
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
from collections import Counter
import torch
import torch.nn.functional as F
from utils import find_bb, get_args

import numpy as np
import pylab as plt

n_shapes = 10


def build_objects(label, center, angle, size = 1, mu=np.array([0, 0, 0]), Sigma=3 * np.identity(3)):

    direction = [0,0,1]
    if label == 1:
        height = size
        obj = pv.Sphere(size/2,center=[center[0],center[1],height/2])
    elif label == 2:
        height = size*0.5
        obj = pv.ParametricEllipsoid(size/2,size*0.5/2,size*0.5/2,center=[center[0],center[1],height/2], direction=[1,0,0])
    elif label == 3:
        height = size*0.3
        obj = pv.ParametricTorus(size*0.7/2,height/2,center=[center[0],center[1],height/2], direction=[1,0,0])
    elif label == 4:
        height = size/np.sqrt(2)
        obj = pv.Cube(x_length=size/np.sqrt(2),y_length=size/np.sqrt(2),z_length=size/np.sqrt(2),center=[center[0],center[1],height/2])
    elif label == 5:
        height = size*0.5/np.sqrt(1.25)
        obj = pv.Cube(x_length=height,y_length=size,z_length=height,center=[center[0],center[1],height/2])
    elif label == 6:
        height = size
        obj = pv.Cube(x_length=size*0.5/np.sqrt(1.25),y_length=size*0.5/np.sqrt(1.25),z_length=size,center=[center[0],center[1],height/2])
    elif label == 7:
        height = size/np.sqrt(2)
        obj = pv.Cylinder(radius=size/2,height=height,center=[center[0],center[1],height/2], direction=direction)
    elif label == 8:
        height = size
        obj = pv.Cylinder(radius=size*0.3/2,height=size*np.sqrt(91)/10,center=[center[0],center[1],height/2], direction=direction)
    elif label == 9:
        height = size*0.3
        obj = pv.Cylinder(radius=size*0.3/2,height=size*np.sqrt(91)/10,center=[center[0],center[1],height/2], direction=[0,1,0])
    elif label == 0:
        height = size
        obj = pv.Cone(height=height,radius=size*0.5*np.sqrt(3)/2,center=[center[0],center[1],height/2], direction=direction)
    obj.texture_map_to_plane(inplace=True)
    return obj.rotate_z(angle, point=obj.center, inplace=True)


textures = [None, ]


# get VAE training images from the previous Faster-RCNN training images
def apt(x, u, input_channels, shout, shin):
    """
    :param x: input image batch, dim BxCxHxW
    :param u: translation parameters as B x 2 tensor
    :param input_channels: int
    :param shout: tuple[int] output shape
    :param shin: tuple[int] input shape
    """
    idty = torch.cat((torch.eye(2), torch.zeros(2).unsqueeze(1)), dim=1)
    theta=(torch.zeros(u.shape[0], 2, 3) + idty.unsqueeze(0))
    theta[:, :, 2] = u
    shm = np.max(shout)
    hm = torch.div((torch.tensor([shm, shm]) - torch.tensor(shout)), 2, rounding_mode="floor")
    grid = F.affine_grid(theta, [x.shape[0], input_channels, shm, shm], align_corners=False) #,align_corners=True)
    z = F.grid_sample(x.view(-1, input_channels, shin[0], shin[1]), grid, padding_mode='border', align_corners=False)#,align_corners=True)
    y = z[:, :, hm[0]:(hm[0]+shout[0]), hm[1]:(hm[1]+shout[1])]
    return y


def make_single_images(predir,args):
    train_object0 = np.load(os.path.join(predir,'train_object0'+args.train_data_suffix+'.npy'))
    train_object1 = np.load(os.path.join(predir,'train_object1'+args.train_data_suffix+'.npy'))
    train_gt = np.load(os.path.join(predir,'train_gt'+args.train_data_suffix+'.npy'))
    train_num = np.minimum(5000,train_object1.shape[0])
    train_object0, train_object1, train_gt = train_object0[:train_num, :], train_object1[:train_num,
                                                                                 :], train_gt[:train_num, :, :]
    print('shape', train_object0.shape, train_object1.shape, train_gt.shape)
    print('Counter', Counter(train_gt[:, :, 4].reshape(-1)))

    print(np.mean(train_gt.reshape(-1, 5)[:, 2] - train_gt.reshape(-1, 5)[:, 0]))
    print(np.mean(train_gt.reshape(-1, 5)[:, 3] - train_gt.reshape(-1, 5)[:, 1]))

    output_size = (50, 50)
    u = torch.tensor([[0.0, 0.0]])
    VAE_train = []
    for i in range(train_num):
        if i % 50 == 0: print(i)

        size0 = [train_gt[i, 0, 3] - train_gt[i, 0, 1] + 1,
                 train_gt[i, 0, 2] - train_gt[i, 0, 0] + 1]
        size1 = [train_gt[i, 1, 3] - train_gt[i, 1, 1] + 1,
                 train_gt[i, 1, 2] - train_gt[i, 1, 0] + 1]
        l0, l1 = max(size0), max(size1)

        # cropping the bounding box
        upperleft = [train_gt[i, 0, 1] - (l0 - size0[0]) // 2, train_gt[i, 0, 0] - (l0 - size0[1]) // 2]
        gt0 = train_object0[i, :].reshape(200, 200, 3)[upperleft[0]:(upperleft[0] + l0),
              upperleft[1]:(upperleft[1] + l0),
              :]
        dat0 = apt(torch.tensor(gt0, dtype=torch.float32).unsqueeze(0).permute([0, 3, 1, 2]), u, 3, output_size,
                   (l0, l0)).permute([0, 2, 3, 1]).cpu().numpy().reshape(output_size[0], output_size[1], 3)
        VAE_train.append(np.round(dat0.reshape(-1)))

        upperleft = [train_gt[i, 1, 1] - (l1 - size1[0]) // 2, train_gt[i, 1, 0] - (l1 - size1[1]) // 2]
        gt1 = train_object1[i, :].reshape(args.test_image_size, args.test_image_size, 3)[upperleft[0]:(upperleft[0] + l1),
              upperleft[1]:(upperleft[1] + l1),
              :]
        if gt1.shape[1] < l1:
            print(i)
            gt1 = np.concatenate((gt1, np.zeros((l1, l1 - gt1.shape[1], 3))), axis=1)
        dat1 = apt(torch.tensor(gt1, dtype=torch.float32).unsqueeze(0).permute([0, 3, 1, 2]), u, 3, output_size,
                   (l1, l1)).permute([0, 2, 3, 1]).cpu().numpy().reshape(output_size[0], output_size[1], 3)
        VAE_train.append(np.round(dat1.reshape(-1)))
        sys.stdout.flush()
    VAE_train = np.array(VAE_train).astype(int)
    np.save(os.path.join(predir,'VAE_train'+args.train_data_suffix+'.npy'), VAE_train)


class Env:
    def __init__(self, dim=200):
        self.pl = pv.Plotter(off_screen=True, window_size=(dim,dim))


    def get_image(self, light1,light2,light3,n_object_labels,centers,angles,color):

        self.pl.set_background('black')
        self.pl.set_position([5, 5, 5])
        self.pl.set_focus([1.5, 1, 0.5])
        self.pl.set_viewup([0, 0, 1])
        self.pl.add_light(light1)
        self.pl.add_light(light2)
        self.pl.add_light(light3)

        # add objects to the scene
        objs_centers=[]
        for i in range(len(n_object_labels)):
            obj = build_objects(label=n_object_labels[i], center=centers[i], angle=angles[i])
            objs_centers.append(obj.center)
            self.pl.add_mesh(
                obj, texture=textures[0], color=color
            )
        img = self.pl.screenshot()
        self.pl.clear()
        return img, self.pl.camera_position, objs_centers

def make_random_color():
    color_rand = np.random.uniform(0.0, 1.0, 3)
    while color_rand.sum() < 1.0:
        color_rand = np.random.uniform(0.0, 1.0, 3)
    return color_rand

def get_centers_for_train():
    center0 = np.random.uniform(0.5, 1.0, 2)
    centers = [center0]

    if np.random.choice(2, 1)[0] == 0: theta = np.random.uniform(0.0,np.pi/2)
    else: theta = np.random.uniform(np.pi,3*np.pi/2)
    diff1, diff2 = np.cos(theta), np.sin(theta)
    centers.append(np.array([center0[0]+diff1, center0[1]+diff2]))

    return centers

def make_two_object_scenes(num_train,predir,args,n_objects_scene=2):
    # number of objects
    dim, train_num, cnt_num = (args.test_image_size, args.test_image_size), num_train, 0
    light1, light2, light3 = pv.Light(light_type='scene light',intensity=0.5), \
                             pv.Light(light_type='scene light',intensity=0.5), \
                             pv.Light(light_type='scene light',intensity=0.2)

    train_data, train_object0, train_object1, train_gt = [],[],[],[]
    print('Train num',train_num)
    # generate training data
    EE=Env(dim=args.test_image_size)

    while cnt_num<train_num:
      n_objects_labels = [cnt_num % 10, (cnt_num // 10) % 10]
      color = make_random_color()
      centers=get_centers_for_train()
      angles = np.random.uniform(0, 360, 2)

      light_direc = [np.random.uniform(30,60), np.random.uniform(45,135)]
      light1.set_direction_angle(light_direc[0], light_direc[1])
      light2.set_direction_angle(light_direc[0], light_direc[1]-90)
      light3.set_direction_angle(90, 45)


      img,_,_=EE.get_image(light1,light2,light3,n_objects_labels,centers,angles,color)

      dat,objs,objs_centers,objs_gts = img,[],[],[]
      cnt = (dat.sum(2)>0).sum()

      for i in range(n_objects_scene):

          img,cam,objc=EE.get_image(light1,light2,light3,[n_objects_labels[i]],[centers[i]],[angles[i]],color)
          objs_centers.append(objc[0])
          objs.append(img)
          objs_gts.append(find_bb(img,size=(args.test_image_size,args.test_image_size)))
          objs_gts[i].append(n_objects_labels[i])

      cnt0, cnt1, occlu_seq = (objs[0].sum(2)>0).sum(), (objs[1].sum(2)>0).sum(), ((
          np.array(cam[0])-objs_centers[0])**2).sum()>((np.array(cam[0])-objs_centers[1])**2).sum()

      if (occlu_seq and cnt<=cnt1+200) or (not occlu_seq and cnt<=cnt0+200) or cnt==cnt0+cnt1:
        #print("skip")
        continue
      cnt_num += 1
      train_data.append(dat.reshape(-1))
      if occlu_seq:
        train_object0.append(objs[0].reshape(-1))
        train_object1.append(objs[1].reshape(-1))
        train_gt.append(np.array(objs_gts))
      else:
        train_object0.append(objs[1].reshape(-1))
        train_object1.append(objs[0].reshape(-1))
        train_gt.append(np.array([objs_gts[1],objs_gts[0]]))
      if cnt_num%10==0: print("cnt_num",cnt_num)
      sys.stdout.flush()
    train_data, train_object0, train_object1, train_gt = np.array(train_data),np.array(train_object0),np.array(train_object1),np.array(train_gt)
    np.save(os.path.join(predir,'train_data'+args.train_data_suffix+'.npy'), train_data)
    np.save(os.path.join(predir,'train_object0'+args.train_data_suffix+'.npy'), train_object0)
    np.save(os.path.join(predir,'train_object1'+args.train_data_suffix+'.npy'), train_object1)
    np.save(os.path.join(predir,'train_gt'+args.train_data_suffix+'.npy'), train_gt)



def generate_center():
  while True:
    x,y = np.random.uniform(-0.5,2.0,2)
    if abs(x-y)<1.2:
       return np.array([x,y])


def map_num_objects(rem,valid):
    if valid:
      if rem<=4: return 3
      elif rem<=9: return 4
      else: assert False
    else:
      if rem <= 2:
          return 5
      elif rem <= 5:
          return 6
      elif rem <= 9:
          return 7
      else:
          assert False


def generate_detection_test_data(test_num,predir,args,valid='False'):
    mod=10
    max_objects=7
    dim, test_num, cnt_num = (args.test_image_size, args.test_image_size), test_num, 0
    light1, light2, light3 = pv.Light(light_type='scene light', intensity=0.5), \
                             pv.Light(light_type='scene light',intensity=0.5),\
                             pv.Light(light_type='scene light', intensity=0.2)
    EE = Env(dim=args.test_image_size)
    test_data, test_gt, test_boxes_gt = [], [], []
    # generate test data
    while cnt_num < test_num:
        n_objects_scene = map_num_objects(cnt_num % mod,valid)
        n_objects_labels = np.random.choice(n_shapes, n_objects_scene, replace=True)
        color=make_random_color()
        centers = []
        k = 0
        while k < 1000 and len(centers) < n_objects_scene:
            k += 1
            # don't want new_center to be too close to old_center
            new_center = generate_center()  # np.random.uniform(-0.5,1.0,2)
            for old_center in centers:
                if ((old_center - new_center) ** 2).sum() < 1:
                    break
            else:
                centers.append(new_center)
        if k >= 1000: continue

        angles = np.random.uniform(0, 360, n_objects_scene)
        light_direc = [np.random.uniform(30, 60), np.random.uniform(45, 135)]
        light1.set_direction_angle(light_direc[0], light_direc[1])
        light2.set_direction_angle(light_direc[0], light_direc[1] - 90)
        light3.set_direction_angle(90, 45)

        img, _, _ = EE.get_image(light1, light2, light3, n_objects_labels, centers, angles, color)

        dat, cams, obj_supports, objs_centers, is_valid = img, [], [], [], True

        for i in range(n_objects_scene):
            img, cam, objc = EE.get_image(light1, light2, light3, [n_objects_labels[i]], [centers[i]], [angles[i]],
                                          color)
            obj_supports.append((img.sum(2) > 0).astype(int))
            cams.append(cam)
            objs_centers.append(objc)

        for i in range(n_objects_scene):
            tmp_support = obj_supports[i].copy()
            for j in range(n_objects_scene):
                # the difference between plotter.camera_position[0] and objs_centers[.] gives us occlusion sequence
                if j != i and ((np.array(cams[i][0]) - objs_centers[i]) ** 2).sum() > (
                        (np.array(cams[i][0]) - objs_centers[j]) ** 2).sum():
                    tmp_support = np.maximum(tmp_support - obj_supports[j], 0)
            # if num of visible pixels of an object<=200, is_valid = False
            if tmp_support.sum() <= 200:
                is_valid = False
                break
        if is_valid == False:
            print(cnt_num,'not valid')
            continue

        cnt_num += 1
        test_data.append(dat.reshape(-1))
        tmp_gt = [n_objects_scene]
        for i in range(n_objects_scene): tmp_gt.append(n_objects_labels[i])
        for _ in range(max_objects - n_objects_scene): tmp_gt.append(-1)
        test_gt.append(tmp_gt)
        if cnt_num % 100 == 0: print("cnt_num", cnt_num)

        # ground truth of bounding boxes
        tmp_boxes_gt = []
        for i in range(n_objects_scene):
            tmp_boxes_gt.append(
                find_bb(np.repeat(obj_supports[i].reshape(args.test_image_size, args.test_image_size, 1), 3, 2).reshape(-1), size=(args.test_image_size, args.test_image_size)))
        for _ in range(max_objects - n_objects_scene): tmp_boxes_gt.append([0, 0, 0, 0])
        test_boxes_gt.append(tmp_boxes_gt)
        sys.stdout.flush()
    test_data=np.array(test_data)
    test_gt=np.array(test_gt)
    test_boxes_gt = np.array(test_boxes_gt)
    stradd=args.train_data_suffix
    if valid:
        stradd=stradd+'_valid'

    np.save(os.path.join(predir,'test_data'+stradd+'.npy'), test_data)
    np.save(os.path.join(predir,'test_gt'+stradd+'.npy'), test_gt)
    np.save(os.path.join(predir,'test_boxes_gt'+stradd+'.npy'), test_boxes_gt)


def show_train_data(predir,args):
    train_data = np.load(os.path.join(predir,'train_data'+args.train_data_suffix+'.npy'))
    train_object0 = np.load(os.path.join(predir,'train_object0'+args.train_data_suffix+'.npy'))
    train_object1 = np.load(os.path.join(predir,'train_object1'+args.train_data_suffix+'.npy'))
    train_gt = np.load(os.path.join(predir,'train_gt'+args.train_data_suffix+'.npy'))

    print(train_data.shape, train_object0.shape, train_object1.shape, train_gt.shape)

    print(train_data.shape, train_gt.shape)

    for i in range(5):
        plt.subplot(1,5,i+1)
        plt.imshow(train_data[i,:].reshape(args.test_image_size,args.test_image_size,3))
        plt.axis('off')
        print(train_gt[i, :, :])
    plt.show()

def show_test_data(predir,args,valid=False):
    stradd = args.train_data_suffix
    if valid:
        stradd = stradd + '_valid'
    test_data=np.load(os.path.join(predir,'test_data'+stradd+'.npy'))
    test_gt=np.load(os.path.join(predir,'test_gt'+stradd+'.npy'))
    test_boxes_gt = np.load(os.path.join(predir,'test_boxes_gt'+stradd+'.npy'))
    print(test_data.shape, test_gt.shape, test_boxes_gt.shape)
    print(Counter(test_gt[:,0]))
    for i in range(5):
        plt.subplot(1,5,i+1)
        plt.imshow(test_data[i,:].reshape(args.test_image_size,args.test_image_size,3))
        plt.axis('off')
        print(test_gt[i,:],test_boxes_gt[i,:,:])

    plt.show()

def show_VAE_data(predir,args):
    VAE_train = np.load(os.path.join(predir,'VAE_train'+args.train_data_suffix+'.npy'))
    train_gt_boxes = np.load(os.path.join(predir,'train_gt'+args.train_data_suffix+'.npy'))
    print('shape',VAE_train.shape, train_gt_boxes.shape)
    print('Counter',Counter(train_gt_boxes[:,:,4].reshape(-1)))
    train_gt_boxes = train_gt_boxes[:,:,4].reshape(-1)
    ii=np.random.choice(list(range(len(VAE_train))),36)
    for i in range(0,36):
        plt.subplot(6,6,i+1)
        plt.imshow(VAE_train[ii[i],:].reshape(args.model_size,args.model_size,3)/255)
        plt.axis('off')
        print(train_gt_boxes[i])
    plt.show()


if __name__ == "__main__":

    args=get_args()
    #make_two_object_scenes(args.cnn_num_train+args.cnn_num_valid,args.predir,args,2)
    #make_single_images(args.predir,args)
    generate_detection_test_data(args.num_test_images,args.predir,args,valid=False)

    #generate_detection_test_data(args.num_test_images,args.predir,args,valid=True)

    if args.draw:

        show_test_data(args.predir,args,valid=True)
        show_test_data(args.predir,args,valid=False)
        show_train_data(args.predir,args)
        show_VAE_data(args.predir,args)









