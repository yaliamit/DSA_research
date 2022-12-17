import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
from torch.autograd import Variable
import numpy as np
from collections import Counter
import time
import torch
from lib import average_VAE_loss, find_code
import netw
import pylab as py
import sys
from utils import get_args


# ==============================================================================

def test_model(net, loader_list, device):
    net.eval()
    running_se=0
    cc=0
    for cls in range(10):
        print(cls)
        running_se_cl=0
        cls_c=0
        for i, data in enumerate(loader_list[cls], 0):
            train = data.to(device)
            mu_code = Variable(torch.zeros(data.shape[0], args.latent_space_dimension).to(device))
            log_sigma_code = Variable(torch.zeros(data.shape[0], args.latent_space_dimension).to(device))
            mask = torch.ones(args.image_size).to(device)

            mu_code_out, log_sigma_code_out, _ = find_code(
                train,
                net,
                None,
                mu_code,
                log_sigma_code,
                None,
                cls,
                mask,  # mask
                args.sigma,
                device,
                iternum=args.test_latent_iternum,
                lr=args.lr_latent_codes,
                shifts=False
            )


            # optimize the network parameters
            output, _ = net(mu_code_out, cls)
            loss, mse = average_VAE_loss(
                mu_code_out, log_sigma_code_out, train, output, args.sigma, mask, device
            )

            running_se_cl += mse.item() * data.shape[0] * (2 * args.sigma * args.sigma)
            cls_c+=data.shape[0]
        if not args.silent:
                ii=np.random.choice(range(data.shape[0]),10)
                py.figure(figsize=(10,2))
                for j,i in enumerate(ii):
                    py.subplot(2,10,j+1)
                    py.imshow(data[i].reshape((50,50,3)))
                    py.axis('off')
                    py.subplot(2,10,10+j+1)
                    py.imshow(output[i].detach().cpu().numpy().reshape((50,50,3)))
                    py.axis('off')
                #py.show()
                py.savefig('recons_'+str(cls))
        cc+=cls_c
        print('mse',cls, running_se_cl / cls_c)
        running_se+=running_se_cl
    print('mse', running_se /cc )


def train_model(net, cls_size, train_loader_list):
    mu_code = []
    log_sigma_code = []
    for cls in range(10):
        mu_code.append(torch.zeros(cls_size[cls], args.latent_space_dimension).to(device))
        log_sigma_code.append(torch.zeros(cls_size[cls], args.latent_space_dimension).to(device))

    start = time.time()

    # decoder optimizer
    optimizer = torch.optim.Adam(
        net.parameters(),
        lr=args.lr_decoder
    )

    for epoch in range(args.n_epochs):

        running_se = 0.0
        for cls in range(10):
            for i, data in enumerate(train_loader_list[cls], 0):
                # send training data to device
                train = data.to(device)
                if type(train) is list:
                    train0=train[0]
                    train1=train[1]
                else:
                    train0=train
                    train1=train
                # zero the gradient of the optimizer
                optimizer.zero_grad()

                mask = torch.ones(args.image_size).to(device)

                if args.latent_opt:
                    # optimize latent embeddings
                    mu_code1 = Variable(mu_code[cls][(i * args.batch_size):(i * args.batch_size + args.batch_size), :])
                    log_sigma_code1 = Variable(
                        log_sigma_code[cls][(i * args.batch_size):(i * args.batch_size + args.batch_size), :])

                    iternum=args.latent_iternum
                    if args.continue_training and epoch==0:
                        iternum=args.test_latent_iternum
                    mu_code1, log_sigma_code1, _ = find_code(
                        train0,
                        net,
                        None,
                        mu_code1,
                        log_sigma_code1,
                        None,
                        cls,
                        mask,  # mask
                        args.sigma,
                        device,
                        iternum=iternum,
                        lr=args.lr_latent_codes,
                        shifts=False
                    )

                    mu_code[cls][(i * args.batch_size):(i * args.batch_size + args.batch_size), :] = mu_code1
                    log_sigma_code[cls][(i * args.batch_size):(i * args.batch_size + args.batch_size),
                    :] = log_sigma_code1

                    # optimize the network parameters
                    output, _ = net(mu_code1, cls)

                    # log_reconst_loss = log_reconstruction_loss(train, output, sigma, mask, device)
                    # loss = log_reconst_loss.mean()

                    # use average VAE loss

                    loss, mse = average_VAE_loss(
                        mu_code1, log_sigma_code1, train1, output, args.sigma, mask, device
                    )
                else:
                    # no optimization of the latent codes, just train the entire model

                    # run through the encoder
                    # output, input, mu, log_var = net.forward(train, cls)

                    # ===============
                    # reshape input into 3 channels for encoder

                    # resize to get channels and permute
                    train_dims = train.view(-1, 50, 50, 3)
                    train_dims = train_dims.permute(0, 3, 1, 2)

                    output, input, mu, log_var = net.forward(train_dims, cls)
                    # ===============

                    # optimize the network parameters using VAE loss
                    loss, mse = average_VAE_loss(
                        mu, log_var, train, output, args.sigma, mask, device
                    )

                loss.backward()
                optimizer.step()
                # update running loss
                running_se += mse.item() * data.shape[0] * (2 * args.sigma * args.sigma)

        print(f"epoch: {epoch + 1}")
        # every 50 epochs save the model state
        if epoch in [100, 200, 300, 400]:
            save_path = final_model_path + str(epoch)
            print(f"saving model state at {save_path}")
            torch.save(net.state_dict(), save_path)

        print('training mse', running_se / args.n_training_images, 'loss', loss.item())
        # valid_mse = validation_mse(net, iternum = min(50*(epoch+1), 2000))
        sys.stdout.flush()
    print(f"finished training in {round(time.time() - start)} seconds")
    print(f"saving final model state at {final_model_path}")
    torch.save(net.state_dict(), final_model_path)

if __name__ == "__main__":


    # parse comand line overrides to config file args
    args = get_args()

    if args.trans_type == 'None':
        args.trans_space_dimension=0

    final_model_path = os.path.join(args.predir,args.model_name)

    n_gpus = torch.cuda.device_count()
    device = torch.device("cuda:"+str(args.gpu) if args.gpu<n_gpus else "cpu")
    print(f"using device: {device}")

    if 'pt' in args.model_name:
        net=netw.decoder_mix( device, args)
        net.to(device)
    # elif 'conv' in args.model_name:
    #     net=models.LargeVanillaVAE(3,10)
    #     net.to(device)
    if args.continue_training:
        net.load_state_dict(torch.load(final_model_path,map_location=torch.device("cpu")))

    # load training data
    # =================================================
    if args.clutter==0:
        print('Loading ',os.path.join(args.predir, 'VAE_data' + args.train_data_suffix + '.npy'))
        VAE_train = np.load(os.path.join(args.predir,'VAE_train'+args.train_data_suffix+'.npy'))[0:args.n_training_images, :] / 255
        VAE_test = np.load(os.path.join(args.predir, 'VAE_train'+args.train_data_suffix+'.npy'))[
                   args.n_training_images:args.n_training_images + args.n_testing_images, :] / 255

    else:
        VAE_train = np.load(os.path.join(args.predir,'VAE_train'+args.train_data_suffix+'.npy'))[0:args.n_training_images, :] / 255
        VAE_train_clutter = np.load(os.path.join(args.predir,'VAE_train_clutter'+args.train_data_suffix+'.npy'))[0:args.n_training_images, :] / 255
        VAE_test = np.load(os.path.join(args.predir, 'VAE_train'+args.train_data_suffix+'.npy'))[
                   args.n_training_images:args.n_training_images + args.n_testing_images, :] / 255
        VAE_test = np.load(os.path.join(args.predir, 'VAE_train_clutter'+args.train_data_suffix+'.npy'))[
                   args.n_training_images:args.n_training_images + args.n_testing_images, :] / 255
    train_gt_boxes = np.load(os.path.join(args.predir,'train_gt'+args.train_data_suffix+'.npy'))[0:int(args.n_training_images / 2), :, :]
    print("training data shape:", VAE_train.shape, train_gt_boxes.shape)
    print("counter", Counter(train_gt_boxes[:, :, 4].reshape(-1)))
    train_gt_boxes = train_gt_boxes[:, :, 4].reshape(-1)
    test_gt_boxes = np.load(os.path.join(args.predir,'train_gt'+args.train_data_suffix+'.npy'))[(args.n_training_images // 2):(args.n_training_images+args.n_testing_images)//2, :, :]
    test_gt_boxes = test_gt_boxes[:, :, 4].reshape(-1)
    # build training loaders for each class
    # =================================================
    train_loader_list = []
    test_loader_list=[]
    cls_size = []
    for cls in range(args.num_class):
        cls_size.append(sum(train_gt_boxes == cls))
        if args.clutter==0:
            trainloader = torch.utils.data.DataLoader(
            VAE_train[train_gt_boxes == cls, :],
            batch_size=args.batch_size,
            shuffle=False,
            #num_workers=2,
            drop_last=False
            )
        else:
            trainloader = torch.utils.data.DataLoader(
                list(zip(VAE_train[train_gt_boxes == cls, :],VAE_train_clutter[train_gt_boxes == cls, :])),
                batch_size=args.batch_size,
                shuffle=False,
                # num_workers=2,
                drop_last=False
            )
        if args.clutter==0:
            testloader = torch.utils.data.DataLoader(
                VAE_test[test_gt_boxes == cls, :],
                batch_size=args.batch_size,
                shuffle=False,
                #num_workers=2,
                drop_last=False
            )
        else:
            testloader = torch.utils.data.DataLoader(
                list(zip(VAE_train[test_gt_boxes == cls, :], VAE_train_clutter[test_gt_boxes == cls, :])),
                batch_size=args.batch_size,
                shuffle=False,
                # num_workers=2,
                drop_last=False
            )
        train_loader_list.append(trainloader)
        test_loader_list.append(testloader)
    print(f"number of training images for each class: {cls_size}")

    # =================================================
    # count number of parameters in the model
    n_params_model = sum(p.numel() for p in net.parameters())
    print(f"number of decoder parameters: {n_params_model}")
    if args.train:
        train_model(net,cls_size, train_loader_list)
        test_model(net, train_loader_list, device)
    else:
        net.load_state_dict(torch.load(final_model_path,map_location=torch.device("cpu")))
        test_model(net, test_loader_list, device)


