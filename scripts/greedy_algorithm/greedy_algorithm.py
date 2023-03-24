import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import torch
from mytorchvision.faster_rcnn import FastRCNNPredictor
import torch.nn.functional as F
from mytorchvision import fasterrcnn_resnet50_fpn

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.autograd import Variable
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from utils import *

import time
import datetime
import json
import netw
import sys
import cv2
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

from netw import initialize_model, network
STEP=0

# suppress warnings from mesh_grid called by other torch function
warnings.filterwarnings("ignore", category=UserWarning, module="torch.functional")


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def run_torch_fn_mp(fn, world_size, args):
    """Run fn, passing it:
    0) rank
    1) world_size
    3) faster_rcnn_model_path
    4) vae_decoder_path
    """
    mp.spawn(fn,
             args=(world_size, args),
             nprocs=world_size,
             join=True)

# =============================================================================

def test_image(fasterrcnn_model, input_image, device,i0, args):
    """Runs the input image through the fasterrcnn model, printing out the number of predictions,
    the prediction labels, and the occlusion scores.

    :param fasterrcnn_model: (fasterrcnn_resnet50_fpn) faster rcnn model
    :param input_image: (tensor) image in H x W x C
    :param keep: number of bounding boxes to use
    """
    # put model in eval mode
    fasterrcnn_model.eval()
    # run the input image through the model
    input_image_torch = torch.tensor(input_image.transpose((2, 0, 1)), dtype=torch.float32).to(device)
    # run model through model
    temp = fasterrcnn_model([input_image_torch])
    bb = temp[0]['boxes'].cpu().detach().numpy().astype(int)
    pred_labels = temp[0]['labels'].cpu().numpy()
    scores = np.round(temp[0]['scores'].cpu().detach().numpy(), 2)
    occlusion_scores = np.round(temp[0]['occlusion_scores'].cpu().detach().numpy(), 3)
    if args.soft_nms:
        pred_labels, scores, bb, occlusion_scores = soft_nms(pred_labels, scores, bb, occlusion_scores)

    image = input_image
    image = np.ascontiguousarray(image)
    count = 0
    plt.imshow(image)
    fig, ax = plt.subplots(1)
    ax.imshow(image)
    for ii,sc,lb in zip(bb,scores,pred_labels):
        if (sc>args.detect_thresh):
            count += 1
            i=np.int32(np.round(ii))
            rect = patches.Rectangle((i[0], i[1]), i[2]-i[0], i[3]-i[1], linewidth=1, edgecolor = 'white', facecolor = "none")
            ax.add_patch(rect)
            plt.text(i[0],i[1],str(lb),fontsize=12,color='white')

    save_location = os.path.join(args.dn, 'output', 'images', 'ground_truth_' + str(i0))
    plt.axis('off')
    plt.savefig(save_location)





def cbr(output,recon,bb_i,args):
    aa=torch.sum(recon,dim=2,keepdim=True)<args.cover_thresh
    bb=(torch.sum(output,dim=2,keepdim=True)>args.cover_thresh)[bb_i[1]:bb_i[3],bb_i[0]:bb_i[2]]

    recon_out=aa*bb*\
          output[bb_i[1]:bb_i[3],bb_i[0]:bb_i[2]]+\
          (~(aa*bb))*recon
    return recon_out

def compute_update_of_full_output(output, recon, upperleft, args):
    for j1 in range(recon.shape[0]):
        for j2 in range(recon.shape[1]):
            if 0 <= upperleft[0] + j1 < args.test_image_size and 0 <= upperleft[1] + j2 < args.test_image_size and sum(
                    recon[j1, j2, :]) > args.cover_thresh > sum(
                output[upperleft[0] + j1, upperleft[1] + j2, :]):
                output[upperleft[0] + j1, upperleft[1] + j2, :] = recon[j1, j2, :]

def compute_detection_visible_support(recon,output,upperleft,args):
    count1, count2 = 0, 0
    for j1 in range(recon.shape[0]):
        for j2 in range(recon.shape[1]):
            if 0 <= upperleft[0] + j1 < args.test_image_size and 0 <= upperleft[1] + j2 < args.test_image_size \
                    and sum(recon[j1, j2, :]) > args.cover_thresh:
                if sum(output[upperleft[0] + j1, upperleft[1] + j2, :]) < args.cover_thresh:
                    count1 += 1
                count2 += 1
    return count1,count2


def apt(x, u, input_channels, target_size, model_size, device):
    """
    :param x: input image batch, dim BxCxHxW
    :param u: translation parameters as B x 2 tensor
    :param input_channels: int
    :param shout: tuple[int] output shape
    :param shin: tuple[int] input shape
    """
    idty = torch.cat((torch.eye(2), torch.zeros(2).unsqueeze(1)), dim=1)
    theta=(torch.zeros(1, 2, 3) + idty.unsqueeze(0)).to(device)
    theta[:, :, 2] = u[0]
    theta[0,0,0]=torch.cos(u[2])
    theta[0,1,1]=theta[0,0,0]
    theta[0,1,0]=torch.sin(u[2])
    theta[0,0,1]=-theta[0,1,0]
    theta[0,0,0]*=(1.+u[1])
    theta[0,1,1]*=(1.+u[1])

    target_m = np.max(target_size)
    hm = torch.div((torch.tensor([target_m, target_m]) - torch.tensor(target_size)), 2, rounding_mode="floor")
    grid = F.affine_grid(theta, [x.shape[0], input_channels, target_m, target_m], align_corners=False) #,align_corners=True)
    z = F.grid_sample(x.view(-1, input_channels, model_size[0], model_size[1]), grid, padding_mode='zeros', align_corners=False)#,align_corners=True)
    y = z[:, :, hm[0]:(hm[0]+target_size[0]), hm[1]:(hm[1]+target_size[1])]
    return y


def find_code_for_target(target, output, upperleft, vae_decoder, cls_label, mask, target_size, device, args, recon_sigma=0.1, iternum=5000):
    """Directly optimize the latent codes associated with `target`,
    using the following sequence:

    (mu, sigma) --> decoder() --> recon_bg() --> loss()

    The output of the decoder is transformed using recon_bg() in order
    so that it may be compared directly with the original image.

    :param target: (tensor) image to optimize latent codes for
    :param output: accumulated existing detections image.
    :param vae_decoder: (torch.nn.module) VAE decoder used to train latent codes
    :param cls_label: (int) class label
    :param mask: (tensor) mask to apply to image
    :param target_size: (tuple) size of target
    :param recon_sigma: (float) sigma value for reconstruction
    :param iternum: number of optimization iterations
    :returns: optimized parameters, negative_kl
    """
    # make sure we backpropagate through latent codes and shift parameters
    mu_code1 = Variable(torch.zeros(1, args.latent_space_dimension)).to(device)
    # shift parameters we are learning
    params_shift = torch.zeros(2)
    params_scale = torch.tensor(0.)
    params_rotate = torch.tensor(torch.pi*args.angle/180.)

    if args.adjust_scale:
        params_scale.requires_grad_(True)
    if args.adjust_rotate:
        params_rotate.requires_grad_(True)


    mu_code1.requires_grad_(True)
    params_shift.requires_grad_(True)
    log_sigma_code1 = Variable(torch.zeros(1, args.latent_space_dimension)).to(device)
    log_sigma_code1.requires_grad_(True)

    # set up optimizers
    optimizer_mu = torch.optim.Adam([mu_code1], lr=args.mu_lr)
    optimizer_params_shift = torch.optim.RMSprop([params_shift], lr=args.shift_params_lr)

    if params_scale.requires_grad:
        optimizer_params_scale = torch.optim.RMSprop([params_scale], lr=args.shift_params_lr)
    if params_rotate.requires_grad:
        optimizer_params_rotate = torch.optim.Adam([params_rotate], lr=10.*args.shift_params_lr)

    optimizer_log_sigma = torch.optim.Adam([log_sigma_code1], lr=args.log_sigma_lr)

    adjustment = 1. #7500 / mask.sum() if args.adj else 1
    target = target.to(device)
    if args.det_background:
        output = output.to(device)
    for j in range(iternum):
        optimizer_mu.zero_grad()
        optimizer_params_shift.zero_grad()
        if params_scale.requires_grad:
            optimizer_params_scale.zero_grad()
        if params_rotate.requires_grad:
            optimizer_params_rotate.zero_grad()

        optimizer_log_sigma.zero_grad()        

        # send params through decoder
        if args.sample:
            z = vae_decoder.reparameterize(mu_code1, log_sigma_code1)
        else:
            z=mu_code1

        recon, _ = vae_decoder(z, cls_label)

        # negative kl term of the loss function
        negative_kl = (torch.ones_like(mu_code1) + 2 * log_sigma_code1 - mu_code1 * mu_code1 - torch.exp(
            2 * log_sigma_code1
        )).sum() / 2

        # reshape image and apply affine transformation
        recon_dim = recon.view(-1, args.model_size, args.model_size, 3)

        if args.check_vae == 0:
            recon = apt(
                recon_dim.permute(0, 3, 1, 2),   # image in BxCxHxW
                (params_shift,params_scale,params_rotate),
                3,
                target_size,
                (args.model_size, args.model_size),
                device
            )
        else:
            recon = recon_dim.permute(0, 3, 1, 2)

        # get image in HxWxC dims
        recon = recon.permute(0, 2, 3, 1)[0]
        if args.det_background:
            recon=cbr(output, recon, upperleft, args)

        mask = mask.view((target_size[0], target_size[1], 3))

        # apply mask to reconstructed image and source
        recon = recon * mask
        test = target * mask

        # compute loss including reconstruction component
        # TODO call vae loss function here
        test_minus_mu = (test - recon)
        loss_code = adjustment * (test_minus_mu * test_minus_mu / (
                2 * recon_sigma * recon_sigma)).sum() - negative_kl

        loss_code.backward(retain_graph=True)

        optimizer_params_shift.step()
        if params_scale.requires_grad:
            optimizer_params_scale.step()
        if params_rotate.requires_grad:
            optimizer_params_rotate.step()
        optimizer_mu.step()
        optimizer_log_sigma.step()

    # return the optimized latent parameters, negative_kl
    return mu_code1, (params_shift, params_scale, params_rotate), negative_kl





def find_largest_iou(i, bb, selected):
    """Finds largest overlap between bounding boxes."""
    largest_j, largest_iou = None, 0
    for j in selected:
        tmp = iou(bb[i], bb[j])
        if tmp > largest_iou:
            largest_j, largest_iou = j, tmp
    return largest_j


def single_reconstruction(vae_decoder, pred_label, target, mask, output, bb_i, device, args, index, back):
    """Given an image with an obfuscation defined by `mask`, attempt to reconstruct the entire image
    by optimizing the image's latent code using the visible component of the image.
    :param vae_decoder: (torch.nn.module) VAE decoder used to train latent codes
    :param pred_label: (int) predicted label of detection
    :param target: (tensor) the possibly image we are trying to reconstruct is based on fastrcnn bounding boxes.
    :param mask: (tensor) occlusion mask - same dims as target.
    :param device: (str) execution device
    :param args: (namespace) script arguments
    :param index: index of detection currently being processed.
    :param back: True if this is a backward pass eliminating a previous detection.
    :return: (np.array) reconstructed image, kl cost of reconstruction
    """
    mask = torch.tensor(mask.reshape(-1))
    local_output=torch.tensor(np.copy(output),dtype=torch.float32)
    # get HxW dimension of output image
    target_size = (target.shape[0], target.shape[1])
    # find the latent mean and shift codes associated with the masked image
    mu_code, params, nkl = find_code_for_target(
        torch.tensor(target, dtype=torch.float32),#.view(1, -1),
        local_output,
        bb_i,
        vae_decoder,
        pred_label,
        mask.to(device),
        target_size,
        device,
        args,
        iternum=args.test_latent_iternum,
    )

    # reconstruct the image out of the latent mean
    recon, _ = vae_decoder(mu_code, pred_label)


    # apply the affine transformation to scale the reconstruction to the size of the detection
    recon_dim = recon.view(-1, args.model_size, args.model_size, 3)

    if args.check_vae == 0:
        recon = apt(
            recon_dim.permute(0, 3, 1, 2),
            params,
            3,
            (max(target_size), max(target_size)),
            (args.model_size, args.model_size),
            device
        )
    else:
        recon = recon_dim.permute(0, 3, 1, 2)


    # flatten back down into ((HxW) x 3), like recon_bg output
    recon = recon[0].permute(1, 2, 0).cpu()
    #recon = cbr(local_output, recon, bb_i, args).cpu()

    if args.draw:
        global STEP
        save_location = os.path.join(args.dn,'output','images','STEP'+str(STEP)+'_target_'+str(index)+'_'+str(back))
        plt.imshow(target.reshape((target_size[0],target_size[1],3))*mask.detach().numpy().reshape((target_size[0],target_size[1],3)))
        plt.savefig(save_location)
        save_location = os.path.join(args.dn,'output','images','STEP'+str(STEP)+'_single_reconstruction_b_'+str(index)+'_'+str(back))

        STEP+=1
        plt.imshow(recon.detach().numpy())
        plt.savefig(save_location)



    # return reconstructed image
    return recon.detach().numpy(), nkl

def compute_det_background(temp_selected,recons,bb, args):
        output_bg = np.zeros((args.test_image_size,args.test_image_size, 3))
        for i in range(len(temp_selected)-1):
            ind=temp_selected[i]

            bb_i = bb[ind]
            target_size = (bb_i[3] - bb_i[1], bb_i[2] - bb_i[0])
            upperleft = [bb_i[1] - (max(target_size) - target_size[0]) // 2,
                         bb_i[0] - (max(target_size) - target_size[1]) // 2]
            compute_update_of_full_output(output_bg, recons[ind][0], upperleft, args)
        return output_bg

def find_mask(bb,target_size,output,args):

    mask = np.ones((target_size[0], target_size[1], 3))
    for x in range(bb[1],bb[3]):
        for y in range(bb[0],bb[2]):
            if np.sum(output[x,y,:])>args.occlusion_thresh:
                mask[x-bb[1],y-bb[2]]=0

    return mask


def whole_reconstruction(vae_decoder, image, output_size, recons, temp_selected, bb, occlusion_scores, pred_labels,
                         device, args, img_index, done_recon, back=False):
    """Given a set of occluded individual component selections temp_selected and a composite image,
    reconstruct each of these selections into the unoccluded full components.
    :param vae_decoder: (torch.nn.module) VAE decoder used to train latent codes
    :param image: (tensor) image to reconstruct
    :param output_size: (tuple[int]) size of output image, same as size of input image (default 200x200).
    :param recons: (list[(tensor, float)]) individually reconstructed images and nkls
    :param temp_selected: current temporarily chosen individual reconstructions
    :param bb: (list[tuple]) bounding boxes of individual reconstructions
    :param occlusion_scores: (list[float]) occlusion scores of individual reconstructions
    :param pred_labels: (list[int]) labels of individual component reconstructions
    :returns: (tensor, list[tensor]) the reconstructed whole image and an array of the singly
        reconstructed components, and the kl cost of latest reconstruction added to the reconstruction list.
    """
    loc_index_list=[]
    output = np.zeros((output_size[0], output_size[1], 3))
    # Make background


    output_bg=compute_det_background(temp_selected, recons, bb, args)
    # occlusion scores of objects selected
    occ_score_selected = [occlusion_scores[i] for i in temp_selected]
    # get the indices of the sorted occlusion scores in descending order
    # i.e. the first element in sequence is the index of the highest occlusion score in occ_score_selected
    sorted_occ_score_locations = np.argsort(occ_score_selected)[::-1]
    for occ_score_index in sorted_occ_score_locations:
        # get the selection index of the component with the ranked occlusion score
        index = temp_selected[occ_score_index]
        bb_i = bb[index]
        target_size = (bb_i[3] - bb_i[1], bb_i[2] - bb_i[0])
        upperleft = [bb_i[1] - (max(target_size) - target_size[0]) // 2, bb_i[0] - (max(target_size) - target_size[1]) // 2]
        # Don't do previous reconstructions again.
        if index not in done_recon:
            target = image[bb_i[1]:bb_i[3], bb_i[0]:bb_i[2], :]



            # find occlusion mask
            mask=find_mask(bb_i, target_size, output, args)
            # if we have fewer than args.occlusion_number_of_pixels pixels above the occlusion threshold, no reconstruction
            if ((target * mask).sum(2) > args.occlusion_thresh).sum() < args.occlusion_number_of_pixels:
                recons.append(None)
                #print('Occlusion too large for image',index,pred_labels[index],((target * mask).sum(2) > args.occlusion_thresh).sum())
                return np.zeros((output_size[0], output_size[1], 3)), recons, None
            else:
                single_recon, nkl = single_reconstruction(
                    vae_decoder,
                    pred_labels[index],
                    target,
                    mask,
                    output_bg,
                    bb_i,
                    device,
                    args,
                    index,
                    back
                )

                recons.append((single_recon, nkl))
                index=len(recons)-1

                # Count how many visible pixels in this detection that aren't covered by an occlusion, if too low reject it.
                # We can't use mask because it is adapted to original detection size, and the recon size is the full size of the square box.


                count1,count2=compute_detection_visible_support(recons[-1][0], output, upperleft, args)
                if count2==0:
                    return np.zeros((output_size[0], output_size[1], 3)), recons, None
                elif count1 / count2 < args.visible_thresh or count1 < args.occlusion_number_of_pixels:
                    return np.zeros((output_size[0], output_size[1], 3)), recons, None
        # This reconstruction doesn't contribute anything to the output
        elif recons[index] is None:
            continue
        compute_update_of_full_output(output,recons[index][0],upperleft,args)
        loc_index_list+=[index]
    if args.draw:
        global STEP
        # save reconstructed image
        plt.figure()
        plt.imshow(output.reshape(args.test_image_size, args.test_image_size, 3))
        for ind in loc_index_list:
            plt.text(bb[ind][0],bb[ind][1],pred_labels[ind],fontsize=12,color='white')
        loc_index_list.sort()
        ss='_'.join([str(a) for a in loc_index_list])
        save_location = os.path.join(args.dn,'output','images','STEP'+str(STEP)+'_whole_reconstruction_'+ss+'_'+str(back))
        STEP+=1
        plt.axis('off')
        plt.savefig(save_location)
    return output, recons, nkl


def detections_selection(vae_decoder, image, scores, bb, occlusion_scores, pred_labels, device, args, img_index):
    """Main entry point for the detections selection algorithm.
    :param vae_decoder: (torch.nn.module) VAE decoder used to train latent codes
    :param image: (tensor) image to run the algorithm on
    :param scores: (list[float]) detection confidence scores
    :param bb: (list[tuple[int]]) estimated bounding boxes
    :param occlusion_scores: (list[float]) occlusion scores of detections
    :param pred_labels: (list[int]) predicted labels of detections
    :param device: (str) execution device
    :param args: (namespace)
    :param image_index: (int) index of image
    :return: (list[int]) selected objects, lowest_loss
    """
    output_size = (image.shape[0], image.shape[1])
    selected = []
    recons = []
    lowest_loss = float("inf")
    # for each prediction, sorted by quality score...
    done_recon=[]
    for i in range(len(pred_labels)):
        if args.draw:
            print('No.', i, bb[i], pred_labels[i])
        # if the confidence score is low enough, ignore the prediction
        if scores[i] < args.detect_thresh:
            recons.append("NA")
            continue
        if args.process_likelihoods:
            temp_recons=recons.copy()
            temp_selected = selected.copy()
            temp_selected.append(i)
            # run whole reconstruction algorithm on image adding i'th prediction
            reconst_with_i, recons,nkl = whole_reconstruction(
                vae_decoder,
                image,
                output_size,
                recons,
                temp_selected,
                bb,
                occlusion_scores,
                pred_labels,
                device,
                args,
                img_index,
                done_recon
            )
            # compute loss between image and complete reconstruction done using the i'th selection
            if nkl is not None:
                temp_selected_nkl_with_i = sum([recons[box_id][1] for box_id in temp_selected])
                loss_with_i = np.sum((image - reconst_with_i) * (image - reconst_with_i)) + args.lamb * len(
                    temp_selected)
                if args.draw:
                    print('class', pred_labels[i], loss_with_i, loss_with_i-temp_selected_nkl_with_i)
                loss_with_i-=temp_selected_nkl_with_i
            else:
                loss_with_i = torch.tensor(float("inf"))
            if args.recheck[0]!=-2:
                torecheck=[]
                if pred_labels[i]>=0 and pred_labels[i]==args.recheck[0]:
                    torecheck=[args.recheck[1]]
                elif args.recheck[0]==-1:
                    torecheck=list(range(args.num_class)).pop(pred_labels[i])

                if len(torecheck)>0:
                  for j in torecheck:
                    temp_temp_recons=temp_recons.copy()
                    temp_pred_labels=pred_labels.copy()
                    temp_pred_labels[i]=j
                    reconst_with_i_ch, recons_ch, nkl_ch = whole_reconstruction(
                        vae_decoder,
                        image,
                        output_size,
                        temp_temp_recons,
                        temp_selected,
                        bb,
                        occlusion_scores,
                        temp_pred_labels,
                        device,
                        args,
                        img_index,
                        done_recon
                     )
                    if nkl_ch is not None:
                        temp_selected_nkl_with_i_ch = sum([recons_ch[box_id][1] for box_id in temp_selected])
                        loss_with_i_ch = np.sum((image - reconst_with_i_ch) * (image - reconst_with_i_ch)) + args.lamb * len(
                            temp_selected)
                        if args.draw:
                            print('class', j, loss_with_i_ch, loss_with_i_ch-temp_selected_nkl_with_i_ch, loss_with_i)
                        loss_with_i_ch-=temp_selected_nkl_with_i_ch
                    else:
                        loss_with_i_ch = torch.tensor(float("inf"))

                    if loss_with_i_ch < loss_with_i:
                        loss_with_i=loss_with_i_ch
                        pred_labels[i]=temp_pred_labels[i]
                        recons=temp_temp_recons




            # find largest overlapping component with the current component
            j = find_largest_iou(i, bb, selected)
            # if args.draw:
            #     print('most_overlap', j)
            # if the largest overlapping component has been selected...
            loss_i_no_j = torch.tensor(float("inf"))
            if j is not None and j in selected:
                # ...try removing it and re-running the whole reconstruction
                temp_selected.remove(j)
                # Take out the previous reconstruction of i it may be different now.
                temp_recons=recons[0:len(recons)-1]
                #temp_occlusion_scores=np.delete(occlusion_scores,j)
                #temp_recons=list(np.delete(np.array(recons),j))
                reconst_i_no_j, recons, nkl_i_no_j = whole_reconstruction(
                    vae_decoder,
                    image,
                    output_size,
                    temp_recons,
                    temp_selected,
                    bb,
                    occlusion_scores,
                    pred_labels,
                    device,
                    args,
                    img_index,
                    done_recon,
                    back=True
                )
                # check the loss of the reconstructed image after removing overlapping component
                if nkl_i_no_j is not None:
                    temp_selected_nkl_i_no_j = sum([recons[box_id][1] for box_id in temp_selected])
                    loss_i_no_j = np.sum((image - reconst_i_no_j) * (image - reconst_i_no_j)) + args.lamb * len(temp_selected) - temp_selected_nkl_i_no_j

            # compare the losses from different selection configurations and choose the configuration which
            # minimizes the loss
            if loss_with_i < min(lowest_loss, loss_i_no_j):
                selected.append(i)
                done_recon.append(i)
                # update minimum loss
                lowest_loss = loss_with_i
                # if args.draw:
                #     print('***********selected', selected, lowest_loss)
            elif loss_i_no_j < min(lowest_loss, loss_with_i):
                selected.remove(j)
                selected.append(i)
                done_recon.remove(j)
                done_recon.append(i)
                recons=temp_recons
                # update minimum loss
                lowest_loss = loss_i_no_j
            if args.draw:
                    print('****selected', selected, '*****labels', [pred_labels[s] for s in selected],'with i',loss_with_i.item(),'no j',loss_i_no_j.item(),'lowest',lowest_loss.item())
        else:
            selected+=[i]
    return selected, lowest_loss

def soft_nms_Gaussian_penalty(iou1, sigma=0.5):
    return np.exp(-(iou1**2)/sigma)


def soft_nms(labels, scores, bb, occlusion_scores):
    l_pred = len(labels)
    visited = set()
    for i in range(l_pred-1):
      # find the one with max score
      max_ind, max_score = -1, -1
      for j in range(l_pred):
        if j not in visited and scores[j]>max_score:
          max_ind, max_score = j, scores[j]
      visited.add(max_ind)
      # update the scores
      for j in range(l_pred):
        if j not in visited:
          iou_j = iou(bb[max_ind], bb[j])
          Gaussian_penalty = soft_nms_Gaussian_penalty(iou_j)
          scores[j] = scores[j]*Gaussian_penalty
    ii=np.flip(np.argsort(scores))

    return labels[ii], scores[ii], bb[ii], occlusion_scores[ii]

def test_example(i0, bg_test, test_gt, test_boxes_gt, fasterrcnn_model, vae_decoder, device, args):
    """Run the detection selection algorithm on the test image indexed by
    i0.
    :param i0: (int) index of test image
    :param bg_test: all the test images.
    :param test_gt: Ground truth labels
    :param test_boxes_gt: Ground boxes.
    :param fasterrcnn_model: (fasterrcnn_resnet50_fpn) RCNN model used for bounding box prediction
    :param vae_decoder: The vae decoder used.
    :param args: (namespace)
    :return: (list[int]) predicted object indices, in order
    """
    # fetch image
    image = bg_test[i0, :].reshape(args.test_image_size, args.test_image_size, 3).copy()
    #image=cv2.GaussianBlur(image,(5,5),2, 2, cv2.BORDER_CONSTANT)
    if args.big_dim!=0 and args.big_dim != args.test_image_size:
        image=resize_objects(args,image)
    if args.angle > 0:
        (h, w) = image.shape[:2]
        (cX, cY) = (w // 2, h // 2)
        # rotate our image by args.angle degrees around the center of the image
        M = cv2.getRotationMatrix2D((cX, cY), args.angle, 1.0)
        image = cv2.warpAffine(image, M, (w, h))

     #image=add_clutter(image,10,20)
    if args.draw:
        imagec=image.copy()
        test_image(fasterrcnn_model, imagec, device, i0, args)
        # todo parametrize files
        print("ground truth", test_gt[i0])

    fasterrcnn_model.eval().to(device)

    # unpack bounding boxes, prediction labels, scores, occlusion scores
    if args.check_vae==1:
        bb=np.array((0,0,args.model_size,args.model_size)).reshape((1,4))
        pred_labels=test_gt[i0][1].reshape((1,))
        scores=np.array(1.).reshape((1,))
        occlusion_scores=np.array(1.).reshape((1,))
    else:
        temp = fasterrcnn_model([torch.tensor(image.astype(np.float32).transpose((2, 0, 1))).to(device)])
        # unpack bounding boxes, prediction labels, scores, occlusion scores
        bb = temp[0]['boxes'].cpu().detach().numpy().astype(int)
        pred_labels = temp[0]['labels'].cpu().numpy()
        scores = np.round(temp[0]['scores'].cpu().detach().numpy(), 2)
        #occlusion_scores = np.argmax(temp[0],['occlusion_scores'].cpu().detach().numpy())
        occlusion_scores = np.round(temp[0]['occlusion_scores'].cpu().detach().numpy(), 3)
        if args.soft_nms:
            pred_labels, scores, bb, occlusion_scores = soft_nms(pred_labels, scores, bb, occlusion_scores)

    # if args.by_occlusion:
    #     ii=np.argsort(occlusion_scores)
    #     pred_labels=pred_labels[ii]
    #     bb=bb[ii]
    if args.draw:
        print(pred_labels, scores, occlusion_scores)

    # we are labeling class 10 ---> class 0
    for i in range(len(pred_labels)):
        if pred_labels[i] == 10:
            pred_labels[i] = 0

    # run detections selection algorithm on image
    selected, lowest_loss = detections_selection(
        vae_decoder,
        image,
        scores,
        bb,
        occlusion_scores,
        pred_labels,
        device,
        args,
        i0
    )

    # calculate correctness, mAP
    gt0 = test_gt[i0, :]
    gt_boxes = test_boxes_gt[i0, 0:gt0[0], :]
    gt_taken = [0 for _ in range(gt0[0])]
    # all boxes found
    all_boxes = [[] for _ in range(10)]
    corrects = []
    # tally up correct selections
    for j0 in range(len(selected)):
        j1 = selected[j0]
        for j2 in range(gt0[0]):
            if gt0[1 + j2] == pred_labels[j1] % 10 and gt_taken[j2] == 0 and iou(
                    bb[j1], gt_boxes[j2, :]) >= 0.5:
                gt_taken[j2] = 1
                all_boxes[gt0[1 + j2]].append([1, scores[j1]])
                corrects.append(1)
                break
        else:
            all_boxes[pred_labels[j1] % 10].append([0, scores[j1]])
    corrects.append(len(pred_labels) - len(selected) - (gt0[0] - sum(gt_taken)))
    return [pred_labels[box_index] for box_index in selected], corrects


def read_decoder(rank, args, device):

    vae_decoder = netw.decoder_mix(device, args)

    if args.run_gpu>=0 or args.run_parallel:
        map_location = f"cuda:{rank}"
    else:
        map_location = f"cpu"

    vae_decoder.load_state_dict(torch.load(os.path.join(args.predir,args.vae_decoder),
                                           map_location=torch.device(map_location)))
    vae_decoder.eval()


    return vae_decoder.to(device)


def read_fasterrcnn(rank, args, max_obj_num=7):
    """Reads in fasterrcnn model."""
    max_obj_num=2
    faster_rcnn = fasterrcnn_resnet50_fpn(pretrained=False)
    # get number of input features for the classifier
    in_features = faster_rcnn.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    faster_rcnn.roi_heads.box_predictor = FastRCNNPredictor(in_features, args.num_class+1,max_obj_num)
    #params = [p for p in faster_rcnn.parameters() if p.requires_grad]
    #optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    # load check point
    if args.run_gpu>=0:
        map_location = rank
    elif args.run_parallel:
        map_location=f"cuda:{rank}"
    else:
        map_location = f"cpu"

    checkpoint = torch.load(os.path.join(args.predir,args.faster_rcnn), map_location=map_location)
    # initialize state_dict from checkpoint to model
    faster_rcnn.load_state_dict(checkpoint['state_dict'])
    if args.soft_nms:
        faster_rcnn.roi_heads.nms_thresh = 1.0
    # initialize optimizer from checkpoint to optimizer
    #optimizer.load_state_dict(checkpoint['optimizer'])
    faster_rcnn.eval()
    return faster_rcnn.to(rank)


def main(rank, world_size, args):
    """Run the detection selection algorithm on the test images, computing accuracy statistics:
    1) The proportion of images for which we have the correct number of objects detected
    2) The proportion of images for which we have the correct number of each class detected
    """
    # print(f"executing greedy algorithm on device {rank}...")
    if args.run_parallel:
        setup(rank, world_size)

    torch.manual_seed(42)

    bg_test,test_boxes_gt,test_gt=get_data(args)
    max_obj_num = test_boxes_gt.shape[1]

    # ========================================================================

    # read decoder into memory in process executing on rank
    if args.run_parallel:
        vae_decoder = read_decoder(rank, args, rank)
    else:
        vae_decoder = read_decoder(args.run_gpu, args, device)

    # read fasterrcnn into memory in process executing on rank
    fasterrcnn_model = read_fasterrcnn(rank, args, max_obj_num)

    start_time = time.time()

    # separate output file for algorithm running on each device
    output_filename = os.path.join(args.dn,'output','output_process_'+str(rank))

    output_file = open(output_filename, "w")
    correct_num_boxes, correct_labels, total = 0, 0, 0
    # each of the four devices gets a quarter of the images
    n_images_device = int(args.num_test_images / world_size)
    # check if we are running on cpu
    if type(rank) is int:
        img_start = rank * n_images_device
        img_end =   n_images_device + rank * n_images_device
    elif args.image_to_process<0:
        img_start = 0
        img_end = n_images_device
    else:
        img_start=args.image_to_process
        img_end=img_start+1
        args.num_test_images=1

    for i in range(img_start, img_end):
        print(f"analyzing image {i} out of {args.num_test_images}...")

        # run detection selection algorithm on image
        tmp, corrects = test_example(
            i,
            bg_test,
            test_gt,
            test_boxes_gt,
            fasterrcnn_model,
            vae_decoder,
            rank,
            args
        )

        correct_num_boxes, correct_labels, total = get_errors(output_file,test_gt, i,
                                                          tmp, correct_num_boxes, correct_labels, total)


    # log total runtime
    time_elapsed = time.time() - start_time
    output_file.write('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    output_file.write('\n')
    # write accuracy statistics to file
    output_file.write(
        f"correct_num_boxes {correct_num_boxes} {correct_num_boxes / total} correct_labels {correct_labels} {correct_labels / total}\n"
    )
    # write just the values for combining at the end
    output_file.write(
        f"{correct_num_boxes} {correct_labels} {time_elapsed}"
    )
    output_file.close()


if __name__ == "__main__":

    args=get_args()
    # get directory of script

    print(args.recheck)

    print("args.soft_nms",args.soft_nms,"args.lamb", args.lamb, "args.visible_thresh", args.visible_thresh, "args.occlusion_number_of_pixels", args.occlusion_number_of_pixels)

    # parse command line overrides to config file args

    if args.trans_type == 'None':
        args.trans_space_dimension=0


    os.system('rm output/output_process*')
    # run on CPU
    device=None
    world_size=1
    if args.run_parallel:
        n_gpus = torch.cuda.device_count()
        world_size=np.minimum(n_gpus,args.run_parallel)
        print(f"executing on {world_size} gpus...")
        # run the main routine, distributing it across GPUs
        run_torch_fn_mp(
            main, world_size, args
        )
    elif args.run_gpu>=0 and torch.cuda.device_count()>0:
        world_size = 1
        device = torch.device("cuda:"+str(args.run_gpu))
        print("executing on gpu",device)
        main(device, 1, args)
    else:
        world_size = 1
        device = torch.device("cpu")
        print("executing on cpu...")
        main(device, 1, args)


    process_results(world_size, args, device)


