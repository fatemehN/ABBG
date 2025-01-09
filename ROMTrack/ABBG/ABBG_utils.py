from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os, cv2
import numpy as np
import torch, math

import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms.functional as tvisf


CUDA_LAUNCH_BLOCKING=1
torch.set_grad_enabled(True)


def where(cond, x, y):
    """
    code from :
        https://discuss.pytorch.org/t/how-can-i-do-the-operation-the-same-as-np-where/1329/8
    """
    cond = cond.float()
    return (cond*x) + ((1-cond)*y)


def sample_target(im, target_bb, search_area_factor, output_sz=256, mode=cv2.BORDER_REPLICATE):
    """ Extracts a crop centered at target_bb box, of size search_area_factor times target_bb(Both height and width)

    args:
        im - cv image
        target_bb - target box [x, y, w, h]
        search_area_factor - Ratio of crop size to target size
        output_sz - (float) Size to which the extracted crop is resized (always square). If None, no resizing is done.

    returns:
        cv image - extracted crop
        float - the factor by which the crop has been resized to make the crop size equal output_size
    """
    #im = im.clone().detach().cpu().numpy()
    x, y, w, h = target_bb.tolist()

    # Crop image
    ws = math.ceil(search_area_factor * w)
    hs = math.ceil(search_area_factor * h)

    if ws < 1 or hs < 1:
        return np.zeros((output_sz, output_sz))
        #raise Exception('Too small bounding box.')

    x1 = round(x + 0.5*w - ws*0.5)
    x2 = x1 + ws

    y1 = round(y + 0.5 * h - hs * 0.5)
    y2 = y1 + hs

    x1_pad = max(0, -x1)
    x2_pad = max(x2-im.shape[1]+1, 0)

    y1_pad = max(0, -y1)
    y2_pad = max(y2-im.shape[0]+1, 0)

    # Crop target
    im_crop = im[y1+y1_pad:y2-y2_pad, x1+x1_pad:x2-x2_pad]

    # Pad
    im_crop_padded = cv2.copyMakeBorder(im_crop, y1_pad, y2_pad, x1_pad, x2_pad, mode)


    im_crop_padded_rsz = cv2.resize(im_crop_padded, (output_sz, output_sz))
    
    return im_crop_padded_rsz #, h_rsz_f, w_rsz_f
    
 
def get_cls_loss(pred, label, select):
    if len(select.size()) == 0 or \
            select.size() == torch.Size([0]):
        return 0
    pred = torch.index_select(pred, 0, select)
    label = torch.index_select(label, 0, select)
    return -F.nll_loss(pred, label.long())

def select_cross_entropy_loss(msk, target):
    pred = msk.view(-1, 2)
    label = target.view(-1)
    obj = label.data.eq(1).nonzero().squeeze()
    bcg = label.data.eq(0).nonzero().squeeze()
    loss_pos = get_cls_loss(pred, label, obj)
    loss_neg = get_cls_loss(pred, label, bcg)
    return loss_pos, loss_neg



def delta2bbox(delta):
    bbox_cxcywh = delta.clone()
    '''based on (128,128) center region'''
    bbox_cxcywh[:, :2] = 128.0 + delta[:, :2] * 128.0  # center offset
    bbox_cxcywh[:, 2:] = 128.0 * torch.exp(delta[:, 2:])  # wh revise
    bbox_xywh = bbox_cxcywh.clone()
    bbox_xywh[:, :2] = bbox_cxcywh[:, :2] - 0.5 * bbox_cxcywh[:, 2:]
    return bbox_xywh


def pred2bbox(prediction, input_type=None):
        if input_type == 'bbox':
            Pbbox = prediction['bbox']
            Pbbox = delta2bbox(Pbbox)
            Pbbox_arr = np.array(Pbbox.squeeze().cpu())
            return Pbbox_arr

        elif input_type == 'corner':
            Pcorner = prediction['corner']  # (x1,y1,x2,y2)
            Pbbox_arr = np.array(Pcorner.squeeze().detach().cpu().numpy())
            Pbbox_arr[2:] = Pbbox_arr[2:] - Pbbox_arr[:2]  # (x1,y1,w,h)
            return Pbbox_arr

        elif input_type == 'mask':
            Pmask = prediction['mask']
            Pmask_arr = np.array(Pmask.squeeze().cpu())  # (H,W) (0,1)
            return Pmask_arr

        else:
            raise ValueError("input_type should be 'bbox' or 'mask' or 'corner' ")

def overlap_ratio(rect1, rect2):
    '''
    Compute overlap ratio between two rects
    - rect: 1d array of [x,y,w,h] or
            2d array of N x [x,y,w,h]
    '''
    # rect1 = np.transpose(rect1)

    if rect1.ndim==1:
        rect1 = rect1[None,:]
    if rect2.ndim==1:
        rect2 = rect2[None,:]

    left = np.maximum(rect1[:,0], rect2[:,0])
    right = np.minimum(rect1[:,0]+rect1[:,2], rect2[:,0]+rect2[:,2])
    top = np.maximum(rect1[:,1], rect2[:,1])
    bottom = np.minimum(rect1[:,1]+rect1[:,3], rect2[:,1]+rect2[:,3])

    intersect = np.maximum(0,right - left) * np.maximum(0,bottom - top)
    union = rect1[:,2]*rect1[:,3] + rect2[:,2]*rect2[:,3] - intersect
    iou = np.clip(intersect / union, 0, 1)
    return iou

def rpn_smoothL1(input, target, label):
    r"""
    :param input: torch.Size([1, 1024, 4])
    :param target: torch.Size([1, 1024, 4])
            label: (torch.Size([1, 1024]) pos neg or ignore
    :return:
    """
    # input = torch.transpose(input, 0, 1)
    pos_index = np.where(label.cpu() == 1)#changed
    # print(input.shape, target.shape)
    # target = torch.from_numpy(target).cuda().float()
    loss = F.smooth_l1_loss(input[pos_index], target[pos_index], reduction='sum')
    return loss

def rect_2_cxy_wh(rect):
    return np.array([rect[0]+rect[2]/2, rect[1]+rect[3]/2, rect[2], rect[3]])

class Preprocessor_wo_mask(object):
    def __init__(self):
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view((1, 3, 1, 1)).cuda()
        self.std = torch.tensor([0.229, 0.224, 0.225]).view((1, 3, 1, 1)).cuda()

    def process(self, img_tensor):
        # Deal with the image patch
        img_tensor_norm = ((img_tensor / 255.0) - self.mean) / self.std  # (1,3,H,W)
        return img_tensor_norm



def ABBG_attack(net, prev_box, img, resize_factor, search_size, temp, epsilon_ = 10, alpha_= 1, max_t=10):  
    """
    ABBG attack for ROMTrack tracker
    Args:
        net: tracker network
        prev_box: previous bounding box
        img: Current frame
        resize_factor: resize factor of the tracker
        search_size: search region size
        temp: template region
        epsilon_: The pixel value of \epsilon parameter in TrackPGD
        alpha_: The step size of gradient update
        max_t: The iteration number
    Returns:
        Adversarial search region 
    """
    pre = Preprocessor_wo_mask()
    img_tensor = torch.tensor(img).cuda().float().permute((2,0,1)).unsqueeze(dim=0)
    image_init = img_tensor
    for t in range(max_t):
        #Numpy to Tensor and Normalize for Tracker
        x_adv = pre.process(img_tensor).cuda().float()
        x_adv = Variable(x_adv, requires_grad=True)

        #ROMTrack Tracking Step
        net = net.cuda()
        out_dict, _ = net.forward_test(temp, x_adv)
        pred_box = out_dict['pred_boxes'].view(-1, 4)
        # # Baseline: Take the mean of all pred boxes as the final result
        pred_box = (pred_box.mean(dim=0) * search_size / resize_factor) #.tolist()  # (cx, cy, w, h) [0,1]
        cx_prev, cy_prev = prev_box[0] + 0.5 * prev_box[2], prev_box[1] + 0.5 * prev_box[3]
        half_side = 0.5 * search_size / resize_factor
        pred_box[0] = pred_box[0] + (cx_prev - half_side) - 0.5 * pred_box[2]
        pred_box[1] = pred_box[1] + (cy_prev - half_side) - 0.5 * pred_box[3]

        true_bbox = torch.Tensor.repeat(pred_box, (1024, 1))
        target_pos = np.array(pred_box.tolist())

        ##Generate adversarial bounding box
        adv_bbox = np.tile(target_pos, (1024, 1))
        true_bbox_np = adv_bbox.copy()

        ##Generate random scale and shift parameters 
        rate_xy1 = np.random.uniform(0.1, 0.4, 1024)
        rate_xy2 = np.random.uniform(0.1, 0.4, 1024)
        rate_wd = np.random.uniform(0.7, 0.9, 1024)

        ##ABBGs
        adv_bbox[..., 0] = target_pos[0] + rate_xy1*target_pos[2]
        adv_bbox[..., 1] = target_pos[1] + rate_xy2*target_pos[3]
        adv_bbox[..., 2] = target_pos[2] * rate_wd
        adv_bbox[..., 3] = target_pos[3] * rate_wd

        #Compute the IoU between ABBs and the true bounding box
        label = overlap_ratio(adv_bbox, true_bbox_np)
        # set thresold to define positive and negative samples, following the training step
        iou_hi = 0.8 * label.max()

        # make labels
        y_pos = np.where(label > iou_hi, 1, 0)
        y_pos = torch.from_numpy(y_pos).cuda().long()

        # calculate regression loss
        adv_bbox = torch.from_numpy(adv_bbox).unsqueeze(0).cuda().float()
        true_bbox = true_bbox.unsqueeze(0).cuda().float()
        #Compute ABBG loss
        loss_pseudo_reg = rpn_smoothL1(true_bbox, adv_bbox, y_pos.unsqueeze(0))
        loss = loss_pseudo_reg   

        # calculate the derivative
        net.zero_grad()
        if x_adv.grad is not None:
            x_adv.grad.data.fill_(0)
        #Backpropagate the loss gradients
        loss.backward(retain_graph=True)

        #Compute Perturbation
        adv_grad = x_adv.grad
        adv_grad = torch.sign(adv_grad) 
        pert = alpha_ * adv_grad 

        #Generate the pertubed search region 
        img_tensor = img_tensor - pert 
        img_tensor = where(img_tensor > image_init + epsilon_, image_init + epsilon_, img_tensor)
        img_tensor = where(img_tensor < image_init - epsilon_, image_init - epsilon_ , img_tensor)
        x_adv.data = torch.clamp(img_tensor, 0, 255)
        img_tensor = x_adv.data
    
    img_adv = x_adv.squeeze(dim=0).squeeze(dim=0).permute((1, 2, 0))
    return img_adv.detach().cpu().numpy()
