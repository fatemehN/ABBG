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


def _convert_score(score):

        score = score.permute(2, 1, 0).contiguous().view(2, -1).permute(1, 0)
        score = F.log_softmax(score, dim=1).data[:, 0].cpu().numpy()
        return score

def _convert_bbox(delta):

    delta1 = delta.permute(2, 1, 0).contiguous().view(4, -1)
    delta = delta1.data.cpu().numpy()

    return delta, delta1

def _create_outputs(outputs, center_pos, s_x, image_sh, window, window_penalty):
    
    score = _convert_score(outputs['pred_logits'])
    pred_bbox, Tpred_bbox = _convert_bbox(outputs['pred_boxes'])

    pscore = score
    # window penalty
    pscore = pscore * (1 - window_penalty) + \
                window * window_penalty

    best_idx = np.argmax(pscore)
    bbox = Tpred_bbox[:, best_idx]
    bbox = bbox * s_x
    bbox[0] = bbox[0] + center_pos[0] - s_x / 2
    bbox[1] = bbox[1] + center_pos[1] - s_x / 2
    # width = bbox[2]
    # height = bbox[3]

    # # clip boundary
    # cx, cy, width, height = _bbox_clip(cx, cy, width,
    #                                         height, image_sh)

    # bbox = [cx - width / 2,
    #         cy - height / 2,
    #         width,
    #         height]

    return bbox #torch.from_numpy(np.array(bbox))


def _bbox_clip(cx, cy, width, height, boundary):
    cx = max(0, min(cx, boundary[1]))
    cy = max(0, min(cy, boundary[0]))
    width = max(10, min(width, boundary[1]))
    height = max(10, min(height, boundary[0]))
    return cx, cy, width, height


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

    # print('rect shapes', rect1.shape, rect2.shape)

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
    # target = torch.from_numpy(target).cuda().float()
    loss = F.smooth_l1_loss(input[pos_index], target[pos_index], reduction='sum')
    return loss

def rect_2_cxy_wh(rect):
    return np.array([rect[0]+rect[2]/2, rect[1]+rect[3]/2, rect[2], rect[3]])



def ABBG_attack(net, img, temp_list, mask_flag, center_pos, s_x, image_sh, window, window_penalty,
                 epsilon_ = 10, alpha_= 1, max_t=10):  
    """
    ABBG attack for MixFormer tracker
    Args:
        net: tracker network
        img: Current frame
        temp_list: template list available through tracker 
        mask_flag: whether you need the mask or not (True or False)
        center_pos: center position of previous bounding box
        s_x: scale parameter available through tracker 
        image_sh: original frame shape
        window: window variable of the tracker
        window_pentalty: window penalty variable of the tracker
        epsilon_: The pixel value of \epsilon parameter in TrackPGD
        alpha_: The step size of gradient update
        max_t: The iteration number
    Returns:
        Adversarial search region 
    """
    
    image_init = img
    #Normalize for TransT-M Tracker
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    inplace = False

    x_adv = Variable(img.data, requires_grad=True)
    for t in range(max_t):
        #Normalize for TransT-M Tracker
        x_adv.data = x_adv.data.float().mul(1.0 / 255.0).clamp(0.0, 1.0)
        x_adv.data[0] = tvisf.normalize(x_adv.data[0], mean, std, inplace)

        # mask_flag = True
        #TrabsTM track step
        outputs = net.track_seg(x_adv, temp_list, mask=mask_flag)
        pred_box = _create_outputs(outputs, center_pos, s_x, image_sh, window, window_penalty)

        true_bbox = torch.Tensor.repeat(pred_box, (1024, 1))
        target_pos = np.array(pred_box.tolist())

        ##Generate adversarial bounding box
        adv_bbox = np.tile(target_pos, (1024, 1))
        true_bbox_np = adv_bbox.copy()

        ##Generate random scale and shift parameters 
        rate_xy1 = np.random.uniform(0.1, 0.4, 1024)
        rate_xy2 = np.random.uniform(0.1, 0.4, 1024)
        rate_wd = np.random.uniform(0.7, 0.9, 1024)

        #ABBGs
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
        loss.backward(retain_graph=True)

        #Compute Perturbation
        adv_grad = x_adv.grad
        adv_grad = torch.sign(adv_grad) 
        pert = alpha_ * adv_grad 

        #Generate the pertubed search region 
        img = img - pert 
        img = where(img > image_init + epsilon_, image_init + epsilon_, img)
        img = where(img < image_init - epsilon_, image_init - epsilon_ , img)
        x_adv.data = torch.clamp(img, 0, 255)
        img = x_adv.data
    
    img_adv = x_adv.data
    return img_adv
