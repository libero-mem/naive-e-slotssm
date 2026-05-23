import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
from scipy.optimize import linear_sum_assignment
from torch import nn
import pdb

import torch
from torchvision.ops.boxes import box_area

def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)


def box_xyxy_to_cxcywh(x):
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2,
         (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)


def box_iou(boxes1, boxes2):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter
    eps = 1e-6
    iou = inter / (union + eps)  # avoid division by zero

    return iou, union


def generalized_box_iou(boxes1, boxes2):
    (boxes1[:, 2:] >= boxes1[:, :2]).all()
    (boxes2[:, 2:] >= boxes2[:, :2]).all()

    iou, union = box_iou(boxes1, boxes2)

    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])
    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    area = wh[:, :, 0] * wh[:, :, 1]  # enclosing area

    eps = 1e-6
    return iou - (area - union) / (area + eps)  # avoid division by zero


class HungarianMatcherbboxDSGG(nn.Module):
    def __init__(self, cost_bbox: float = 1, cost_giou: float = 1):
        super().__init__()
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        assert cost_bbox != 0 or cost_giou != 0, 'all costs cant be 0'

    @torch.no_grad()
    def forward(self, outputs, targets):
        bs, num_queries = outputs.shape[:2]
        indices = []

        for b in range(bs):
            out_bbox = outputs[b].float()
            if targets[b] is None:
                continue
            tgt_boxes = targets[b].float().to(out_bbox.device)

            cost_bbox = torch.cdist(out_bbox, tgt_boxes, p=1)
            cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_boxes))

            C = self.cost_bbox * cost_bbox + self.cost_giou * cost_giou

            C = C.view(num_queries, -1).cpu()
            indices.append(linear_sum_assignment(C))

        return [
            (
                torch.as_tensor(i, dtype=torch.int64),
                torch.as_tensor(j, dtype=torch.int64),
            )
            for i, j in indices
        ]
        

class HungarianMatcherbboxTRACK(nn.Module):
    def __init__(self, cost_bbox: float = 1, cost_giou: float = 1):
        super().__init__()
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        assert cost_bbox != 0 or cost_giou != 0, 'all costs cant be 0'

    @torch.no_grad()
    def forward(self, outputs, targets):
        bs, num_queries, horizon = outputs.shape[:3]
        indices = []

        for b in range(bs):
            out_bbox = outputs[b].float()
            if targets[b] is None:
                continue
            tgt_boxes = targets[b].float().to(out_bbox.device)

            out_bbox, tgt_boxes = out_bbox.permute(1, 0, 2), tgt_boxes.permute(1, 0, 2)
            cost_bbox = torch.cdist(out_bbox, tgt_boxes, p=1).sum(0)

            cost_giou = 0
            for h in range(horizon):
                # print(b, h, box_cxcywh_to_xyxy(out_bbox[h,:,:4]), box_cxcywh_to_xyxy(tgt_boxes[h,:,:4]))
                # print(-generalized_box_iou(box_cxcywh_to_xyxy(out_bbox[h,:,:4]), box_cxcywh_to_xyxy(tgt_boxes[h,:,:4])))
                cost_giou += -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox[h,:,:4]), box_cxcywh_to_xyxy(tgt_boxes[h,:,:4]))

            C = self.cost_bbox * cost_bbox + self.cost_giou * cost_giou

            C = C.view(num_queries, -1).cpu()
            try:
                indices.append(linear_sum_assignment(C))
            except:
                print("WHAT")
                print(out_bbox[:,:,:4])
                print(tgt_boxes[:,:,:4])
                print(C)
        return [
            (
                torch.as_tensor(i, dtype=torch.int64),
                torch.as_tensor(j, dtype=torch.int64),
            )
            for i, j in indices
        ]


def build_bbox_matcher(cost_bbox=2.5, cost_giou=2, with_tracking=False):
    if not with_tracking:
        return HungarianMatcherbboxDSGG(cost_bbox=cost_bbox, cost_giou=cost_giou)
    else:
        return HungarianMatcherbboxTRACK(cost_bbox=cost_bbox, cost_giou=cost_giou)


def _get_src_permutation_idx(indices):
    batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
    src_idx = torch.cat([src for (src, _) in indices])
    return batch_idx, src_idx


def _get_matched_pairs(src_data, target_data, indices):
    idx = _get_src_permutation_idx(indices)
    src_data = src_data[idx]
    target_data = torch.cat([t[i] for t, (_, i) in zip(target_data, indices)], dim=0)
    return src_data, target_data

def _get_unmatched_src_permutation_idx(src_data, indices):
    """
    Return (batch_idx, src_idx) for all UNMATCHED predictions.
    
    Args:
        indices: list of (src_indices, tgt_indices) for each image in the batch
        src_data: the model outputs, shape [batch_size, num_preds, ...]
    """
    batch_size, num_preds = src_data.shape[:2]
    device = src_data.device

    unmatched_batch_idx = []
    unmatched_src_idx = []

    # Loop over each element in the batch
    for i, (src, _) in enumerate(indices):
        # src contains the matched prediction indices for image i
        # We'll compute the "unmatched" indices by difference
        all_indices = torch.arange(num_preds, device=device)
        matched_mask = torch.zeros(num_preds, dtype=torch.bool, device=device)
        matched_mask[src] = True  # Mark matched predictions
        unmatched_mask = ~matched_mask
        unmatched = all_indices[unmatched_mask]  # Indices that are NOT matched

        # Keep track of the batch index for each unmatched prediction
        unmatched_batch_idx.append(torch.full_like(unmatched, i, dtype=torch.long))
        unmatched_src_idx.append(unmatched)

    # Concatenate everything into two 1D tensors
    unmatched_batch_idx = torch.cat(unmatched_batch_idx)
    unmatched_src_idx = torch.cat(unmatched_src_idx)

    return unmatched_batch_idx, unmatched_src_idx


def get_unmatched_src_data(src_data, indices):
    """
    Return just the UNMATCHED portion of src_data, flattened across the batch.
    
    Args:
        src_data: model outputs, shape [batch_size, num_preds, ...]
        indices: list of (src_indices, tgt_indices), matched indices per image
    """
    unmatched_idx = _get_unmatched_src_permutation_idx(src_data, indices)
    unmatched_idx = (unmatched_idx[0].cpu(), unmatched_idx[1].cpu())
    # Advanced indexing using (batch_idx, src_idx)
    unmatched_src_data = src_data[unmatched_idx]
    return unmatched_src_data


def loss_boxes(outputs, targets, indices, num_interactions):
    # idx = _get_src_permutation_idx(indices)
    # src_boxes = outputs[idx]
    # target_boxes = torch.cat([t[i] for t, (_, i) in zip(targets, indices)], dim=0)
    src_boxes, target_boxes = _get_matched_pairs(outputs, targets, indices)
    losses = {}
    if src_boxes.shape[0] == 0:
        losses['loss_bbox'] = 0
        losses['loss_giou'] = 0
        losses['loss_objectness'] = 0
    else:
        loss_bbox = F.l1_loss(src_boxes[:,:4], target_boxes, reduction='none')
        losses['loss_bbox'] = loss_bbox.sum() / num_interactions
        loss_giou = 1 - torch.diag(generalized_box_iou(box_cxcywh_to_xyxy(src_boxes[:,:4]),
                                                            box_cxcywh_to_xyxy(target_boxes)))
    
        losses['loss_giou'] = loss_giou.sum() / num_interactions

        pred_ones = src_boxes[:,4] # needs to be aligned with 1
        pred_zeros = get_unmatched_src_data(outputs, indices)[:,4] # needs to be aligned with 0

        assert(pred_ones.shape[0]+pred_zeros.shape[0] == outputs.shape[0]*outputs.shape[1])
        
        preds = torch.cat([pred_ones, pred_zeros], dim=0)
        labels = torch.cat([torch.ones_like(pred_ones), 
                            torch.zeros_like(pred_zeros)], dim=0).to(outputs.device)
        
        losses['loss_objectness'] = torch.nn.functional.binary_cross_entropy(preds, labels)
                                                 
    return losses



class BoxLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bbox_matcher = build_bbox_matcher(with_tracking=True)

    def forward(self, bboxes, gt_bboxes):
        num_interactions = sum(t.shape[0] for t in gt_bboxes)
        num_interactions = torch.as_tensor([num_interactions], dtype=torch.float, device=bboxes.device)
        num_interactions = torch.clamp(num_interactions, min=1).item()
        indices = self.bbox_matcher(bboxes[:,:,:4], gt_bboxes)
        return loss_boxes(bboxes, gt_bboxes, indices, num_interactions), indices


class CaptionLoss(nn.Module):
    def __init__(self, ignore_index=32000):
        super().__init__()
        self.ignore_index = ignore_index

    def forward(self, captions, gt_captions, indices):
        # idx = _get_src_permutation_idx(indices)
        # src_captions = captions[idx]
        # target_captions = torch.cat([t[i] for t, (_, i) in zip(gt_captions, indices)], dim=0)
        src_captions, target_captions = _get_matched_pairs(captions, gt_captions, indices)
        B, S, dim = src_captions.shape
        src_captions = src_captions.reshape(B*S, -1)
        target_captions = target_captions.reshape(B*S)
        caption_loss = F.cross_entropy(src_captions, target_captions, ignore_index=self.ignore_index)
        return caption_loss

class MSEFeatureLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.criterion = torch.nn.MSELoss()  # By default, reduction='mean'

    def forward(self, features, gt_features, indices):
        src_features, target_features = _get_matched_pairs(features, gt_features, indices)
        return self.criterion(src_features, target_features)
    

class L1ImputeLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.criterion = torch.nn.L1Loss()  # By default, reduction='mean'

    def forward(self, impute_learning_data):
        text_pred, text_gt = impute_learning_data['text_pred'], impute_learning_data['text_gt']
        visual_pred, visual_gt = impute_learning_data['visual_pred'], impute_learning_data['visual_gt']

        text_impute_losses = []
        visual_impute_losses = []
        for ind in range(3):
            text_impute_losses.append(self.criterion(text_pred[:,ind], 
                                                     text_gt[:,ind].detach()))
            visual_impute_losses.append(self.criterion(visual_pred[:,ind], 
                                                       visual_gt[:,ind].detach()))

        return text_impute_losses, visual_impute_losses



class BoxLossWithTrack(nn.Module):
    def __init__(self, objectness_required=False, eval_mode=False):
        super().__init__()
        self.bbox_coef = 1 # 2.5
        self.giou_coef = 1 # 2
        self.obj_coef = 1
        self.objectness_required = objectness_required
        self.eval_mode = eval_mode

    def loss_boxes(self, outputs, targets, indices, num_interactions):
        # idx = _get_src_permutation_idx(indices)
        # src_boxes = outputs[idx]
        # target_boxes = torch.cat([t[i] for t, (_, i) in zip(targets, indices)], dim=0)
        src_boxes, target_boxes = _get_matched_pairs(outputs, targets, indices) # works well
        # print(src_boxes.shape)
        # print(target_boxes.shape); 
        # print(target_boxes); 1/0
        losses = {}
        if src_boxes.shape[0] == 0:
            losses['loss_bbox'] = 0
            losses['loss_giou'] = 0
            losses['loss_objectness'] = 0
        else:
            bn, horizon, dim = src_boxes.shape
            loss_bbox = F.l1_loss(src_boxes[:,:,:4], target_boxes[:,:,:4], reduction='none')
            losses['loss_bbox'] = loss_bbox.sum() / (num_interactions) * self.bbox_coef

            loss_giou = 0
            for h in range(horizon):
                loss_giou += 1 - torch.diag(generalized_box_iou(box_cxcywh_to_xyxy(src_boxes[:,h,:4]), 
                                                            box_cxcywh_to_xyxy(target_boxes[:,h,:4])))
            losses['loss_giou'] = loss_giou.sum() / (num_interactions) * self.giou_coef

            pred_ones = src_boxes[:,:,4] # needs to be aligned with 1
            pred_zeros = get_unmatched_src_data(outputs, indices)[:,:,4] # needs to be aligned with 0

            assert(pred_ones.shape[0]+pred_zeros.shape[0] == outputs.shape[0]*outputs.shape[1])
            
            preds = torch.cat([pred_ones, pred_zeros], dim=0).flatten()
            labels = torch.cat([torch.ones_like(pred_ones), 
                                torch.zeros_like(pred_zeros)], dim=0).flatten()

            if self.objectness_required:
                with torch.cuda.amp.autocast(enabled=False):
                    losses['loss_objectness'] = torch.nn.functional.binary_cross_entropy(preds, labels, reduction='sum') / num_interactions * self.obj_coef

            if self.eval_mode:
                # Threshold the predictions at 0.5
                binary_preds = (preds >= 0.5).int()

                # Compare with ground truth
                correct = (binary_preds == labels).sum().item()
                total = labels.numel()

                # Compute accuracy
                accuracy = correct / total
                losses['accuracy_objectness'] = accuracy

        return losses


    def forward(self, bboxes, gt_bboxes, indices):
        num_interactions = sum(t.shape[0] for t in gt_bboxes)
        num_interactions = torch.as_tensor([num_interactions], dtype=torch.float, device=bboxes.device)
        num_interactions = torch.clamp(num_interactions, min=1).item()
        return self.loss_boxes(bboxes, gt_bboxes, indices, num_interactions)


class MSEFeatureLossWithTrack(nn.Module):
    def __init__(self, eval_mode=False):
        super().__init__()
        self.depth_coef = 0.02
        self.eval_mode = eval_mode
        if self.eval_mode:
            self.criterion = torch.nn.MSELoss(reduction='mean')  # By default, reduction='mean'
        else:
            self.criterion = torch.nn.MSELoss(reduction='sum')  # By default, reduction='mean'

    def forward(self, features, gt_features, indices):
        num_interactions = sum(t.shape[0] for t in gt_features)
        num_interactions = torch.as_tensor([num_interactions], dtype=torch.float, device=features.device)
        num_interactions = torch.clamp(num_interactions, min=1).item()

        src_features, target_features = _get_matched_pairs(features, gt_features, indices)
        src_features = src_features.flatten(0,1)
        target_features = target_features.flatten(0,1)
        
        
        # Create a mask: True if the batch contains at least one value not equal to -1
        mask = (target_features != -1).any(dim=(1, 2, 3))

        # Use the mask to index the valid batch indices
        a_filtered = src_features[mask]
        b_filtered = target_features[mask]

        # # Print the shape to verify
        # print(a_filtered.shape, b_filtered.shape)
        # 1/0
        if self.eval_mode:
            return {
                'loss_obj_depth': self.criterion(a_filtered, b_filtered)
            }
        
        return {
            'loss_obj_depth': self.criterion(a_filtered, b_filtered) / num_interactions * self.depth_coef
        }

import numpy as np
def all_to_one(outputs, targets):
    bs, num_queries, horizon = outputs.shape[:3]
    indices = []
    for b in range(bs):
        indices.append((np.arange(num_queries), np.zeros(num_queries)))
    # print(indices); 1/0
    return [
        (
            torch.as_tensor(i, dtype=torch.int64),
            torch.as_tensor(j, dtype=torch.int64),
        )
        for i, j in indices
    ]

class ObjectCentricLoss(nn.Module):
    # object centric losses is designed for slots to learn
    # # # 2D bounding box and objectness prediction
    # # # depth estimation
    # # # consistently through time
    #
    # It uses bipartite matching between predictions and gts
    # in order to select the best pairs of correspondence.
    #
    def __init__(self, bbox_matching='standard', objectness_required=True, eval_mode=False):
        super().__init__()
        if bbox_matching == 'standard':
            self.bbox_matcher = build_bbox_matcher(with_tracking=True)
        else:
            self.bbox_matcher = all_to_one
        self.bbox_loss = BoxLossWithTrack(objectness_required=objectness_required, eval_mode=eval_mode)
        self.recon_loss = MSEFeatureLossWithTrack(eval_mode=eval_mode)

    def forward(self, object_preds, object_gts, view='main'):
        assert(view in ['main', 'wrist'])
        if view == 'main':
            bz, horizon, num_view, num_boxes, box_dim = object_preds["bboxes"].shape
            batched_gt_bboxes = []; batched_pd_bboxes = object_preds["bboxes"][:,:,0].permute(0, 2, 1, 3)
            batched_gt_depths = []; batched_pd_depths = object_preds["depths"][:,:,0].permute(0, 2, 1, 3, 4, 5)
            for i in range(bz):
                batched_gt_bboxes.append(torch.stack(list(object_gts["bboxes"]["main"][i].values())))
                batched_gt_depths.append(torch.stack(list(object_gts["depths"]["main"][i].values())))
        
        elif view == 'wrist':
            bz, horizon, num_view, num_boxes, box_dim = object_preds["bboxes"].shape
            batched_gt_bboxes = []; batched_pd_bboxes = object_preds["bboxes"][:,:,1].permute(0, 2, 1, 3)
            batched_gt_depths = []; batched_pd_depths = object_preds["depths"][:,:,1].permute(0, 2, 1, 3, 4, 5)
            for i in range(bz):
                batched_gt_bboxes.append(torch.stack(list(object_gts["bboxes"]["wrist"][i].values())))
                batched_gt_depths.append(torch.stack(list(object_gts["depths"]["wrist"][i].values())))

        # find optimal assignment based on bboxes
        indices = self.bbox_matcher(batched_pd_bboxes, batched_gt_bboxes)
        # print(self.bbox_matcher)
        # print(indices); 1/0
        losses = {}
        losses.update(self.bbox_loss(batched_pd_bboxes, batched_gt_bboxes, indices))
        losses.update(self.recon_loss(batched_pd_depths, batched_gt_depths, indices))
        return losses


def jaccard_loss(pred, true, smooth=100):
    """Computes the Jaccard loss, a.k.a the IoU loss.
    Jaccard = (|X & Y|)/ (|X|+ |Y| - |X & Y|)
            = sum(|A*B|)/(sum(|A|)+sum(|B|)-sum(|A*B|))
    Note that PyTorch optimizers minimize a loss. In this
    case, we would like to maximize the jaccard loss so we
    return the negated jaccard loss.
    Args:
        true (tensor): 1D ground truth tensor.
        preds (tensor): 1D prediction truth tensor.
        eps (int): Smoothing factor
    Returns:
        jacc_loss: the Jaccard loss.
    """
    intersection = torch.sum(true*pred)
    jac = (intersection + smooth) / (torch.sum(true) + torch.sum(pred) - intersection + smooth)
    return (1 - jac) * smooth

def recall(true, pred):
    """
    Computes Recall (Sensitivity) between ground truth and predictions.
    
    Args:
        true (tensor): 1D ground truth tensor (binary labels: 0 or 1).
        pred (tensor): 1D prediction tensor (probabilities or logits).
    
    Returns:
        recall (float): The recall score.
    """
    pred = pred.round()  # Convert probabilities/logits to binary (0 or 1)
    
    tp = torch.sum((true == 1) & (pred == 1)).float()  # True Positives
    fn = torch.sum((true == 1) & (pred == 0)).float()  # False Negatives

    recall = tp / (tp + fn + 1e-6)  # Avoid division by zero

    return recall

class BoxLossWithTrackV2(nn.Module):
    def __init__(self, objectness_required=False, interaction_required=False, eval_mode=False):
        super().__init__()
        self.bbox_coef = 1 # 2.5
        self.giou_coef = 1 # 2
        self.obj_coef = 1
        self.objectness_required = objectness_required
        self.interaction_required = interaction_required
        self.eval_mode = eval_mode

    def loss_boxes(self, outputs, targets, indices, num_interactions):
        # idx = _get_src_permutation_idx(indices)
        # src_boxes = outputs[idx]
        # target_boxes = torch.cat([t[i] for t, (_, i) in zip(targets, indices)], dim=0)
        src_boxes, target_boxes = _get_matched_pairs(outputs, targets, indices) # works well
        # print(src_boxes.shape)
        # print(target_boxes.shape); 
        # print(target_boxes); 1/0
        losses = {}
        if src_boxes.shape[0] == 0:
            losses['loss_bbox'] = 0
            losses['loss_giou'] = 0
            losses['loss_objectness'] = 0
        else:
            bn, horizon, dim = src_boxes.shape
            loss_bbox = F.l1_loss(src_boxes[:,:,:4], target_boxes[:,:,:4], reduction='none')
            losses['loss_bbox'] = loss_bbox.sum() / (num_interactions) * self.bbox_coef

            loss_giou = 0
            for h in range(horizon):
                loss_giou += 1 - torch.diag(generalized_box_iou(box_cxcywh_to_xyxy(src_boxes[:,h,:4]), 
                                                            box_cxcywh_to_xyxy(target_boxes[:,h,:4])))
            losses['loss_giou'] = loss_giou.sum() / (num_interactions) * self.giou_coef

            pred_ones = src_boxes[:,:,4] # needs to be aligned with 1
            unmatched_preds = get_unmatched_src_data(outputs, indices)
            pred_zeros = unmatched_preds[:,:,4] # needs to be aligned with 0

            assert(pred_ones.shape[0]+pred_zeros.shape[0] == outputs.shape[0]*outputs.shape[1])
            
            preds = torch.cat([pred_ones, pred_zeros], dim=0).flatten()
            labels = torch.cat([torch.ones_like(pred_ones), 
                                torch.zeros_like(pred_zeros)], dim=0).flatten()

            if self.objectness_required:
                with torch.cuda.amp.autocast(enabled=False):
                    losses['loss_objectness'] = torch.nn.functional.binary_cross_entropy(preds, labels, reduction='sum') / num_interactions * self.obj_coef

            # print(src_boxes[:,:,5].shape)
            # print(src_boxes[:,:,5])

            # print(unmatched_preds[:,:,5].shape)
            # print(unmatched_preds[:,:,5])

            # print(target_boxes[:,:,5].shape)
            # print(target_boxes[:,:,5])

            # print(torch.zeros_like(unmatched_preds[:,:,5]).shape)
            # print(torch.zeros_like(unmatched_preds[:,:,5]))

            if self.eval_mode:
                # Threshold the predictions at 0.5
                binary_preds = (preds >= 0.5).int()

                # Compare with ground truth
                correct = (binary_preds == labels).sum().item()
                total = labels.numel()

                # Compute accuracy
                accuracy = correct / total
                losses['accuracy_objectness'] = accuracy

            if self.interaction_required:
                preds = torch.cat([src_boxes[:,:,5], unmatched_preds[:,:,5]], dim=0).float()
                labels = torch.cat([target_boxes[:,:,5], torch.zeros_like(unmatched_preds[:,:,5])], dim=0).float()
                # 1/0
                with torch.cuda.amp.autocast(enabled=False):
                    weights = torch.where(labels == 1, 2.0, 1.0)  # Higher weight for positives
                    losses['loss_interactable'] = torch.nn.functional.binary_cross_entropy(preds, labels, reduction='sum', weight=weights) / num_interactions * self.obj_coef
                losses['recall_interactable'] = recall(labels, preds.detach())

        return losses

    def forward(self, bboxes, gt_bboxes, indices):
        num_interactions = sum(t.shape[0] for t in gt_bboxes)
        num_interactions = torch.as_tensor([num_interactions], dtype=torch.float, device=bboxes.device)
        num_interactions = torch.clamp(num_interactions, min=1).item()
        return self.loss_boxes(bboxes, gt_bboxes, indices, num_interactions)

class SegmentationLossWithTrack(nn.Module):
    def __init__(self, foreground_weight=100.0, eval_mode=False):
        super().__init__()
        self.eval_mode = eval_mode
        # self.criterion = self.jaccard_loss

        self.pos_weight = torch.tensor([foreground_weight])
        self.criterion = F.binary_cross_entropy_with_logits

    def jaccard_loss(self, preds, targets, eps=1e-7):
        # preds: shape [B, D], values in [0,1]
        # targets: shape [B, D], binary (0 or 1)
        intersection = (preds * targets).sum(dim=1)
        union = (preds + targets - preds * targets).sum(dim=1)
        jaccard = (intersection + eps) / (union + eps)
        loss = 1.0 - jaccard
        return loss

    def get_jaccard_loss(self, features, gt_features, indices):
        device = features.device
        src_features, target_features = _get_matched_pairs(features, gt_features, indices)
        # print(torch.min(src_features), torch.max(src_features), torch.mean(src_features))
        src_features = src_features.detach().flatten(2,3).flatten(0,1).sigmoid()
        target_features = target_features.detach().flatten(2,3).flatten(0,1).float()
        return self.jaccard_loss(src_features, target_features).mean()

    def forward(self, features, gt_features, indices):
        # num_interactions = sum(t.shape[0] for t in gt_features)
        # num_interactions = torch.as_tensor([num_interactions], dtype=torch.float, device=features.device)
        # num_interactions = torch.clamp(num_interactions, min=1).item()
        device = features.device
        src_features, target_features = _get_matched_pairs(features, gt_features, indices)
        # print(torch.min(src_features), torch.max(src_features), torch.mean(src_features))
        src_features = src_features.flatten()
        target_features = target_features.flatten().float()
        if self.eval_mode:
            return {
                'loss_obj_seg': self.criterion(src_features, target_features, pos_weight=self.pos_weight.to(device), reduction='sum')
            }
        
        return {
            'loss_obj_seg': self.criterion(src_features, target_features, pos_weight=self.pos_weight.to(device))
        }

class ObjectCentricLossV2(nn.Module):
    # object centric losses is designed for slots to learn
    # # # 2D bounding box and objectness prediction
    # # # depth estimation
    # # # consistently through time
    # # # with active object filtering
    #
    # It uses bipartite matching between predictions and gts
    # in order to select the best pairs of correspondence.
    #
    def __init__(self, bbox_matching='standard', objectness_required=True, eval_mode=False):
        super().__init__()
        if bbox_matching == 'standard':
            self.bbox_matcher = build_bbox_matcher(with_tracking=True)
        else:
            self.bbox_matcher = all_to_one
        self.bbox_loss = BoxLossWithTrackV2(objectness_required=objectness_required, eval_mode=eval_mode)
        self.recon_loss = MSEFeatureLossWithTrack(eval_mode=eval_mode)
        self.interactable_loss = torch.nn.functional.binary_cross_entropy

    def forward(self, object_preds, object_gts, view='main'):
        assert(view in ['main', 'wrist'])
        if view == 'main':
            bz, horizon, num_view, num_boxes, box_dim = object_preds["bboxes"].shape
            batched_gt_bboxes = []; batched_pd_bboxes = object_preds["bboxes"][:,:,0].permute(0, 2, 1, 3)
            batched_gt_depths = []; batched_pd_depths = object_preds["depths"][:,:,0].permute(0, 2, 1, 3, 4, 5)
            for i in range(bz):
                batched_gt_bboxes.append(torch.stack(list(object_gts["bboxes"]["main"][i].values())))
                batched_gt_depths.append(torch.stack(list(object_gts["depths"]["main"][i].values())))
        
        elif view == 'wrist':
            bz, horizon, num_view, num_boxes, box_dim = object_preds["bboxes"].shape
            batched_gt_bboxes = []; batched_pd_bboxes = object_preds["bboxes"][:,:,1].permute(0, 2, 1, 3)
            batched_gt_depths = []; batched_pd_depths = object_preds["depths"][:,:,1].permute(0, 2, 1, 3, 4, 5)
            for i in range(bz):
                batched_gt_bboxes.append(torch.stack(list(object_gts["bboxes"]["wrist"][i].values())))
                batched_gt_depths.append(torch.stack(list(object_gts["depths"]["wrist"][i].values())))

        # find optimal assignment based on bboxes
        indices = self.bbox_matcher(batched_pd_bboxes, batched_gt_bboxes)
        # print(self.bbox_matcher)
        # print(indices); 1/0
        losses = {}
        losses.update(self.bbox_loss(batched_pd_bboxes, batched_gt_bboxes, indices))
        losses.update(self.recon_loss(batched_pd_depths, batched_gt_depths, indices))
        return losses

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        bce_loss = nn.functional.binary_cross_entropy_with_logits(inputs, targets.float(), reduction='none')
        probs = torch.sigmoid(inputs)
        pt = torch.where(targets == 1, probs, 1 - probs)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()

def jaccard_loss(logits, targets, smooth=1e-6):
    """
    logits: raw output from the model (before sigmoid)
    targets: binary labels (0 or 1)
    """
    probs = torch.sigmoid(logits)
    intersection = (probs * targets).sum()
    union = (probs + targets - probs * targets).sum()
    loss = 1 - (intersection + smooth) / (union + smooth)
    return loss

class RobotSSMObjectLossWithTrack(nn.Module):
    # object centric losses is designed for slots to learn
    # # # 2D bounding box and objectness prediction
    # # # depth estimation
    # # # consistently through time
    # # # with active object filtering
    #
    # It uses bipartite matching between predictions and gts
    # in order to select the best pairs of correspondence.
    #
    def __init__(self, bbox_matching='standard', objectness_required=True, seg_required=True, event_required=True, 
                       interact_required=False, eval_mode=False):
        super().__init__()
        if bbox_matching == 'standard':
            self.bbox_matcher = build_bbox_matcher(with_tracking=True)
        else:
            self.bbox_matcher = all_to_one
        self.seg_required = seg_required

        self.bbox_loss = BoxLossWithTrackV2(objectness_required=objectness_required, eval_mode=eval_mode)
        if seg_required:
            self.seg_loss = SegmentationLossWithTrack(eval_mode=eval_mode)
        else:
            self.seg_loss  = None
        if interact_required:
            self.interactable_loss = torch.nn.functional.binary_cross_entropy
        else:
            self.interactable_loss = None
        if event_required:
            self.event_pos_weight = torch.tensor([5.0])
            self.event_loss = F.binary_cross_entropy_with_logits
            self.event_loss_focal = FocalLoss()
            # self.event_loss = jaccard_loss

    def preprocess(self, all_obj_bboxes, all_obj_segids, all_pixel_seg_values, all_interaction_cnts):
        object_gts = {}

        # Get obj boxes
        object_gts["bboxes"] = [] # [b x o x h x 5]
        for b, obj_bboxes in enumerate(all_obj_bboxes):
            temp_obj_bboxes = []
            for obj_data in obj_bboxes:
                temp_obj_bboxes.append(np.stack(obj_data, axis=0))
            temp_obj_bboxes = torch.tensor(
                np.stack(temp_obj_bboxes, axis=0)
            )
            object_gts["bboxes"].append(temp_obj_bboxes)

        # Get obj seg maps
        object_gts["segs"] = []  # [b x o x h x img_dim]
        for b, obj_segids in enumerate(all_obj_segids):
            obj_masks = []
            for s, segid in enumerate(obj_segids):
                obj_masks.append(all_pixel_seg_values[b] == segid)
            obj_masks = torch.stack(obj_masks, dim=0)
            object_gts["segs"].append(obj_masks)

        # Get interaction events
        object_gts["itrn_cnt"] = [] # [b x o x (h-1)]
        # for b, obj_itrn_cnts in enumerate(all_interaction_cnts):
        #     temp_obj_cnts = []
        #     for obj_data in obj_itrn_cnts:
        #         temp_obj_cnts.append(torch.tensor(obj_data)[1:] != obj_data[0])
        #     temp_obj_cnts = torch.stack(temp_obj_cnts, dim=0).float()
        #     object_gts["itrn_cnt"].append(temp_obj_cnts)
        return object_gts

    def update_weights(self, losses, weights):
        updated_losses = {}
        for key, value in losses.items():
            updated_losses[key] = value * weights[key]
        return updated_losses

    def get_match_indices(self, object_preds, object_gts):
        device = object_preds["bboxes"].device
        bz, num_boxes, horizon, box_dim = object_preds["bboxes"].shape
        batched_gt_bboxes = []; batched_pd_bboxes = object_preds["bboxes"]
        for i in range(bz):
            batched_gt_bboxes.append(object_gts["bboxes"][i].to(device))

        # find optimal assignment based on bboxes
        indices = self.bbox_matcher(batched_pd_bboxes, batched_gt_bboxes)
        return indices

    def get_interactable_loss(self, interact_predictions, interactables, indices, device):
        src_interactables, target_interactables = _get_matched_pairs(interact_predictions, interactables, indices) # works well
        target_interactables = target_interactables.to(device)
        pred_ones = src_interactables[:,:] # needs to be aligned with 1
        unmatched_preds = get_unmatched_src_data(interact_predictions, indices)
        pred_zeros = unmatched_preds[:,:] # needs to be aligned with 0
        
        loss = {}
        preds = torch.cat([pred_ones, pred_zeros], dim=0).flatten()
        labels = torch.cat([torch.ones_like(pred_ones), 
                            torch.zeros_like(pred_zeros)], dim=0).flatten().to(device)
        with torch.cuda.amp.autocast(enabled=False):
            weights = torch.where(labels == 1, 2.0, 1.0)  # Higher weight for positives
            loss = torch.nn.functional.binary_cross_entropy(preds, labels, reduction='mean', weight=weights)

        return loss

    def get_event_prediction_loss(self, event_predictions, events, indices, device, include_event_metrics=True, event_pos_weight=None):
        src_events, target_events = _get_matched_pairs(event_predictions, events, indices) # works well
        target_events = target_events.to(device)
        if event_pos_weight is None:
            event_pos_weight = self.event_pos_weight.to(device)
        else:
            event_pos_weight = torch.tensor(event_pos_weight).to(device)
        loss = self.event_loss(src_events, target_events, pos_weight=event_pos_weight, reduction='mean')
        loss += self.event_loss_focal(src_events, target_events)

        if include_event_metrics:
            probs = torch.sigmoid(src_events.detach())
            preds = (probs > 0.8).long()  # you can tune the threshold later

            preds = preds.long()
            targets = target_events.long()
            
            true_positives = ((preds == 1) & (targets == 1)).sum().item()
            false_positives = ((preds == 1) & (targets == 0)).sum().item()
            false_negatives = ((preds == 0) & (targets == 1)).sum().item()
            
            precision = true_positives / (true_positives + false_positives + 1e-8)
            recall = true_positives / (true_positives + false_negatives + 1e-8)
            f1 = 2 * precision * recall / (precision + recall + 1e-8)
            
            metrics = {
                'precision': precision,
                'recall': recall,
                'f1': f1
            }
            return loss, metrics

        return loss, None

    def forward(self, object_preds, object_gts, indices=None, implicit_jaccard=True):
        device = object_preds["bboxes"].device
        bz, num_boxes, horizon, box_dim = object_preds["bboxes"].shape
        batched_gt_bboxes = []; batched_pd_bboxes = object_preds["bboxes"]
        batched_gt_segs = []; batched_pd_segs = object_preds["segs"]
        for i in range(bz):
            batched_gt_bboxes.append(object_gts["bboxes"][i].to(device))
            batched_gt_segs.append(object_gts["segs"][i].to(device))
        # find optimal assignment based on bboxes
        if indices is None:
            indices = self.bbox_matcher(batched_pd_bboxes, batched_gt_bboxes)
        # print(self.bbox_matcher)
        # print(indices); 1/0
        losses = {}
        losses.update(self.bbox_loss(batched_pd_bboxes, batched_gt_bboxes, indices))
        losses.update(self.seg_loss(batched_pd_segs, batched_gt_segs, indices))
        if implicit_jaccard:
            self.jaccard_loss = self.seg_loss.get_jaccard_loss(batched_pd_segs, batched_gt_segs, indices)
        return losses, indices

    def get_jaccard_evaluations(self):
        return self.jaccard_loss

def info_nce_for_one_list_pair(list1, list2, temperature=0.07):
    """
    list1, list2: lists of (key, feature) pairs (already aligned or not).
    temperature: float

    Returns: scalar InfoNCE loss
    """
    # 1) Convert to dict and align
    dict1 = dict(list1)
    dict2 = dict(list2)
    common_keys = list(set(dict1.keys()) & set(dict2.keys()))

    if len(common_keys) == 0:
        return 0
    # 2) Build feature tensors
    x = torch.stack([dict1[k] for k in common_keys])  # shape: (N, D)
    y = torch.stack([dict2[k] for k in common_keys])  # shape: (N, D)
    # 3) Optionally normalize
    x = torch.nn.functional.normalize(x, dim=1)
    y = torch.nn.functional.normalize(y, dim=1)

    # 4) Compute similarity (logits)
    logits = (x @ y.T) / temperature  # shape: (N, N)

    # 5) Create labels (i -> i) and compute cross-entropy
    labels = torch.arange(x.size(0)).to(logits.device)
    loss = torch.nn.functional.cross_entropy(logits, labels)

    return loss


def batched_info_nce_loss(batched_list1, batched_list2, temperature=0.07):
    """
    Computes the average InfoNCE loss over a batch of list-pairs.
    """
    total_loss = 0.0
    n_samples = len(batched_list1)
    
    for i in range(n_samples):
        loss_i = info_nce_for_one_list_pair(batched_list1[i], batched_list2[i], temperature)
        total_loss += loss_i
    
    return total_loss / n_samples
