import torch
import torch.nn.functional as F
import numpy as np

import os
import random
import numpy as np 
import pandas as pd 
import torch

from .submod_selection.selection_utils import *
from .submod_selection.factory import get_selection_function

# def filter_similarity(labels_img, boxes_img, feats_img, cls_img, obj_img, num_proposals):
#     """
#     Filters proposals for one image.
    
#     Args:
#         labels_img: (num_boxes,) Tensor of ROI labels.
#         boxes_img:  (num_boxes, 4) Tensor of bounding boxes.
#         feats_img:  (num_boxes, d_model) Tensor of proposal features.
#         cls_img:    (num_boxes, num_cls) Tensor of class logits.
#         obj_img:    (num_boxes, 1) Tensor of objectness scores.
#         num_proposals: desired final number of proposals.
#         unknown_sim_thresh: cosine similarity threshold for unknown proposals.
    
#     Returns:
#         filt_boxes: (num_proposals, 4)
#         filt_feats: (num_proposals, d_model)
#         filt_cls:   (num_proposals, num_cls)
#         filt_obj:   (num_proposals, 1)
#         filt_labels:(num_proposals,)
#     """
#     # Get indices for known and unknown proposals.
#     known_mask = (labels_img != -1)
#     unknown_mask = (labels_img == -1)
#     known_idx = torch.nonzero(known_mask, as_tuple=False).squeeze(1)
#     unknown_idx = torch.nonzero(unknown_mask, as_tuple=False).squeeze(1)
    
#     # Optionally filter unknown proposals using cosine similarity.
#     if known_idx.numel() > 0 and unknown_idx.numel() > 0:
#         known_feats = feats_img[known_idx]  # (n_known, d_model)
#         unknown_feats = feats_img[unknown_idx]  # (n_unknown, d_model)
#         known_proto = known_feats.mean(dim=0)  # (d_model,)
#         known_proto_norm = known_proto / (known_proto.norm(p=2) + 1e-6)
#         unknown_feats_norm = unknown_feats / (unknown_feats.norm(p=2, dim=-1, keepdim=True) + 1e-6)
#         cos_sim = (unknown_feats_norm * known_proto_norm.unsqueeze(0)).sum(dim=-1)
#         sim_mask = cos_sim >= unknown_sim_thresh
#         unknown_idx = unknown_idx[sim_mask]
    
#     # Combine indices.
#     combined_idx = torch.cat([known_idx, unknown_idx], dim=0)
#     num_known = known_idx.numel()
#     allowed_unknown = max(num_proposals - num_known, 0)
    
#     # If too many unknown proposals, randomly sample.
#     if unknown_idx.numel() > allowed_unknown:
#         perm = torch.randperm(unknown_idx.numel(), device=unknown_idx.device)
#         unknown_idx = unknown_idx[perm[:allowed_unknown]]
#         combined_idx = torch.cat([known_idx, unknown_idx], dim=0)
    
#     # If still too few proposals, pad by repeating a candidate valid proposal.
#     if combined_idx.numel() < num_proposals:
#         pad_count = num_proposals - combined_idx.numel()
#         # Use the first valid index if exists; otherwise, use index 0.
#         candidate = combined_idx[0].unsqueeze(0) if combined_idx.numel() > 0 else torch.tensor([0], device=boxes_img.device)
#         pad_indices = candidate.repeat(pad_count)
#         combined_idx = torch.cat([combined_idx, pad_indices], dim=0)
#     # If too many (should rarely happen), randomly sample exactly num_proposals.
#     elif combined_idx.numel() > num_proposals:
#         perm = torch.randperm(combined_idx.numel(), device=combined_idx.device)
#         combined_idx = combined_idx[perm[:num_proposals]]
    
#     # Now index all outputs.
#     filt_boxes = boxes_img[combined_idx]
#     filt_cls   = cls_img[combined_idx]
#     filt_obj   = obj_img[combined_idx]
    
#     return filt_boxes, filt_cls, filt_obj


def filter_submod_selection(selection_func_name, labels_img, boxes_img, feats_img, cls_img, obj_img, num_proposals):
    """
    Filters proposals for one image based on submodular selection strategies.
    
    Args:
        labels_img: (num_boxes,) Tensor of ROI labels.
        boxes_img:  (num_boxes, 4) Tensor of bounding boxes.
        feats_img:  (num_boxes, d_model) Tensor of proposal features.
        cls_img:    (num_boxes, num_cls) Tensor of class logits.
        obj_img:    (num_boxes, 1) Tensor of objectness scores.
        num_proposals: desired final number of proposals.
    
    Returns:
        filt_boxes: (num_proposals, 4)
        filt_feats: (num_proposals, d_model)
        filt_cls:   (num_proposals, num_cls)
        filt_obj:   (num_proposals, 1)
        filt_labels:(num_proposals,)
    """
    # Get indices for known and unknown proposals.
    known_mask = (labels_img != 0)
    unknown_mask = (labels_img == 0)
    known_idx = torch.nonzero(known_mask, as_tuple=False).squeeze(1)
    unknown_idx = torch.nonzero(unknown_mask, as_tuple=False).squeeze(1)

    Fn = get_selection_function(selection_func_name)

    similarity_metric = 'cosine'  # or 'euclidean'
    similarity_fn = get_similarity_fn(metric=similarity_metric)

    # Check if both known and unknowns exist
    if known_idx.numel() > 0 and unknown_idx.numel() > 0:
        known_feats = feats_img[known_idx]  # (n_known, d_model) # already normalized
        
        # Submod selection
        if "LDCG" in selection_func_name:
            obj_cg_background = Fn(feats_img, known_feats, similarity_fn, lamda = 0.5, nu = 1.8)
        else:
            obj_cg_background = Fn(feats_img, known_feats, similarity_fn, lamda = 0.5, nu = 5.0)
        
        bg_budget = int(0.3 * feats_img.shape[0]) # Select approx 30% of the samples as BG

        bg_selected = obj_cg_background.maximize(bg_budget)
        bg_selected_idx = [s[0] for s in bg_selected]

        feats_known_and_background = torch.vstack((known_feats, feats_img[bg_selected_idx]))

        if "LDCG" in selection_func_name:
            obj_cg_unknown = Fn(feats_img, feats_known_and_background, similarity_fn, lamda = 0.5, nu = 1.4)
        else:
            obj_cg_unknown = Fn(feats_img, feats_known_and_background, similarity_fn, lamda = 0.5, nu = 2.5)

        uk_budget = num_proposals #int(num_proposals - known_feats.shape[0])

        uk_selected = obj_cg_unknown.maximize(uk_budget)
        unknown_box_idx = torch.tensor([s[0] for s in uk_selected]).to(feats_img.device)
    else:
        print(known_idx.shape, unknown_idx.shape)
        # If there are no unknown proposals, return an empty tensor
        unknown_box_idx = torch.empty((0,), dtype=torch.long).to(feats_img.device) 
    
    return unknown_box_idx, boxes_img[unknown_box_idx], obj_img[unknown_box_idx]
