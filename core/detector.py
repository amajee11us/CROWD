import math
import random
from typing import List
from collections import namedtuple

import torch
import torch.nn.functional as F
from torch import nn

from detectron2.layers import batched_nms
from detectron2.modeling import META_ARCH_REGISTRY, build_backbone, detector_postprocess

from detectron2.structures import Boxes, ImageList, Instances

from .loss import SetCriterionDynamicK, HungarianMatcherDynamicK
from .head import DynamicHead
from .util.box_ops import box_cxcywh_to_xyxy, box_xyxy_to_cxcywh
from .util.misc import nested_tensor_from_tensor_list
from .selector import filter_similarity, filter_submod_selection

__all__ = ["RandBox"]

ModelPrediction = namedtuple('ModelPrediction', ['pred_noise', 'pred_x_start'])


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


def extract(a, t, x_shape):
    """extract the appropriate  t  index for a batch of indices"""
    batch_size = t.shape[0]
    out = a.gather(-1, t)
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


@META_ARCH_REGISTRY.register()
class RandBox(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.device = torch.device(cfg.MODEL.DEVICE)

        self.in_features = cfg.MODEL.ROI_HEADS.IN_FEATURES
        self.num_classes = cfg.MODEL.NUM_CLASSES     # 80 classes + 1 unknown + 1 bg
        self.num_proposals = cfg.MODEL.NUM_PROPOSALS # This is 500 in OrthogonalDet
        self.hidden_dim = cfg.MODEL.HIDDEN_DIM
        self.num_heads = cfg.MODEL.NUM_HEADS
        self.sampling_method = cfg.MODEL.SAMPLING_METHOD
        self.disentangled = cfg.MODEL.DISENTANGLED  
        # Build Backbone.
        self.backbone = build_backbone(cfg)
        self.size_divisibility = self.backbone.size_divisibility

        # build diffusion
        timesteps = 1000
        sampling_timesteps = cfg.MODEL.SAMPLE_STEP
        self.objective = 'pred_x0'
        betas = cosine_beta_schedule(timesteps)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.)
        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.multiple_sample = cfg.MODEL.M_STEP
        self.sampling_timesteps = default(sampling_timesteps, timesteps)
        assert self.sampling_timesteps <= timesteps
        self.ddim_sampling_eta = 1.
        self.self_condition = False
        self.scale = cfg.MODEL.SNR_SCALE

        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)
        # calculations for diffusion q(x_t | x_{t-1}) and others

        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        self.register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)

        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

        self.register_buffer('posterior_variance', posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain

        self.register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min=1e-20)))
        self.register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        self.register_buffer('posterior_mean_coef2',
                             (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))
        self.x_dic = torch.rand((10000, 4))
        self.x_meta = torch.arange(start=-2, end=2, step=0.4)
        for i1 in range(10):
            for i2 in range(10):
                for i3 in range(10):
                    for i4 in range(10):
                        self.x_dic[i1 * 1000 + i2 * 100 + i3 * 10 + i4][0], \
                        self.x_dic[i1 * 1000 + i2 * 100 + i3 * 10 + i4][1], \
                        self.x_dic[i1 * 1000 + i2 * 100 + i3 * 10 + i4][2], \
                        self.x_dic[i1 * 1000 + i2 * 100 + i3 * 10 + i4][3] = self.x_meta[i1], self.x_meta[i2], \
                        self.x_meta[i3], self.x_meta[i4]
        self.x_dic = self.x_dic[torch.randperm(self.x_dic.size(0))]
        # Build Dynamic Head.
        self.head = DynamicHead(cfg=cfg, roi_input_shape=self.backbone.output_shape())
        # Loss parameters:
        class_weight = cfg.MODEL.CLASS_WEIGHT
        giou_weight = cfg.MODEL.GIOU_WEIGHT
        l1_weight = cfg.MODEL.L1_WEIGHT
        nc_weight = cfg.MODEL.NC_WEIGHT
        no_object_weight = cfg.MODEL.NO_OBJECT_WEIGHT
        decorr_weight = cfg.MODEL.DECORR_WEIGHT
        self.deep_supervision = cfg.MODEL.DEEP_SUPERVISION
        self.use_nms = cfg.MODEL.USE_NMS

        # Build Criterion.
        matcher = HungarianMatcherDynamicK(
            cfg=cfg, cost_class=class_weight, cost_bbox=l1_weight, cost_giou=giou_weight
        )
        weight_dict = {"loss_ce": class_weight, "loss_bbox": l1_weight, "loss_giou": giou_weight,
                       "loss_nc_ce": nc_weight, "loss_decorr": decorr_weight}
        if self.deep_supervision:
            aux_weight_dict = {}
            for i in range(self.num_heads - 1):
                aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)

        losses = ["labels", "boxes"]
        if cfg.MODEL.NC:
            losses += ["nc_labels"]
        if decorr_weight > 0:
            losses += ["decorr"]

        self.criterion = SetCriterionDynamicK(
            cfg=cfg, num_classes=self.num_classes, matcher=matcher, weight_dict=weight_dict, eos_coef=no_object_weight,
            losses=losses)

        pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).to(self.device).view(3, 1, 1)
        pixel_std = torch.Tensor(cfg.MODEL.PIXEL_STD).to(self.device).view(3, 1, 1)
        self.normalizer = lambda x: (x - pixel_mean) / pixel_std
        self.to(self.device)

    def predict_noise_from_start(self, x_t, t, x0):
        return (
                (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) /
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )

    def model_predictions(self, backbone_feats, images_whwh, x, t, x_self_cond=None, clip_x_start=False, sample_i=0):
        if self.sampling_method == 'Random':
            x_boxes = torch.clamp(x, min=-1 * self.scale, max=self.scale)
            x_boxes = ((x_boxes / self.scale) + 1) / 2
        else:
            x_boxes = self.x_dic.to(x.device)[self.num_proposals * sample_i:self.num_proposals * (sample_i + 1), :]
            x_boxes = ((x_boxes / self.scale) + 1) / 2

        x_boxes = box_cxcywh_to_xyxy(x_boxes)
        x_boxes = x_boxes * images_whwh[:, None, :]
        outputs_class, outputs_objectness, outputs_coord,_ = self.head(backbone_feats, x_boxes, t, None)

        x_start = outputs_coord[-1]  # (batch, num_proposals, 4) predict boxes: absolute coordinates (x1, y1, x2, y2)
        x_start = x_start / images_whwh[:, None, :]
        x_start = box_xyxy_to_cxcywh(x_start)
        x_start = (x_start * 2 - 1.) * self.scale
        x_start = torch.clamp(x_start, min=-1 * self.scale, max=self.scale)
        pred_noise = self.predict_noise_from_start(x, t, x_start)

        return ModelPrediction(pred_noise, x_start), outputs_class, outputs_objectness, outputs_coord

    @torch.no_grad()
    def ddim_sample(self, batched_inputs, backbone_feats, images_whwh, images, clip_denoised=True, do_postprocess=True):
        batch = images_whwh.shape[0]
        shape = (batch, self.num_proposals, 4)
        total_timesteps, sampling_timesteps, eta, objective = self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta, self.objective

        # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))  # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        img = torch.randn(shape, device=self.device)

        x_start = None
        if self.sampling_method == 'Random':
            for time, time_next in time_pairs:
                time_cond = torch.full((batch,), time, device=self.device, dtype=torch.long)
                self_cond = x_start if self.self_condition else None

                preds, class_cat, objectness_cat, coord_cat = self.model_predictions(backbone_feats, images_whwh, img, time_cond,
                                                                     self_cond, clip_x_start=clip_denoised)
                pred_noise, x_start = preds.pred_noise, preds.pred_x_start
        else:
            for sample_step in range(self.multiple_sample):
                for time, time_next in time_pairs:
                    time_cond = torch.full((batch,), time, device=self.device, dtype=torch.long)
                    self_cond = x_start if self.self_condition else None

                    preds, outputs_class, outputs_objectness, outputs_coord = self.model_predictions(backbone_feats, images_whwh, img,
                                                                                 time_cond,
                                                                                 self_cond, clip_x_start=clip_denoised,
                                                                                 sample_i=sample_step)
                if sample_step == 0:
                    class_cat = outputs_class
                    objectness_cat = outputs_objectness
                    coord_cat = outputs_coord
                else:
                    class_cat = torch.cat((class_cat, outputs_class), 2)
                    objectness_cat = torch.cat((objectness_cat, outputs_objectness), 2)
                    coord_cat = torch.cat((coord_cat, outputs_coord), 2)

        results = self.inference(class_cat[-1], objectness_cat[-1], coord_cat[-1], images.image_sizes)

        if do_postprocess:
            processed_results = []
            for results_per_image, input_per_image, image_size in zip(results, batched_inputs, images.image_sizes):
                height = input_per_image.get("height", image_size[0])
                width = input_per_image.get("width", image_size[1])
                r = detector_postprocess(results_per_image, height, width)
                processed_results.append({"instances": r})
            return processed_results

    # forward diffusion
    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def forward(self, batched_inputs, do_postprocess=True):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:

                * image: Tensor, image in (C, H, W) format.
                * instances: Instances

                Other information that's included in the original dicts, such as:

                * "height", "width" (int): the output resolution of the model, used in inference.
                  See :meth:`postprocess` for details.
        """
        images, images_whwh = self.preprocess_image(batched_inputs)
        if isinstance(images, (list, torch.Tensor)):
            images = nested_tensor_from_tensor_list(images)

        # Feature Extraction.
        src = self.backbone(images.tensor)
        features = list()
        for f in self.in_features:
            feature = src[f]
            features.append(feature)

        # Prepare Proposals.
        if not self.training:
            results = self.ddim_sample(batched_inputs, features, images_whwh, images, do_postprocess=do_postprocess)
            return results

        if self.training:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
            targets, x_boxes, noises, t, roi_labels = self.prepare_targets(gt_instances)
            t = t.squeeze(-1)
            x_boxes = x_boxes * images_whwh[:, None, :]

            pre_filter_outputs_class, pre_filter_output_objectness, pre_filter_outputs_coord, pre_filter_proposal_features = self.head(
                                                                                     features, x_boxes, t, None, roi_labels=roi_labels)
            
            print(roi_labels.shape[0])
            filtered_results = [
                    filter_submod_selection(
                        roi_labels[i], 
                        pre_filter_outputs_coord[-1][i], 
                        pre_filter_proposal_features[i], 
                        pre_filter_outputs_class[-1][i], 
                        pre_filter_output_objectness[-1][i],
                        self.num_proposals
                    )
                    for i in range(roi_labels.shape[0])
                ]
            # Unzip the results (each is a tuple of tensors).
            outputs_coord, outputs_class, output_objectness = map(
                lambda *x: torch.stack(x, dim=0), *filtered_results
            )
            
            # print(outputs_coord.shape, output_objectness.shape, outputs_class.shape)            
            output = {'pred_logits': outputs_class, 'pred_objectness': output_objectness, 'pred_boxes': outputs_coord}
            if self.deep_supervision:
                output['aux_outputs'] = [{'pred_logits': a, 'pred_objectness': b, 'pred_boxes': c}
                                         for a, b, c in zip(pre_filter_outputs_class[:-1], pre_filter_output_objectness[:-1], pre_filter_outputs_coord[:-1])]
            # print(output['pred_logits'].shape)
            
            # exit()
            loss_dict = self.criterion(output, targets)
            weight_dict = self.criterion.weight_dict
            for k in loss_dict.keys():
                if k in weight_dict:
                    loss_dict[k] *= weight_dict[k]
            return loss_dict

    def prepare_diffusion_concat(self, gt_boxes):
        """
        Given ground-truth boxes (normalized as (cx, cy, w, h)), this function prepares
        an initial set of proposals for the diffusion process.
        
        Here we generate twice as many proposals as the final target number.
        """
        # Use double the number of proposals initially.
        initial_num = self.num_proposals * 2

        t = torch.randint(0, self.num_timesteps, (1,), device=self.device).long()
        noise = torch.randn(initial_num, 4, device=self.device)

        num_gt = gt_boxes.shape[0]
        if not num_gt:  # if no gt boxes, generate a dummy gt box
            gt_boxes = torch.as_tensor([[0.5, 0.5, 1., 1.]], dtype=torch.float, device=self.device)
            num_gt = 1

        # Generate placeholders for the unknown proposals (this is computed but may be unused
        # if you later merge with gt; we keep it here for consistency).
        box_placeholder = torch.randn(initial_num - num_gt, 4, device=self.device) / 6.0 + 0.5
        box_placeholder[:, 2:] = torch.clip(box_placeholder[:, 2:], min=1e-4)
        
        # Generate the initial proposals (randomly) with size initial_num.
        x_start = torch.randn(initial_num, 4, device=self.device)
        x_start = (x_start * 2. - 1.) * self.scale

        # Apply the diffusion forward process.
        x = self.q_sample(x_start=x_start, t=t, noise=noise)
        x = torch.clamp(x, min=-1 * self.scale, max=self.scale)
        x = ((x / self.scale) + 1) / 2.

        diff_boxes = box_cxcywh_to_xyxy(x)

        return diff_boxes, noise, t

    def prepare_targets(self, targets):
        new_targets = []
        diffused_boxes = []
        noises = []
        ts = []
        roi_labels_list = []  # To store per-image ROI labels.

        for targets_per_image in targets:
            target = {}
            h, w = targets_per_image.image_size
            image_size_xyxy = torch.as_tensor([w, h, w, h], dtype=torch.float, device=self.device)
            gt_classes = targets_per_image.gt_classes
            gt_boxes = targets_per_image.gt_boxes.tensor / image_size_xyxy
            gt_boxes = box_xyxy_to_cxcywh(gt_boxes)
            
            # Generate diffused proposals using the doubled initial number.
            d_boxes, d_noise, d_t = self.prepare_diffusion_concat(gt_boxes)
            diffused_boxes.append(d_boxes)
            noises.append(d_noise)
            ts.append(d_t)
            
            target["labels"] = gt_classes.to(self.device)
            target["boxes"] = gt_boxes.to(self.device)
            target["boxes_xyxy"] = targets_per_image.gt_boxes.tensor.to(self.device)
            target["image_size_xyxy"] = image_size_xyxy.to(self.device)
            image_size_xyxy_tgt = image_size_xyxy.unsqueeze(0).repeat(len(gt_boxes), 1)
            target["image_size_xyxy_tgt"] = image_size_xyxy_tgt.to(self.device)
            target["area"] = targets_per_image.gt_boxes.area().to(self.device)
            new_targets.append(target)

            # ---- Create the ROI labels for this image ----
            # We now create a tensor of length `initial_num` (i.e. double self.num_proposals).
            num_gt = gt_classes.shape[0]
            initial_num = self.num_proposals * 2
            roi_labels_img = torch.full((initial_num,), -1, dtype=gt_classes.dtype, device=self.device)
            # For the first num_gt proposals, assign the ground-truth class labels.
            if num_gt > 0:
                roi_labels_img[:num_gt] = gt_classes
            roi_labels_list.append(roi_labels_img)
        
        # Stack all per-image roi_labels into a single tensor of shape (batch_size, initial_num).
        roi_labels = torch.stack(roi_labels_list, dim=0)

        return new_targets, torch.stack(diffused_boxes), torch.stack(noises), torch.stack(ts), roi_labels


    def inference(self, box_cls, box_objectness, box_pred, image_sizes):
        """
        Arguments:
            box_cls (Tensor): tensor of shape (batch_size, num_proposals, K).
                The tensor predicts the classification probability for each proposal.
            box_objectness (Tensor): tensors of shape (batch_size, num_proposals, 1).
                The tensor predicts the objectness for each proposal.
            box_pred (Tensor): tensors of shape (batch_size, num_proposals, 4).
                The tensor predicts 4-vector (x,y,w,h) box
                regression values for every proposal
            image_sizes (List[torch.Size]): the input image sizes

        Returns:
            results (List[Instances]): a list of #images elements.
        """
        assert len(box_cls) == len(image_sizes)
        results = []

        if self.sampling_method == 'Random':
            multiple_sample = 1
        else:
            multiple_sample = self.multiple_sample
            
        # scores - represent the loss values

        if self.disentangled == 0:
            scores = torch.sigmoid(box_cls)
        else:
            scores = torch.softmax(box_cls, dim=-1) * box_objectness
        labels = torch.arange(self.num_classes, device=self.device). \
            unsqueeze(0).repeat(self.num_proposals * multiple_sample, 1).flatten(0, 1)

        for i, (scores_per_image, box_pred_per_image, image_size) in enumerate(zip(
                scores, box_pred, image_sizes
        )):
            scores_per_image, topk_indices = scores_per_image.flatten(0, 1).topk(
                self.num_proposals * multiple_sample, sorted=False)
            labels_per_image = labels[topk_indices]
            box_pred_per_image = box_pred_per_image.view(-1, 1, 4).repeat(1, self.num_classes, 1).view(-1, 4)
            box_pred_per_image = box_pred_per_image[topk_indices]

            if self.use_nms:
                keep = batched_nms(box_pred_per_image, scores_per_image, labels_per_image, 0.6)
                box_pred_per_image = box_pred_per_image[keep]
                scores_per_image = scores_per_image[keep]
                labels_per_image = labels_per_image[keep]

            # rescale scores to accommodate score threshold
            if self.disentangled == 2:
                scores_per_image[labels_per_image != self.num_classes-1] *= 0.75
                scores_per_image[labels_per_image == self.num_classes-1] *= 2

            result = Instances(image_size)
            result.pred_boxes = Boxes(box_pred_per_image)
            result.scores = scores_per_image
            result.pred_classes = labels_per_image
            results.append(result)

        return results

    def preprocess_image(self, batched_inputs):
        """
        Normalize, pad and batch the input images.
        """
        images = [self.normalizer(x["image"].to(self.device)) for x in batched_inputs]
        images = ImageList.from_tensors(images, self.size_divisibility)

        images_whwh = list()
        for bi in batched_inputs:
            h, w = bi["image"].shape[-2:]
            images_whwh.append(torch.tensor([w, h, w, h], dtype=torch.float32, device=self.device))
        images_whwh = torch.stack(images_whwh)

        return images, images_whwh
