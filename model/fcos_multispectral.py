from .head import ClsCntRegHead
from .fpn_neck import FPN, FPN_multispectral
from .backbone.resnet import resnet50
import torch.nn as nn
from .loss import GenTargets, LOSS, coords_fmap2orig, LOSS_multispectral_v1, LOSS_multispectral_v2
import torch
from .config import DefaultConfig
import numpy as np


class FCOS(nn.Module):

    def __init__(self, config=None):
        super().__init__()
        if config is None:
            config = DefaultConfig  # load config
        self.backbone_rgb = resnet50(pretrained=config.pretrained,
                                     if_include_top=False)  # set backbone of VIS as ResNet 50
        self.backbone_ir = resnet50(pretrained=config.pretrained,
                                    if_include_top=False)  # set backbone of IR as ResNet 50
        self.fpn_DMAF = FPN_multispectral(config.fpn_out_channels, use_p5=config.use_p5)  # set FPN embedded with DMAF
        # self.fpn_ir=FPN(config.fpn_out_channels,use_p5=config.use_p5)
        self.head_rgb = ClsCntRegHead(config.fpn_out_channels, config.class_num,  # set detection heads of VIS
                                      config.use_GN_head, config.cnt_on_reg, config.prior)
        self.head_ir = ClsCntRegHead(config.fpn_out_channels, config.class_num,  # set detection heads of IR
                                     config.use_GN_head, config.cnt_on_reg, config.prior)
        self.config = config  # load config

    def train(self, mode=True):
        '''
        set module training mode, and frozen bn
        '''
        super().train(mode=True)

        def freeze_bn(module):  # freeze batch normalization
            if isinstance(module, nn.BatchNorm2d):
                module.eval()
            classname = module.__class__.__name__
            if classname.find('BatchNorm') != -1:
                for p in module.parameters(): p.requires_grad = False  # without gradient decent

        if self.config.freeze_bn:
            self.apply(freeze_bn)  # freeze batch normalization
            print("INFO===>success frozen BN")
        if self.config.freeze_stage_1:  # freeze the Stage 1 in Resnet
            self.backbone_rgb.freeze_stages(1)
            self.backbone_ir.freeze_stages(1)
            print("INFO===>success frozen backbone stage1")

    def forward(self, x_rgb, x_ir):
        '''
        Returns
        list [cls_logits,cnt_logits,reg_preds]
        cls_logits  list contains five [batch_size,class_num,h,w]
        cnt_logits  list contains five [batch_size,1,h,w]
        reg_preds   list contains five [batch_size,4,h,w]
        '''
        C3_r, C4_r, C5_r = self.backbone_rgb(x_rgb)  # put rgb images into backbone
        C3_i, C4_i, C5_i = self.backbone_ir(x_ir)  # put ir images into backbone
        all_P_r, all_P_i = self.fpn_DMAF([C3_r, C4_r, C5_r], [C3_i, C4_i, C5_i])  # output P3~P7 from FPN
        # all_P_i =self.fpn_rgb([C3_r,C4_r,C5_r])
        cls_logits_r, cnt_logits_r, reg_preds_r = self.head_rgb(all_P_r)  # output detected results of VIS
        cls_logits_i, cnt_logits_i, reg_preds_i = self.head_ir(all_P_i)  # output detected results of IR
        return [cls_logits_r, cnt_logits_r, reg_preds_r], [cls_logits_i, cnt_logits_i, reg_preds_i]


class DetectHead(nn.Module):
    def __init__(self, score_threshold, nms_iou_threshold, max_detection_boxes_num, strides, config=None):
        super().__init__()
        self.score_threshold = score_threshold  # confidence threshold
        self.nms_iou_threshold = nms_iou_threshold  # nms iou threshold
        self.max_detection_boxes_num = max_detection_boxes_num  # maximum number of detection boxes
        self.strides = strides  # H_ori/Hp Hp [Hp3, Hp4, Hp5, Hp6, Hp7] [8,16,32,64,128]
        if config is None:  # load config
            self.config = DefaultConfig
        else:
            self.config = config

    def forward(self, inputs):
        '''
        inputs  list [cls_logits,cnt_logits,reg_preds]
        cls_logits  list contains five [batch_size,class_num,h,w]
        cnt_logits  list contains five [batch_size,1,h,w]
        reg_preds   list contains five [batch_size,4,h,w]
        '''
        cls_logits, coords = self._reshape_cat_out(inputs[0], self.strides)  # [batch_size,sum(_h*_w),class_num]
        cnt_logits, _ = self._reshape_cat_out(inputs[1], self.strides)  # [batch_size,sum(_h*_w),1]
        reg_preds, _ = self._reshape_cat_out(inputs[2], self.strides)  # [batch_size,sum(_h*_w),4]

        cls_preds = cls_logits.sigmoid_() # sigmoid activation
        cnt_preds = cnt_logits.sigmoid_() # sigmoid activation

        coords = coords.cuda() if torch.cuda.is_available() else coords

        cls_scores, cls_classes = torch.max(cls_preds, dim=-1)  # [batch_size,sum(_h*_w)]
        if self.config.add_centerness:
            cls_scores = torch.sqrt(cls_scores * (cnt_preds.squeeze(dim=-1)))  # [batch_size,sum(_h*_w)]
        cls_classes = cls_classes + 1  # [batch_size,sum(_h*_w)]

        boxes = self._coords2boxes(coords, reg_preds)  # [batch_size,sum(_h*_w),4]

        # select topk
        max_num = min(self.max_detection_boxes_num, cls_scores.shape[-1])
        topk_ind = torch.topk(cls_scores, max_num, dim=-1, largest=True, sorted=True)[1]  # [batch_size,max_num]
        _cls_scores = []
        _cls_classes = []
        _boxes = []
        for batch in range(cls_scores.shape[0]):
            _cls_scores.append(cls_scores[batch][topk_ind[batch]])  # [max_num]
            _cls_classes.append(cls_classes[batch][topk_ind[batch]])  # [max_num]
            _boxes.append(boxes[batch][topk_ind[batch]])  # [max_num,4]
        cls_scores_topk = torch.stack(_cls_scores, dim=0)  # [batch_size,max_num]
        cls_classes_topk = torch.stack(_cls_classes, dim=0)  # [batch_size,max_num]
        boxes_topk = torch.stack(_boxes, dim=0)  # [batch_size,max_num,4]
        assert boxes_topk.shape[-1] == 4
        return self._post_process([cls_scores_topk, cls_classes_topk, boxes_topk])

    def _post_process(self, preds_topk):
        '''
        cls_scores_topk [batch_size,max_num]
        cls_classes_topk [batch_size,max_num]
        boxes_topk [batch_size,max_num,4]
        '''
        _cls_scores_post = []
        _cls_classes_post = []
        _boxes_post = []
        cls_scores_topk, cls_classes_topk, boxes_topk = preds_topk
        for batch in range(cls_classes_topk.shape[0]):
            mask = cls_scores_topk[batch] >= self.score_threshold
            _cls_scores_b = cls_scores_topk[batch][mask]  # [?]
            _cls_classes_b = cls_classes_topk[batch][mask]  # [?]
            _boxes_b = boxes_topk[batch][mask]  # [?,4]
            nms_ind = self.batched_nms(_boxes_b, _cls_scores_b, _cls_classes_b, self.nms_iou_threshold)
            _cls_scores_post.append(_cls_scores_b[nms_ind])
            _cls_classes_post.append(_cls_classes_b[nms_ind])
            _boxes_post.append(_boxes_b[nms_ind])
        scores, classes, boxes = torch.stack(_cls_scores_post, dim=0), torch.stack(_cls_classes_post,
                                                                                   dim=0), torch.stack(_boxes_post,
                                                                                                       dim=0)

        return scores, classes, boxes

    @staticmethod
    def box_nms(boxes, scores, thr):
        '''
        boxes: [?,4]
        scores: [?]
        '''
        if boxes.shape[0] == 0:
            return torch.zeros(0, device=boxes.device).long()
        assert boxes.shape[-1] == 4
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.sort(0, descending=True)[1]
        keep = []
        while order.numel() > 0:
            if order.numel() == 1:
                i = order.item()
                keep.append(i)
                break
            else:
                i = order[0].item()
                keep.append(i)

            xmin = x1[order[1:]].clamp(min=float(x1[i]))
            ymin = y1[order[1:]].clamp(min=float(y1[i]))
            xmax = x2[order[1:]].clamp(max=float(x2[i]))
            ymax = y2[order[1:]].clamp(max=float(y2[i]))
            inter = (xmax - xmin).clamp(min=0) * (ymax - ymin).clamp(min=0)
            iou = inter / (areas[i] + areas[order[1:]] - inter)
            idx = (iou <= thr).nonzero().squeeze()
            if idx.numel() == 0:
                break
            order = order[idx + 1]
        return torch.LongTensor(keep)

    def batched_nms(self, boxes, scores, idxs, iou_threshold):

        if boxes.numel() == 0:
            return torch.empty((0,), dtype=torch.int64, device=boxes.device)
        # strategy: in order to perform NMS independently per class.
        # we add an offset to all the boxes. The offset is dependent
        # only on the class idx, and is large enough so that boxes
        # from different classes do not overlap
        max_coordinate = boxes.max()
        offsets = idxs.to(boxes) * (max_coordinate + 1)
        boxes_for_nms = boxes + offsets[:, None]
        keep = self.box_nms(boxes_for_nms, scores, iou_threshold)
        return keep

    def _coords2boxes(self, coords, offsets):
        '''
        Args
        coords [sum(_h*_w),2]
        offsets [batch_size,sum(_h*_w),4] ltrb
        '''
        x1y1 = coords[None, :, :] - offsets[..., :2]
        x2y2 = coords[None, :, :] + offsets[..., 2:]  # [batch_size,sum(_h*_w),2]
        boxes = torch.cat([x1y1, x2y2], dim=-1)  # [batch_size,sum(_h*_w),4]
        return boxes

    def _reshape_cat_out(self, inputs, strides):
        '''
        Args
        inputs: list contains five [batch_size,c,_h,_w]
        Returns
        out [batch_size,sum(_h*_w),c]
        coords [sum(_h*_w),2]
        '''
        # P3~P7: five [batch_size,c,_h,_w]  strides [8,16,32,64,128]
        batch_size = inputs[0].shape[0] # batch size
        c = inputs[0].shape[1]  # channel
        out = []
        coords = []
        for pred, stride in zip(inputs, strides): # for example,(P3, 8)
            pred = pred.permute(0, 2, 3, 1) # [batch, _h, _w, c]
            coord = coords_fmap2orig(pred, stride).to(device=pred.device) # output coordination related to original images
            pred = torch.reshape(pred, [batch_size, -1, c])  #[batch, hw, c]
            out.append(pred) # out = [p3,p4,p5,p6,p7]
            coords.append(coord) # coords = [coord[p3],coord[p4],coord[p5],coord[p6],coord[p7]]
        return torch.cat(out, dim=1), torch.cat(coords, dim=0)  # concat


class DetectHead_multispectral(nn.Module):
    def __init__(self, score_threshold, nms_iou_threshold, max_detection_boxes_num, strides, config=None):
        super().__init__()
        self.score_threshold = score_threshold
        self.nms_iou_threshold = nms_iou_threshold
        self.max_detection_boxes_num = max_detection_boxes_num
        self.strides = strides
        if config is None:
            self.config = DefaultConfig
        else:
            self.config = config

    def forward(self, inputs_r, inputs_i):
        '''
        inputs  list [cls_logits,cnt_logits,reg_preds]
        cls_logits  list contains five [batch_size,class_num,h,w]
        cnt_logits  list contains five [batch_size,1,h,w]
        reg_preds   list contains five [batch_size,4,h,w]
        '''
        cls_logits_r, coords_r = self._reshape_cat_out(inputs_r[0], self.strides)  # [batch_size,sum(_h*_w),class_num]
        cnt_logits_r, _ = self._reshape_cat_out(inputs_r[1], self.strides)  # [batch_size,sum(_h*_w),1]
        reg_preds_r, _ = self._reshape_cat_out(inputs_r[2], self.strides)  # [batch_size,sum(_h*_w),4]

        cls_logits_i, coords_i = self._reshape_cat_out(inputs_i[0], self.strides)  # [batch_size,sum(_h*_w),class_num]
        cnt_logits_i, _ = self._reshape_cat_out(inputs_i[1], self.strides)  # [batch_size,sum(_h*_w),1]
        reg_preds_i, _ = self._reshape_cat_out(inputs_i[2], self.strides)  # [batch_size,sum(_h*_w),4]

        cls_preds_r = cls_logits_r.sigmoid_() # sigmoid activation
        cnt_preds_r = cnt_logits_r.sigmoid_() # sigmoid activation

        cls_preds_i = cls_logits_i.sigmoid_() # sigmoid activation
        cnt_preds_i = cnt_logits_i.sigmoid_() # sigmoid activation

        coords_r = coords_r.cuda() if torch.cuda.is_available() else coords_r  # generate coordiniation of rgb
        coords_i = coords_i.cuda() if torch.cuda.is_available() else coords_i  # generate coordiniation of ir

        cls_scores_r, cls_classes_r = torch.max(cls_preds_r, dim=-1)  # input: [batch_size,sum(_h*_w),class_num] output: max_value & max_value_index[batch_size,sum(_h*_w)] rgb
        cls_scores_i, cls_classes_i = torch.max(cls_preds_i, dim=-1)  # input: [batch_size,sum(_h*_w),class_num] output: max_value & max_value_index[batch_size,sum(_h*_w)] ir

        # v1 vs v2: As for v1, centerness_loss_sqrt == False, cls_scores = cls_scores*cnt_preds;
        # as for v2, centerness_loss_sqrt == True, cls_scores = sqrt(cls_scores*cnt_preds)
        if self.config.centerness_loss_sqrt:
            if self.config.add_centerness:
                cls_scores_r = torch.sqrt(cls_scores_r * (cnt_preds_r.squeeze(dim=-1)))  # [batch_size,sum(_h*_w)]
                cls_scores_i = torch.sqrt(cls_scores_i * (cnt_preds_i.squeeze(dim=-1)))  # [batch_size,sum(_h*_w)]

            else:
                pass
        else:
            if self.config.add_centerness:
                cls_scores_r = cls_scores_r * (cnt_preds_r.squeeze(dim=-1))  # [batch_size,sum(_h*_w)]
                cls_scores_i = cls_scores_i * (cnt_preds_i.squeeze(dim=-1))  # [batch_size,sum(_h*_w)]
            else:
                pass
        cls_classes_r = cls_classes_r + 1  # [batch_size,sum(_h*_w)] original class index starts from 0
        cls_classes_i = cls_classes_i + 1  # [batch_size,sum(_h*_w)] original class index starts from 0

        boxes_r = self._coords2boxes(coords_r, reg_preds_r)  # [batch_size,sum(_h*_w),4] from offset(l,r,t,b) to (x_lt,y_lt,x_rb,y_rb)
        boxes_i = self._coords2boxes(coords_i, reg_preds_i)  # [batch_size,sum(_h*_w),4] from offset(l,r,t,b) to (x_lt,y_lt,x_rb,y_rb)

        boxes = torch.concat([boxes_r, boxes_i], dim=1) # loss v2: concat boxes
        cls_scores = torch.concat([cls_scores_r, cls_scores_i], dim=1) # loss v2: concat scores
        cls_classes = torch.concat([cls_classes_r, cls_classes_i], dim=1) # loss v2: concat classes
        # select topk
        max_num = min(self.max_detection_boxes_num, cls_scores.shape[-1]) # top k boxes can be used for detection
        # max_num_i = min(self.max_detection_boxes_num, cls_scores_i.shape[-1])
        topk_ind = torch.topk(cls_scores, max_num, dim=-1, largest=True, sorted=True)[1]  # [batch_size,max_num] find the indexes of top k sample
        _cls_scores = []
        _cls_classes = []
        _boxes = []
        for batch in range(cls_scores.shape[0]):
            _cls_scores.append(cls_scores[batch][topk_ind[batch]])  # [max_num] top k samples for every image
            _cls_classes.append(cls_classes[batch][topk_ind[batch]])  # [max_num] top k samples for every image
            _boxes.append(boxes[batch][topk_ind[batch]])  # [max_num,4] top k samples for every image
        cls_scores_topk = torch.stack(_cls_scores, dim=0)  # [batch_size,max_num]
        cls_classes_topk = torch.stack(_cls_classes, dim=0)  # [batch_size,max_num]
        boxes_topk = torch.stack(_boxes, dim=0)  # [batch_size,max_num,4]
        assert boxes_topk.shape[-1] == 4
        return self._post_process([cls_scores_topk, cls_classes_topk, boxes_topk])

    def _post_process(self, preds_topk):
        '''
        cls_scores_topk [batch_size,max_num]
        cls_classes_topk [batch_size,max_num]
        boxes_topk [batch_size,max_num,4]
        '''
        _cls_scores_post = []
        _cls_classes_post = []
        _boxes_post = []
        cls_scores_topk, cls_classes_topk, boxes_topk = preds_topk
        for batch in range(cls_classes_topk.shape[0]):
            mask = cls_scores_topk[batch] >= self.score_threshold # indice of boxes whose score are larger than score_threhold
            _cls_scores_b = cls_scores_topk[batch][mask]  # [?] according to "mask", extract corresponding scores
            _cls_classes_b = cls_classes_topk[batch][mask]  # [?] according to "mask", extract corresponding classes
            _boxes_b = boxes_topk[batch][mask]  # [?,4] according to "mask", extract corresponding boxes
            nms_ind = self.batched_nms(_boxes_b, _cls_scores_b, _cls_classes_b, self.nms_iou_threshold) # NMS return final output boxes' indice
            _cls_scores_post.append(_cls_scores_b[nms_ind])  # list
            _cls_classes_post.append(_cls_classes_b[nms_ind]) # list
            _boxes_post.append(_boxes_b[nms_ind]) # list
        # scores, classes, boxes = torch.stack(_cls_scores_post, dim=0), torch.stack(_cls_classes_post,  # batch = 1
        #                                                                            dim=0), torch.stack(_boxes_post,
        #                                                                                                dim=0)
        scores, classes, boxes = _cls_scores_post, _cls_classes_post, _boxes_post  # return list[batch_size]

        return scores, classes, boxes

    @staticmethod
    def box_nms(boxes, scores, thr):
        '''
        boxes: [?,4]
        scores: [?]
        '''
        if boxes.shape[0] == 0: # no detected boxes
            return torch.zeros(0, device=boxes.device).long()
        assert boxes.shape[-1] == 4
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3] # x_lt, y_lt, x_rb, y_rb
        areas = (x2 - x1 + 1) * (y2 - y1 + 1) # areas of predicted boxes
        order = scores.sort(0, descending=True)[1]  # values of sorted scores from high to low
        keep = []
        while order.numel() > 0:  # number of scores
            if order.numel() == 1:  # if the number of scores is equal to 1
                i = order.item()
                keep.append(i)
                break
            else:
                i = order[0].item() # the first one with highest score
                keep.append(i) # directly add it into the final output boxes

            xmin = x1[order[1:]].clamp(min=float(x1[i]))  # x_lt of minimum boxes
            ymin = y1[order[1:]].clamp(min=float(y1[i]))  # y_lt of minimum boxes
            xmax = x2[order[1:]].clamp(max=float(x2[i]))  # x_rb of minimum boxes
            ymax = y2[order[1:]].clamp(max=float(y2[i]))  # y_rb of minimum boxes
            # inter = (xmax - xmin).clamp(min=0) * (ymax - ymin).clamp(min=0)
            inter = (xmax - xmin + 1).clamp(min=0) * (ymax - ymin + 1).clamp(min=0)  # calcualte intersection
            # areas1 = areas[i] + areas[order[1:]] - inter
            iou = inter / (areas[i] + areas[order[1:]] - inter)  # calculate iou
            idx = (iou <= thr).nonzero().squeeze()  # if iou < threshold, reserve
            if idx.numel() == 0: # no remaining boxes, break the loop
                break
            order = order[idx + 1] #
        return torch.LongTensor(keep)

    def batched_nms(self, boxes, scores, idxs, iou_threshold):

        if boxes.numel() == 0: # no detected boxes
            return torch.empty((0,), dtype=torch.int64, device=boxes.device)
        # strategy: in order to perform NMS independently per class.
        # we add an offset to all the boxes. The offset is dependent
        # only on the class idx, and is large enough so that boxes
        # from different classes do not overlap
        max_coordinate = boxes.max()
        offsets = idxs.to(boxes) * (max_coordinate + 1)
        boxes_for_nms = boxes + offsets[:, None]
        keep = self.box_nms(boxes_for_nms, scores, iou_threshold)
        return keep

    def _coords2boxes(self, coords, offsets):
        '''
        Args
        coords [sum(_h*_w),2]
        offsets [batch_size,sum(_h*_w),4] ltrb
        '''
        x1y1 = coords[None, :, :] - offsets[..., :2]
        x2y2 = coords[None, :, :] + offsets[..., 2:]  # [batch_size,sum(_h*_w),2]
        boxes = torch.cat([x1y1, x2y2], dim=-1)  # [batch_size,sum(_h*_w),4]
        return boxes

    def _reshape_cat_out(self, inputs, strides):
        '''
        Args
        inputs: list contains five [batch_size,c,_h,_w]
        Returns
        out [batch_size,sum(_h*_w),c]
        coords [sum(_h*_w),2]
        '''
        # P3~P7: five [batch_size,c,_h,_w]  strides [8,16,32,64,128]
        batch_size = inputs[0].shape[0]  # batch size
        c = inputs[0].shape[1]  # channel
        out = []
        coords = []
        for pred, stride in zip(inputs, strides):  # for example,(P3, 8)
            pred = pred.permute(0, 2, 3, 1)  # [batch, _h, _w, c]
            coord = coords_fmap2orig(pred, stride).to(
                device=pred.device)  # output coordination related to original images
            pred = torch.reshape(pred, [batch_size, -1, c])  # [batch, hw, c]
            out.append(pred)  # out = [p3,p4,p5,p6,p7]
            coords.append(coord)  # coords = [coord[p3],coord[p4],coord[p5],coord[p6],coord[p7]]
        return torch.cat(out, dim=1), torch.cat(coords, dim=0)  # concat


class ClipBoxes(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, batch_imgs, batch_boxes):
        # pledge that the boxes do not exceed the region of images
        for i in range(len(batch_boxes)):
            batch_boxes[i] = batch_boxes[i].clamp_(min=0)
            h, w = batch_imgs.shape[2:]
            batch_boxes[i][..., [0, 2]] = batch_boxes[i][..., [0, 2]].clamp_(max=w - 1)
            batch_boxes[i][..., [1, 3]] = batch_boxes[i][..., [1, 3]].clamp_(max=h - 1)
        return batch_boxes


class FCOSDetector(nn.Module):
    def __init__(self, mode="training", loss="v1", config=None):
        super().__init__()
        if config is None:
            config = DefaultConfig  # load config
        self.mode = mode  # training or inference
        self.fcos_body = FCOS(config=config)  # load FCOS Model
        self.stride = config.strides  # strides
        self.norm_reg_targets = config.norm_reg_targets  # True: fcos v2; False: fcos v1
        if mode == "training":
            self.target_layer_r = GenTargets(strides=config.strides, limit_range=config.limit_range,
                                             # VIS fcos generate the targets of locations (x, y)
                                             sample_radiu_ratio=config.sample_radiu_ratio,
                                             norm_reg_targets=config.norm_reg_targets)
            self.target_layer_i = GenTargets(strides=config.strides, limit_range=config.limit_range,
                                             # IR fcos generate the targets of locations (x, y)
                                             sample_radiu_ratio=config.sample_radiu_ratio,
                                             norm_reg_targets=config.norm_reg_targets)

            # loss version:
            # v1: two types of detection head are separated, two groups of losses
            # v2: two types of detection head are fused, only one group of losses
            if loss == "v1":
                self.loss_layer = LOSS_multispectral_v1(config=config)
            elif loss == "v2":
                self.loss_layer = LOSS_multispectral_v2(config=config)
        elif mode == "inference":
            self.detection_head = DetectHead_multispectral(config.score_threshold, config.nms_iou_threshold,
                                                           # detection heads
                                                           config.max_detection_boxes_num, config.strides, config)
            self.clip_boxes = ClipBoxes()  # avoid generated boxes exceeding the region of images

    def forward(self, inputs):
        '''
        inputs
        [training] list  batch_rgb_imgs, batch_ir_imgs, batch_boxes,batch_classes
        [inference] img
        '''
        if self.mode == "training":
            batch_rgb_imgs, batch_ir_imgs, batch_boxes, batch_classes = inputs  # inputs [batch_rgb_imgs, batch_ir_imgs,batch_boxes, batch_classes]
            out_r, out_i = self.fcos_body(batch_rgb_imgs,
                                          batch_ir_imgs)  # out_r:  [cls_logits_r,cnt_logits_r,reg_preds_r] out_i:  [cls_logits_i,cnt_logits_i,reg_preds_i]
            targets_r = self.target_layer_r([out_r, batch_boxes, batch_classes])  # targets of different samples of VIS
            targets_i = self.target_layer_i([out_i, batch_boxes, batch_classes])  # targets of different samples of IR
            losses = self.loss_layer([out_r, out_i, targets_r, targets_i])  # compute loss

            # loss version:
            # v1: [cls_loss,cnt_loss,reg_loss,total_loss]
            # v2: [cls_loss_r,cnt_loss_r, reg_loss_r, cls_loss_i, cnt_loss_i, reg_loss_i, total_loss]
            return losses
        elif self.mode == "inference":
            # raise NotImplementedError("no implement inference model")
            '''
            for inference mode, img should preprocessed before feeding in net
            '''
            batch_rgb_imgs, batch_ir_imgs = inputs  # inputs [batch_rgb_imgs, batch_ir_imgs]
            out_r, out_i = self.fcos_body(batch_rgb_imgs,
                                          batch_ir_imgs)  # out_r:  [cls_logits_r,cnt_logits_r,reg_preds_r] out_i:  [cls_logits_i,cnt_logits_i,reg_preds_i]
            scores, classes, boxes = self.detection_head(out_r, out_i)  # output predicted boxes, scores and classes

            # v1 vs v2: As for v1, norm_reg_targets == False, directly inference the real boxes;
            # as for v2, norm_reg_targets == True, the boxes(offsets) have been normalized, so multiply them with strides
            if self.norm_reg_targets:
                for i in range(len(self.stride)):
                    boxes[i] = boxes[i] * torch.tensor(self.stride[i])

            boxes = self.clip_boxes(batch_rgb_imgs, boxes)  # avoid generated boxes exceeding the region of images

            return scores, classes, boxes




