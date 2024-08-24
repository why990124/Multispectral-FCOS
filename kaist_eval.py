import os

from pycocotools.cocoeval import COCOeval
import numpy as np
import json
from tqdm import tqdm
from torchvision.datasets import CocoDetection
from torchvision import transforms
import cv2
from model.fcos_multispectral import FCOSDetector
from dataset.KAIST_dataset import KaistDetection
import torch
from utils.mr_fppi import mr_fppi, lamr
import time
from model.config import DefaultConfig
from model.config_v1 import DefaultConfig_v1
from model.config_v2 import DefaultConfig_v2

def fppi_per_class(tp, conf, pred_cls, target_cls, image_num, plot=False, save_dir='.', names=(), return_plt=False):
    """ Compute the false positives per image (FPPW) metric, given the recall and precision curves.
    Source:
    # Arguments
        tp:  True positives (nparray, nx1 or nx10).
        conf:  Objectness value from 0-1 (nparray).
        pred_cls:  Predicted object classes (nparray).
        target_cls:  True object classes (nparray).
        plot:  Plot precision-recall curve at mAP@0.5
        save_dir:  Plot save directory
    # Returns
        The fppi curve
    """
    # Sort by objectness
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    # Find unique classes
    unique_classes = np.unique(target_cls)
    nc = unique_classes.shape[0]  # number of classes, number of detections

    # Create Precision-Recall curve and compute AP for each class
    px, py = np.linspace(0, 1, 1000), np.linspace(0, 100, 1000)  # for plotting
    r = np.zeros((nc, 1000))
    miss_rate = np.zeros((nc, 1000))
    fppi = np.zeros((nc, 1000))
    miss_rate_at_fppi = np.zeros((nc, 3))  # missrate at fppi 1, 0.1, 0.01
    p_miss_rate = np.array([1, 0.1, 0.01])
    for ci, c in enumerate(unique_classes):
        i = pred_cls == c
        n_l = (target_cls == c).sum()  # number of labels
        n_p = i.sum()  # number of predictions

        if n_p == 0 or n_l == 0:
            continue
        else:
            # Accumulate FPs and TPs
            fpc = (1 - tp[i]).cumsum(0)
            tpc = tp[i].cumsum(0)

            # Recall
            recall = tpc / (n_l + 1e-16)  # recall curve
            r[ci] = np.interp(-px, -conf[i], recall[:, 0], left=0)  # negative x, xp because xp decreases
            miss_rate[ci] = 1 - r[ci]

            fp_per_image = fpc / image_num
            fppi[ci] = np.interp(-px, -conf[i], fp_per_image[:, 0], left=0)

            miss_rate_at_fppi[ci] = np.interp(-p_miss_rate, -fppi[ci], miss_rate[ci])

    # if plot:
    #     fig = plot_fppi_curve(fppi, miss_rate, miss_rate_at_fppi, Path(save_dir) / 'fppi_curve.png', names)

    # if return_plt:
    #     return fppi, miss_rate, miss_rate_at_fppi, fig

    return miss_rate, fppi, miss_rate_at_fppi


def IOU(boxA, boxB):
    boxA = [int(x) for x in boxA]
    boxB = [int(x) for x in boxB]

    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    iou = interArea / float(boxAArea + boxBArea - interArea)

    return iou


class KAIST_Generator(KaistDetection):
    CLASSES_NAME = (
    '__back_ground__', 'person')
    def __init__(self,rgb_imgs_path,ir_imgs_path,anno_path,resize_size=[800,1333]):
        super().__init__(rgb_imgs_path,ir_imgs_path,anno_path)
        # super.__init__ content :self.ir_root = ir_root;  self.coco = COCO(annFile); self.ids = list(sorted(self.coco.imgs.keys()))

        print("INFO====>check annos, filtering invalid data......")
        ids=[]
        for id in self.ids:   # coco.imgs.keys img ids in annotations eg, 20190000001
            ann_id=self.coco.getAnnIds(imgIds=id,iscrowd=None)  # coco's function, return annotation ids in this img
            ann=self.coco.loadAnns(ann_id)  # according to ann_id, load annotations
            if self._has_valid_annotation(ann):  # if annotation and box exist, true
                ids.append(id)
        self.ids=ids  # img ids
        self.category2id = {v: i + 1 for i, v in enumerate(self.coco.getCatIds())}  # return catagory id {"class": "class id" }
        self.id2category = {v: k for k, v in self.category2id.items()} # return id {"class id": "class" }

        self.resize_size=resize_size  # resize images
        self.mean=[0.40789654, 0.44719302, 0.47026115]
        self.std=[0.28863828, 0.27408164, 0.27809835]
        

    def __getitem__(self,index):
        
        rgb_img,ir_img, ann, id=super().__getitem__(index)

        ann = [o for o in ann if o['iscrowd'] == 0]
        boxes = [o['bbox'] for o in ann]
        boxes=np.array(boxes,dtype=np.float32)
        #xywh-->xyxy
        boxes[...,2:]=boxes[...,2:]+boxes[...,:2]
        rgb_img=np.array(rgb_img)
        ir_img=np.array(ir_img)

        rgb_img, ir_img, boxes, scale = self.preprocess_img_boxes(rgb_img, ir_img, boxes, self.resize_size)  # add ir_img
        # img=draw_bboxes(img,boxes)
        

        classes = [o['category_id'] for o in ann]
        classes = [self.category2id[c] for c in classes]
        


        rgb_img = transforms.ToTensor()(rgb_img)
        ir_img = transforms.ToTensor()(ir_img)
        # img= transforms.Normalize(self.mean, self.std,inplace=True)(img)
        boxes=torch.from_numpy(boxes)
        # classes=np.array(classes,dtype=np.int64)
        classes = torch.LongTensor(classes)

        return rgb_img, ir_img, boxes, classes, scale, id

    def preprocess_img_boxes(self, r_img, i_img, boxes, input_ksize):
        '''
        resize image and bboxes
        Returns
        image_paded: input_ksize
        bboxes: [None,4]
        '''
        min_side, max_side = input_ksize
        h, w, _ = r_img.shape

        smallest_side = min(w, h)
        largest_side = max(w, h)
        scale = min_side / smallest_side  # zoom factor
        if largest_side * scale > max_side:
            scale = max_side / largest_side  # zoom factor
        nw, nh = int(scale * w), int(scale * h)
        rgb_image_resized = cv2.resize(r_img, (nw, nh))  # opencv resize
        ir_image_resized  = cv2.resize(i_img, (nw, nh))  # opencv resize

        pad_w = 32 - nw % 32  # padding range
        pad_h = 32 - nh % 32  # padding range

        rgb_image_paded = np.zeros(shape=[nh + pad_h, nw + pad_w, 3], dtype=np.uint8)  # pad vis images
        ir_image_paded = np.zeros(shape=[nh + pad_h, nw + pad_w, 3], dtype=np.uint8)  # pad ir images
        rgb_image_paded[:nh, :nw, :] = rgb_image_resized  # pad vis images
        ir_image_paded[:nh, :nw, :] = ir_image_resized  # pad ir images

        if boxes is None:
            return rgb_image_paded, ir_image_paded
        else:
            boxes[:, [0, 2]] = boxes[:, [0, 2]] * scale
            boxes[:, [1, 3]] = boxes[:, [1, 3]] * scale
            return rgb_image_paded, ir_image_paded, boxes, scale



    def _has_only_empty_bbox(self,annot):
        return all(any(o <= 1 for o in obj['bbox'][2:]) for obj in annot)


    def _has_valid_annotation(self,annot):
        if len(annot) == 0:
            return False

        if self._has_only_empty_bbox(annot):
            return False

        return True
    def collate_fn(self,data):
        rgb_imgs_list,ir_imgs_list, boxes_list,classes_list,scale, id=zip(*data)
        assert len(rgb_imgs_list)==len(ir_imgs_list)==len(boxes_list)==len(classes_list)
        batch_size=len(boxes_list)
        rgb_pad_imgs_list=[]
        ir_pad_imgs_list=[]
        pad_boxes_list=[]
        pad_classes_list=[]

        rgb_h_list = [int(s.shape[1]) for s in rgb_imgs_list]
        ir_h_list = [int(s.shape[1]) for s in ir_imgs_list]
        rgb_w_list = [int(s.shape[2]) for s in rgb_imgs_list]
        ir_w_list = [int(s.shape[2]) for s in ir_imgs_list]
        rgb_max_h = np.array(rgb_h_list).max()
        ir_max_h = np.array(ir_h_list).max()
        rgb_max_w = np.array(rgb_w_list).max()
        ir_max_w = np.array(ir_w_list).max()
        for i in range(batch_size):
            rgb_img=rgb_imgs_list[i]
            ir_img=ir_imgs_list[i]
            rgb_pad_imgs_list.append(transforms.Normalize(self.mean, self.std,inplace=True)(torch.nn.functional.pad(rgb_img,(0,int(rgb_max_w-rgb_img.shape[2]),0,int(rgb_max_h-rgb_img.shape[1])),value=0.)))
            ir_pad_imgs_list.append(transforms.Normalize(self.mean, self.std,inplace=True)(torch.nn.functional.pad(ir_img,(0,int(ir_max_w-ir_img.shape[2]),0,int(ir_max_h-ir_img.shape[1])),value=0.)))


        max_num=0
        for i in range(batch_size):
            n=boxes_list[i].shape[0]
            if n>max_num:max_num=n
        for i in range(batch_size):
            pad_boxes_list.append(torch.nn.functional.pad(boxes_list[i],(0,0,0,max_num-boxes_list[i].shape[0]),value=-1))
            pad_classes_list.append(torch.nn.functional.pad(classes_list[i],(0,max_num-classes_list[i].shape[0]),value=-1))


        batch_boxes=torch.stack(pad_boxes_list)
        batch_classes=torch.stack(pad_classes_list)
        batch_rgb_imgs=torch.stack(rgb_pad_imgs_list)
        batch_ir_imgs=torch.stack(ir_pad_imgs_list)

        return batch_rgb_imgs,batch_ir_imgs,batch_boxes,batch_classes,scale, id
class Box:
    """ This is a generic bounding box representation.
    This class provides some base functionality to both annotations and detections.

    Attributes:
        class_label (string): class string label; Default **''**
        object_id (int): Object identifier for reid purposes; Default **0**
        x_top_left (Number): X pixel coordinate of the top left corner of the bounding box; Default **0.0**
        y_top_left (Number): Y pixel coordinate of the top left corner of the bounding box; Default **0.0**
        width (Number): Width of the bounding box in pixels; Default **0.0**
        height (Number): Height of the bounding box in pixels; Default **0.0**
    """
    def __init__(self):
        self.class_label = ''   # class string label
        self.object_id = 0      # object identifier
        self.x_top_left = 0.0   # x pixel coordinate top left of the box
        self.y_top_left = 0.0   # y pixel coordinate top left of the box
        self.width = 0.0        # width of the box in pixels
        self.height = 0.0       # height of the box in pixels
        self.confidence = 0.0       # height of the box in pixels

def evaluate_kaist(generator, model, threshold=0.4):
    """ Use the pycocotools to evaluate a COCO model on a dataset.

    Args
        oU NMSgenerator : The generator for g
        model     : The model to evaluate.
        threshold : The score threshold to use.
    """
    # start collecting results
    results = []
    image_ids = []
    detections_results = {}

    index = 0
    for data in tqdm(generator):
        rgb_img,ir_img, gt_boxes,gt_labels,scale,ids = data

        # run network
        scores,labels,boxes = model([rgb_img.cuda(), ir_img.cuda()])  # list of scores, labels and boxes
        batch_size = len(scores)  # batch size

        scores = [scores[i].detach().cpu().numpy() for i in range(batch_size)]  # tranform tensor into array
        labels = [labels[i].detach().cpu().numpy() for i in range(batch_size)]  # tranform tensor into array
        boxes = [boxes[i].detach().cpu().numpy() for i in range(batch_size)]  # tranform tensor into array
        # correct boxes for image scale
        boxes = [boxes[i]/scale[i] for i in range(batch_size)]  # restore dimension
        # change to (x, y, w, h) (MS COCO standard)
        for i in range(batch_size):
            boxes[i][:,2] =boxes[i][:,2]-boxes[i][:,0]
            boxes[i][:,3] =boxes[i][:,3]-boxes[i][:,1]

        # compute predicted labels and scores

        for box, score, label, id in zip(boxes, scores, labels, ids):
            # scores are sorted, so we can break
            # box [4,1] score float label
            detections =[]
            for j in range(score.shape[0]):
                # detections = []
                if score[j] < threshold:
                    break
            # for lamr calcualtion
                detection= Box()  # inital Box class
                detection.class_label = label[j]  # label
                detection.confidence = score[j]  # score
                detection.x_top_left = box[j][0]  # x_top_left
                detection.y_top_left = box[j][1]  # y_top_left
                detection.width = box[j][2]  # width of the box in pixels
                detection.height = box[j][3]  # height of the box in pixels

                detections.append(detection)

            # append detection for each positively labeled class for coco evaluation
                image_result = {
                    'image_id'    : id,
                    'category_id' : generator.dataset.id2category[label[j]],
                    'score'       : float(score[j]),
                    'bbox'        : box[j].tolist(),
                }

            # append detection to results
                results.append(image_result)
            # set a list
            detections_results[id] = detections
        # append image to list of processed images
            image_ids.append(id)

        index += 1

    if not len(results):
        return
    #
    # # write output
    json.dump(results, open('coco_bbox_results.json', 'w'), indent=4)  # save the detection results in a json
    # json.dump(image_ids, open('{}_processed_image_ids.json'.format(generator.set_name), 'w'), indent=4)

    # load results in COCO evaluation tool
    coco_true = generator.dataset.coco
    miss_rate, fppi = mr_fppi(detections_results, coco_true.imgToAnns, overlap_threshold=0.5)  # miss rate,  false positive per image
    mr = lamr(miss_rate, fppi)  # log average miss rate

        # evaluate coco
    coco_pred = coco_true.loadRes('coco_bbox_results.json')  # load this json
        # run COCO evaluation
    coco_eval = COCOeval(coco_true, coco_pred, 'bbox') # coco evaluation
    coco_eval.params.imgIds = image_ids  #
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    print("average MR: ", mr)

    return (mr, coco_eval.stats)

    # print("end")



if __name__ == "__main__":

    generator=KAIST_Generator(rgb_imgs_path="E:/dataset/coco/images/val_rgb",
                                  ir_imgs_path="E:/dataset/coco/images/val_ir",
                                  anno_path="E:/dataset//coco/annotations/val_2017.json",
                              resize_size=[512,640])

    generator = torch.utils.data.DataLoader(generator, batch_size=1, shuffle=False,
                                               collate_fn=generator.collate_fn,
                                               num_workers=0, worker_init_fn=np.random.seed(0))
    model=FCOSDetector(mode="inference", loss= 'v2', config = DefaultConfig_v1)

    model = torch.nn.DataParallel(model)
    model = model.cuda().eval()
    weight_path = "model_26.pth"

    model.load_state_dict(torch.load(weight_path, map_location=torch.device('cpu')))
    evaluate_kaist(generator, model, threshold=0.25)

    # This part is used for
    # list = os.listdir(weight_path)
    # This part is used for
    # for i in range(len(list)):
    #     weight_p = os.path.join(weight_path,"model_{}.pth".format(str(i+1)))
    #     epoch = str(i+1)
    #     model.load_state_dict(torch.load(weight_p,map_location=torch.device('cpu')))
    # # model.load_state_dict(torch.load("./checkpoint/2024_08_20_09_44_03_multispectral_pedestrian_detection_no_data_aug_loss_v2_fcos_v1/model_84.pth",map_location=torch.device('cpu')))
    # # model.load_state_dict(torch.load("./checkpoint/2024_08_17_22_12_16/model_8.pth",map_location=torch.device('cpu')))
    #     mr, cocoeval_results = evaluate_kaist(generator,model,threshold=0.25)
    #     with open("mr.txt", "a") as f:
    #         f.write("epoch " + epoch + ":"+ str(mr) + "\n")
    #
    #     with open("coco_eval.txt", "a") as f1:
    #         iStr = (' AP(IoU=0.50:0.95 all maxDets 100):%0.3f' % cocoeval_results[0] + '\n'
    #              'AP(IoU=0.50 all maxDets 10000):%0.3f ' % cocoeval_results[1] + '\n'
    #             'AP(IoU=0.75 all maxDets 10000):%0.3f ' % cocoeval_results[2] + '\n'
    #             'AP(IoU=0.50:0.95 small maxDets 10000):%0.3f ' % cocoeval_results[3] + '\n'
    #             'AP(IoU=0.50:0.95 medium maxDets 10000):%0.3f ' % cocoeval_results[4] + '\n'
    #             'AP(IoU=0.50:0.95 large maxDets 10000):%0.3f' %cocoeval_results[5] + '\n'
    #             'AR(IoU=0.50:0.95 all maxDets 1):%0.3f' % cocoeval_results[6] + '\n'
    #             'AR(IoU=0.50:0.95 all maxDets 10):%0.3f' % cocoeval_results[7] + '\n'
    #             'AR(IoU=0.50:0.95 all maxDets 10000):%0.3f' % cocoeval_results[8] + '\n'
    #             'AR(IoU=0.50:0.95 small maxDets 10000):%0.3f' %cocoeval_results[9] + '\n'
    #             'AR(IoU=0.50:0.95 medium maxDets 10000):%0.3f' % cocoeval_results[10] + '\n'
    #             'AR(IoU=0.50:0.95 large maxDets 10000):%0.3f' % cocoeval_results[11] + '\n')
    #         f1.write("epoch " + epoch + ":" + "\n")
    #         f1.write(iStr)

