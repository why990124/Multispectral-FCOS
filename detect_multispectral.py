import cv2
from model.fcos_multispectral import FCOSDetector
import torch
from torchvision import transforms
import numpy as np
from dataset.VOC_dataset import VOCDataset
import time
import matplotlib.patches as patches
import  matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator
from model.config_v1 import DefaultConfig_v1
from model.config_v2 import DefaultConfig_v2
import os
import argparse

CLASSES_NAME = ['__back_ground__',"person"]

def preprocess_img(image,input_ksize):
    '''
    resize image and bboxes 
    Returns
    image_paded: input_ksize  
    bboxes: [None,4]
    '''
    min_side, max_side    = input_ksize
    h,  w, _  = image.shape

    smallest_side = min(w,h)
    largest_side=max(w,h)
    scale=min_side/smallest_side
    if largest_side*scale>max_side:
        scale=max_side/largest_side
    nw, nh  = int(scale * w), int(scale * h)
    image_resized = cv2.resize(image, (nw, nh))

    pad_w=32-nw%32
    pad_h=32-nh%32

    image_paded = np.zeros(shape=[nh+pad_h, nw+pad_w, 3],dtype=np.uint8)
    image_paded[:nh, :nw, :] = image_resized
    return image_resized
    
def convertSyncBNtoBN(module):
    module_output = module
    if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
        module_output = torch.nn.BatchNorm2d(module.num_features,
                                               module.eps, module.momentum,
                                               module.affine,
                                               module.track_running_stats)
        if module.affine:
            module_output.weight.data = module.weight.data.clone().detach()
            module_output.bias.data = module.bias.data.clone().detach()
        module_output.running_mean = module.running_mean
        module_output.running_var = module.running_var
    for name, child in module.named_children():
        module_output.add_module(name,convertSyncBNtoBN(child))
    del module
    return module_output
if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--confidence_threshold", type=int, default=0.25, help="confidence score of evalution and inference")
    parser.add_argument("--batch_size", type=int, default=1, help="size of each image batch")
    parser.add_argument("--n_cpu", type=int, default=0, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_vis", type=str, default='test_img/test_img_vis', help="original visual-optical images")
    parser.add_argument("--img_ir", type=str, default='test_img/test_img_ir', help="original infrared images")
    parser.add_argument("--save_results_vis", type=str, default="detect_results/vis", help="save path of visual-optical detection results")
    parser.add_argument("--save_results_ir", type=str, default="detect_results/ir", help="save path of infrared detection results")
    parser.add_argument("--weight_path", type=str, default="model_26.pth", help="save path of infrared detection results")
    parser.add_argument("--save_weight_step", type=int, default=1, help="step of saving weights")
    # parser.add_argument("--config", type=str, default="multispectral_pedestrian_detection_no_data_aug_loss_v2_fcos_v1", help="name of projects")
    parser.add_argument("--image_size", type=int, default=[512,640], help="confidence score of evalution and inference")
    parser.add_argument("--loss_version", type=str, default="v2", help="loss version")
    parser.add_argument("--fcos_version", type=str, default="v1", help="fcos version")
    opt = parser.parse_args()

    if not os.path.exists(opt.save_results_vis):
        os.makedirs(opt.save_results_vis)
    if not os.path.exists(opt.save_results_ir):
        os.makedirs(opt.save_results_ir)

    colors = [(128, 188,255), (199, 34, 40)]  # box colors in terms of classes

    # threshold = 0.25  # confidence threshold
    # input_size = [512,640]  # image size
    model = FCOSDetector(mode="inference", config=DefaultConfig_v1)  # inital FCOSDetector
    model = torch.nn.DataParallel(model)  # use multiple GPU
    model = model.cuda().eval()  # set evaluation pattern
    # model.load_state_dict(torch.load("./model_45.pth", map_location=torch.device('cpu')))
    model.load_state_dict(torch.load(opt.weight_path, map_location=torch.device('cpu')))

    print("===>success loading model")

    #image paths
    root_rgb = opt.img_vis
    root_ir = opt.img_ir
    # root_rgb="E:/dataset/images_divide/val_rgb"
    # root_ir = "E:/dataset/images_divide/val_ir"
    input_size =opt.image_size
    # image list
    names=os.listdir(root_rgb)
    for name in names:
        # read image
        img_bgr_rgb=cv2.imread(os.path.join(root_rgb, name))
        img_bgr_ir=cv2.imread(os.path.join(root_ir, name))
        img_rgb_rgb = cv2.cvtColor(img_bgr_rgb.copy(), cv2.COLOR_BGR2RGB)
        img_rgb_ir = cv2.cvtColor(img_bgr_ir.copy(), cv2.COLOR_BGR2RGB)

        # padd images
        img_rgb=preprocess_img(img_bgr_rgb,input_size)
        img_ir=preprocess_img(img_bgr_ir,input_size)


        # transform numpy to tensor
        img1_rgb=transforms.ToTensor()(img_rgb)
        img1_ir=transforms.ToTensor()(img_ir)

        # normalization
        img1_rgb= transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225],inplace=True)(img1_rgb)
        img1_ir= transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225],inplace=True)(img1_ir)
        img1_rgb=img1_rgb
        img1_ir=img1_ir

        # zoom scale
        min_side, max_side = input_size
        h, w, _ = img_bgr_rgb.shape

        smallest_side = min(w, h)
        largest_side = max(w, h)
        scale = min_side / smallest_side
        if largest_side * scale > max_side:
            scale = max_side / largest_side

        # start time
        start_t=time.time()

        # detect
        with torch.no_grad():
            out=model([img1_rgb.unsqueeze_(dim=0),img1_ir.unsqueeze_(dim=0)])  # individual image detection

        # end time
        end_t=time.time()
        # time cost
        cost_t=1000*(end_t-start_t)
        print("===>success processing img, cost time %.2f ms"%cost_t)
        # scores, labels, boxes
        scores,labels,boxes=out

        boxes = boxes[0].cpu().numpy()  # transfor tensor into numpy
        labels = labels[0].cpu().numpy()  # transfor tensor into numpy
        scores = scores[0].cpu().numpy()  # transfor tensor into numpy
        boxes/= scale  # restore boxes

        for i,box in enumerate(boxes):
            if scores[i] > opt.confidence_threshold:
                pt1 = (int(box[0]), int(box[1]))
                pt2 = (int(box[2]), int(box[3]))
                b_color = colors[int(labels[i]) - 1]
                # draw box
                img_rgb= cv2.rectangle(img_rgb, pt1, pt2, b_color,1)
                img_ir= cv2.rectangle(img_ir, pt1, pt2, b_color,1)
                # write label and score
                cv2.putText(img_rgb, "%s %.3f" % (CLASSES_NAME[int(labels[i])], scores[i]), (int(box[0]), int(box[1])-8),
                            cv2.FONT_HERSHEY_COMPLEX, 0.5, b_color, 1, cv2.LINE_AA)
                cv2.putText(img_ir, "%s %.3f" % (CLASSES_NAME[int(labels[i])], scores[i]),
                            (int(box[0]), int(box[1]) - 8),
                            cv2.FONT_HERSHEY_COMPLEX, 0.5, b_color, 1, cv2.LINE_AA)

        cv2.imwrite(os.path.join(opt.save_results_vis, name),img_rgb)
        cv2.imwrite(os.path.join(opt.save_results_ir, name),img_ir)






