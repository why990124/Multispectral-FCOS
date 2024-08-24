from torchvision.datasets import CocoDetection
import torch
import numpy as np
from torchvision import transforms
import cv2
from PIL import  Image
import random
from torchvision.datasets.coco import VisionDataset
from typing import Any, Callable, List, Optional, Tuple
import os

def flip(rgb_img, ir_img, boxes):
    rgb_img = rgb_img.transpose(Image.FLIP_LEFT_RIGHT)
    ir_img = ir_img.transpose(Image.FLIP_LEFT_RIGHT) # add ir_img
    w = rgb_img.width
    if boxes.shape[0] != 0:
        xmin = w - boxes[:,2]
        xmax = w - boxes[:,0]
        boxes[:, 2] = xmax
        boxes[:, 0] = xmin
    return rgb_img, ir_img, boxes # add ir_img

class KaistDetection(VisionDataset):
    """`MS Coco Detection <https://cocodataset.org/#detection-2016>`_ Dataset.

    It requires the `COCO API to be installed <https://github.com/pdollar/coco/tree/master/PythonAPI>`_.

    Args:
        root (string): Root directory where images are downloaded to.
        annFile (string): Path to json annotation file.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.PILToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        transforms (callable, optional): A function/transform that takes input sample and its target as entry
            and returns a transformed version.
    """

    def __init__(
        self,
        root: str,
        ir_root: str,
        annFile: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        transforms: Optional[Callable] = None,
    ) -> None:
        super().__init__(root, transforms, transform, target_transform)
        from pycocotools.coco import COCO
        self.ir_root = ir_root
        self.coco = COCO(annFile)
        self.ids = list(sorted(self.coco.imgs.keys()))

    def _load_image(self, id: int) -> Image.Image:
        path = self.coco.loadImgs(id)[0]["file_name"]
        return Image.open(os.path.join(self.root, path)).convert("RGB"), Image.open(os.path.join(self.ir_root, path)).convert("RGB")

    def _load_target(self, id: int) -> List[Any]:
        return self.coco.loadAnns(self.coco.getAnnIds(id))

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        id = self.ids[index]
        rgb_image, ir_image = self._load_image(id)
        target = self._load_target(id)

        # if self.transforms is not None:
        #     image, target = self.transforms(image, target)

        return rgb_image, ir_image, target, id

    def __len__(self) -> int:
        return len(self.ids)

class KaistDataset(KaistDetection):
    CLASSES_NAME = (
    '__back_ground__', 'person')
    def __init__(self,rgb_imgs_path, ir_imgs_path, anno_path, resize_size=[800,1333], is_train = True, transform=None):
        super().__init__(rgb_imgs_path, ir_imgs_path, anno_path)
        # super.__init__ content :self.ir_root = ir_root;  self.coco = COCO(annFile); self.ids = list(sorted(self.coco.imgs.keys()))

        print("INFO====>check annos, filtering invalid data......")
        ids=[]
        for id in self.ids:  # coco.imgs.keys img ids in annotations eg, 20190000001
            ann_id=self.coco.getAnnIds(imgIds=id,iscrowd=None)  # coco's function, return annotation ids in this img
            ann=self.coco.loadAnns(ann_id)  # according to ann_id, load annotations
            if self._has_valid_annotation(ann): # if annotation and box exist, true
                ids.append(id)
        self.ids=ids # img ids
        self.category2id = {v: i + 1 for i, v in enumerate(self.coco.getCatIds())}  # return catagory id {"class": "class id" }
        self.id2category = {v: k for k, v in self.category2id.items()} # return id {"class id": "class" }

        self.transform=transform  # transform
        self.resize_size=resize_size  # resize images

        self.mean=[0.40789654, 0.44719302, 0.47026115]
        self.std=[0.28863828, 0.27408164, 0.27809835]
        self.train = is_train # train true

    def __getitem__(self,index):

        rgb_img, ir_img, ann, _ = super().__getitem__(index)  # get vis image, ir images, annotations and image ids

        ann = [o for o in ann if o['iscrowd'] == 0]  # no 'iscrowd'
        boxes = [o['bbox'] for o in ann]  # add boxes
        boxes=np.array(boxes,dtype=np.float32)  # numpy array float32
        #xywh-->xyxy
        boxes[...,2:]=boxes[...,2:]+boxes[...,:2]  # from xywh to xyxy
        if self.train:
            if random.random() < 0.5 :
                rgb_img, ir_img, boxes = flip(rgb_img, ir_img, boxes)  # flip images

        rgb_img = np.array(rgb_img)
        ir_img = np.array(ir_img) # add ir_img

        rgb_img, ir_img, boxes=self.preprocess_img_boxes(rgb_img,ir_img, boxes,self.resize_size) # preprocess vis images, ir images simultaneously



        classes = [o['category_id'] for o in ann]  #  class id
        classes = [self.category2id[c] for c in classes]   #  class id



        rgb_img = transforms.ToTensor()(rgb_img)  # transform array to tensor
        ir_img = transforms.ToTensor()(ir_img)  # transform array to tensor
        # img= transforms.Normalize(self.mean, self.std,inplace=True)(img)  # no data augment
        boxes=torch.from_numpy(boxes)  # transform array to tensor
        classes=torch.LongTensor(classes)   # transform tensor to Long tensor

        return rgb_img,ir_img, boxes, classes

    def preprocess_img_boxes(self,rgb_img, ir_img, boxes,input_ksize):
        '''
        resize image and bboxes
        Returns
        image_paded: input_ksize
        bboxes: [None,4]
        '''
        min_side, max_side    = input_ksize
        h,  w, _  = rgb_img.shape

        smallest_side = min(w,h)
        largest_side = max(w,h)
        scale=min_side/smallest_side  # zoom factor
        if largest_side*scale>max_side:
            scale=max_side/largest_side  # zoom factor
        nw, nh  = int(scale * w), int(scale * h)
        rgb_image_resized = cv2.resize(rgb_img, (nw, nh))  # opencv resize
        ir_image_resized = cv2.resize(ir_img, (nw, nh))  # opencv resize

        pad_w=32-nw%32  # padding range
        pad_h=32-nh%32  # padding range

        rgb_image_paded = np.zeros(shape=[nh+pad_h, nw+pad_w, 3],dtype=np.uint8)  # pad vis images
        ir_image_paded = np.zeros(shape=[nh+pad_h, nw+pad_w, 3],dtype=np.uint8) # pad ir images
        rgb_image_paded[:nh, :nw, :] = rgb_image_resized  # pad vis images
        ir_image_paded [:nh, :nw, :] = ir_image_resized   # pad ir images

        if boxes is None:
            return rgb_image_paded, ir_image_paded # add ir_img
        else:
            boxes[:, [0, 2]] = boxes[:, [0, 2]] * scale
            boxes[:, [1, 3]] = boxes[:, [1, 3]] * scale
            return rgb_image_paded, ir_image_paded, boxes # add ir_img



    def _has_only_empty_bbox(self,annot):
        return all(any(o <= 1 for o in obj['bbox'][2:]) for obj in annot)


    def _has_valid_annotation(self,annot):
        if len(annot) == 0:
            return False

        if self._has_only_empty_bbox(annot):
            return False

        return True

    def collate_fn(self,data):
        rgb_imgs_list,ir_imgs_list, boxes_list,classes_list=zip(*data)  #
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

        return batch_rgb_imgs,batch_ir_imgs,batch_boxes,batch_classes







if __name__=="__main__":

    dataset=KaistDataset("D:\\pycharmproject\\FCOS-PyTorch-37.2AP-master\\coco\\train_rgb", "D:\\pycharmproject\\FCOS-PyTorch-37.2AP-master\\coco\\train_ir", "D:\\pycharmproject\\FCOS-PyTorch-37.2AP-master\\coco\\annotations\\instances_train2017.json")
    # img,boxes,classes=dataset[0]
    # print(boxes,classes,"\n",img.shape,boxes.shape,classes.shape,boxes.dtype,classes.dtype,img.dtype)
    # cv2.imwrite("./123.jpg",img)
    rgb_img,ir_image, boxes,classes=dataset.collate_fn([dataset[0],dataset[1],dataset[2]])
    print(boxes,"\n",classes,"\n",rgb_img.shape,"\n", ir_image.shape,"\n",boxes.shape,"\n",classes.shape,"\n",boxes.dtype,"\n",classes.dtype,"\n",rgb_img.dtype,"\n",ir_image.dtype)
