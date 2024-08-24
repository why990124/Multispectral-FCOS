from model.fcos_multispectral import FCOSDetector
import torch
from dataset.COCO_dataset import COCODataset
from dataset.KAIST_dataset import KaistDataset
import math, time
from dataset.augment import Transforms
import os
import numpy as np
import random
import torch.backends.cudnn as cudnn
import argparse
from kaist_eval import KAIST_Generator, evaluate_kaist
import logging
from model.config_v1 import DefaultConfig_v1
from model.config_v2 import DefaultConfig_v2
from model.config import DefaultConfig
import torchviz
import graphviz



def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=150, help="number of epochs")
    parser.add_argument("--batch_size", type=int, default=1, help="size of each image batch")
    parser.add_argument("--n_cpu", type=int, default=0, help="number of cpu threads to use during batch generation")
    parser.add_argument("--n_gpu", type=str, default='0', help="index of gpu to use during training")
    parser.add_argument("--eval_step", type=int, default=1, help="steps between evaluation")
    parser.add_argument("--img_size", type=list, default=[512, 640], help="input image_size")
    parser.add_argument("--save_path", type=str, default="./checkpoint", help="folder to save weights")
    parser.add_argument("--save_weight_step", type=int, default=1, help="step of saving weights")
    parser.add_argument("--project_name", type=str,
                        default="multispectral_pedestrian_detection_no_data_aug_loss_v2_fcos_v1",
                        help="name of projects")
    parser.add_argument("--confidence_threshold", type=int, default=0.5,
                        help="confidence score of evalution and inference")
    parser.add_argument("--loss_version", type=str, default="v1", help="loss version")
    parser.add_argument("--fcos_version", type=str, default="v1", help="fcos version")
    opt = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = opt.n_gpu
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    np.random.seed(0)
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(0)
    transform = Transforms()
    train_dataset = KaistDataset(rgb_imgs_path="../FCOS-PyTorch-37.2AP-master/coco/train_rgb",
                                 ir_imgs_path="../FCOS-PyTorch-37.2AP-master/coco/train_ir",
                                 anno_path='../FCOS-PyTorch-37.2AP-master/coco/annotations/instances_train2017.json',
                                 resize_size=opt.img_size,
                                 transform=transform)

    val_dataset = KAIST_Generator(rgb_imgs_path="../FCOS-PyTorch-37.2AP-master/coco/val_rgb",
                                  ir_imgs_path="../FCOS-PyTorch-37.2AP-master/coco/val_ir",
                                  anno_path="../FCOS-PyTorch-37.2AP-master/coco/annotations/instances_val2017.json",
                                  resize_size=opt.img_size)

    # loss version:
    # v1: two types of detection head are separated, two groups of losses
    # v1: two types of detection head are fused, only one group of losses

    # fcos version:
    # fcos v1: config = DefaultConfig_v1
    # fcos v2: config = DefaultConfig_v2
    if opt.fcos_version == "v1":
        model_config = DefaultConfig_v1
    elif opt.fcos_version == "v2":
        model_config = DefaultConfig_v2
    elif opt.fcos_version == "test":
        model_config = DefaultConfig

    model = FCOSDetector(mode="training", loss=opt.loss_version, config=model_config).cuda()
    model_eval = FCOSDetector(mode="inference", loss=opt.loss_version, config=model_config).cuda()
    model = torch.nn.DataParallel(model)
    model_eval = torch.nn.DataParallel(model_eval)
    BATCH_SIZE = opt.batch_size
    EPOCHS = opt.epochs
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                                               collate_fn=train_dataset.collate_fn,
                                               num_workers=opt.n_cpu, worker_init_fn=np.random.seed(0))

    torch.utils.data.DataLoader(val_dataset, batch_size=4, shuffle=True,
                                collate_fn=val_dataset.collate_fn,
                                num_workers=8, worker_init_fn=np.random.seed(0))

    steps_per_epoch = len(train_dataset) // BATCH_SIZE
    TOTAL_STEPS = steps_per_epoch * EPOCHS
    WARMUP_STEPS = 0
    WARMUP_FACTOR = 1.0 / 3.0
    GLOBAL_STEPS = 0
    LR_INIT = 0.01
    optimizer = torch.optim.SGD(model.parameters(), lr=LR_INIT, momentum=0.9, weight_decay=0.0001)
    lr_schedule = [10, 100]  # epcch number


    def lr_func(step):
        lr = LR_INIT
        if step < WARMUP_STEPS:
            alpha = float(step) / WARMUP_STEPS
            warmup_factor = WARMUP_FACTOR * (1.0 - alpha) + alpha
            lr = lr * warmup_factor
        else:
            for i in range(len(lr_schedule)):
                if step < lr_schedule[i]:
                    break
                lr *= 0.1
        return float(lr)


    model.train()
    # model.load_state_dict(torch.load("./model_45.pth", map_location=torch.device('cpu')))
    now = int(time.time())
    # 转换为其他日期格式,如:"%Y-%m-%d %H:%M:%S"
    timeArray = time.localtime(now)
    otherStyleTime = time.strftime("%Y_%m_%d_%H_%M_%S", timeArray)

    checkpoint_save_path_name = os.path.join(opt.save_path, otherStyleTime + "_" + opt.project_name)
    if not os.path.exists(checkpoint_save_path_name):
        os.makedirs(checkpoint_save_path_name)
    logger = get_logger('logs/{}.log'.format(otherStyleTime + "_" + opt.project_name))
    # logger_precision = get_logger('logs/{}.log'.format("precision" + otherStyleTime + "_" + opt.project_name))

    logger.info('start training!')
    for epoch in range(EPOCHS):

        for epoch_step, data in enumerate(train_loader):

            batch_rgb_imgs, batch_ir_imgs, batch_boxes, batch_classes = data
            batch_rgb_imgs = batch_rgb_imgs.cuda()
            batch_ir_imgs = batch_ir_imgs.cuda()
            batch_boxes = batch_boxes.cuda()
            batch_classes = batch_classes.cuda()

            lr = lr_func(epoch + 1)
            for param in optimizer.param_groups:
                param['lr'] = lr

            start_time = time.time()

            optimizer.zero_grad()
            output = model([batch_rgb_imgs, batch_ir_imgs, batch_boxes, batch_classes])
            # output = model(losses)
            graph = torchviz.make_dot(output, params=dict(model.named_parameters()))
            graph.render("CNN_graph_v1")  # 保存计算图为 PDF 文件
            break

        break

    #         loss = losses[-1]
    #         loss.mean().backward()
    #         torch.nn.utils.clip_grad_norm_(model.parameters(), 3)
    #         optimizer.step()
    #
    #         end_time = time.time()
    #         cost_time = int((end_time - start_time) * 1000)
    #         remaining_time_second = cost_time / 1000 * (TOTAL_STEPS - epoch * steps_per_epoch - epoch_step)
    #
    #         MinutesGet, SecondsGet = divmod(remaining_time_second, 60)
    #
    #         HoursGet, MinutesGet = divmod(MinutesGet, 60)
    #
    #         if opt.loss_version == "v1":
    #             logger.info(
    #                 "global_steps:%d epoch:%d steps:%d/%d cls_loss_vis:%.4f cnt_loss_vis:%.4f reg_loss_vis:%.4f cls_loss_ir:%.4f"
    #                 " cnt_loss_ir:%.4f reg_loss_ir:%.4f cost_time:%dms lr=%.4e total_loss:%.4f remaining_time:%d:%d:%d" % \
    #                 (GLOBAL_STEPS, epoch + 1, epoch_step + 1, steps_per_epoch, losses[0].mean(), losses[1].mean(),
    #                  losses[2].mean(), losses[3].mean(), losses[4].mean(), losses[5].mean(), cost_time, lr, loss.mean(),
    #                  HoursGet, MinutesGet, SecondsGet))
    #         elif opt.loss_version == "v2":
    #             logger.info(
    #                 "global_steps:%d epoch:%d steps:%d/%d cls_loss:%.4f cnt_loss:%.4f reg_loss:%.4f cost_time:%dms lr=%.4e total_loss:%.4f remaining_time:%d:%d:%d" % \
    #                 (GLOBAL_STEPS, epoch + 1, epoch_step + 1, steps_per_epoch, losses[0].mean(), losses[1].mean(),
    #                  losses[2].mean(), cost_time, lr, loss.mean(), HoursGet, MinutesGet, SecondsGet))
    #         # print(
    #         #     "global_steps:%d epoch:%d steps:%d/%d cls_loss:%.4f cnt_loss:%.4f reg_loss:%.4f cost_time:%dms lr=%.4e total_loss:%.4f" % \
    #         #     (GLOBAL_STEPS, epoch + 1, epoch_step + 1, steps_per_epoch, losses[0].mean(), losses[1].mean(),
    #         #      losses[2].mean(), cost_time, lr, loss.mean()))
    #
    #         GLOBAL_STEPS += 1
    #
    #     if (epoch + 1) % opt.save_weight_step == 0:
    #         torch.save(model.state_dict(), os.path.join(checkpoint_save_path_name, "model_{}.pth".format(epoch + 1)))
    #         logger.info(
    #             "Weight of epoch %d" % (epoch + 1) + " has been saved in " + os.path.join(checkpoint_save_path_name,
    #                                                                                       "model_{}.pth".format(
    #                                                                                           epoch + 1)) + '\n')
    #
    #     if (epoch + 1) % opt.eval_step == 0:
    #
    #         model_eval = model_eval.cuda().eval()
    #         model_eval.load_state_dict(
    #             model.state_dict())
    #
    #         results = evaluate_kaist(val_dataset, model_eval, threshold=opt.confidence_threshold)
    #
    #         if results is None:
    #
    #             logger.info("------------------------------------------------------------------" + '\n' +
    #                         "There is no detected objects with confidence larger than threshold" + '\n' +
    #                         "------------------------------------------------------------------")
    #         else:
    #             mr, cocoeval_results = results
    #             iStr = (' AP(IoU=0.50:0.95 all maxDets 100):<18' % cocoeval_results[0] + '\n'
    #                                                                                      'AP(IoU=0.50 all maxDets 10000):<18 ' %
    #                     cocoeval_results[1] + '\n'
    #                                           'AP(IoU=0.75 all maxDets 10000):<18 ' % cocoeval_results[2] + '\n'
    #                                                                                                         'AP(IoU=0.50:0.95 small maxDets 10000):<18 ' %
    #                     cocoeval_results[3] + '\n'
    #                                           'AP(IoU=0.50:0.95 medium maxDets 10000):<18 ' % cocoeval_results[4] + '\n'
    #                                                                                                                 'AP(IoU=0.50:0.95 large maxDets 10000):<18' %
    #                     cocoeval_results[5] + '\n'
    #                                           'AR(IoU=0.50:0.95 all maxDets 1):<18' % cocoeval_results[6] + '\n'
    #                                                                                                         'AR(IoU=0.50:0.95 all maxDets 10):<18' %
    #                     cocoeval_results[7] + '\n'
    #                                           'AR(IoU=0.50:0.95 all maxDets 10000):<18' % cocoeval_results[8] + '\n'
    #                                                                                                             'AR(IoU=0.50:0.95 small maxDets 10000):<18' %
    #                     cocoeval_results[9] + '\n'
    #                                           'AR(IoU=0.50:0.95 medium maxDets 10000):<18' % cocoeval_results[10] + '\n'
    #                                                                                                                 'AR(IoU=0.50:0.95 large maxDets 10000):<18') % \
    #                    cocoeval_results[11] + '\n'
    #
    #             logger.info('\n' + "------------------------------------------------------------------" + '\n' +
    #                         "epoch:%d" % (epoch) + "\t" + "Average MR: %4f" % (mr) + '\n')
    #
    #             logger.info(iStr)
    #             logger.info("------------------------------------------------------------------" + '\n')
    #
    # logger.info('finish training!')









