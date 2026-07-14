from __future__ import print_function

import os
import copy
import numpy as np
import sys
import argparse
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from modules.HDC_utils import set_model, set_uq_model
from modules.ioueval import *
import torch.backends.cudnn as cudnn
from postproc.KNN import KNN
from common.avgmeter import *

from torchhd import functional
from torchhd import embeddings

VAL_CNT = 10

class BasicHD():
    def __init__(self, ARCH, DATA, datadir, logdir, modeldir,
                logger):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # tensorboard
        self.logger = logger

        self.ARCH = ARCH
        self.DATA = DATA
        self.datadir = datadir
        self.logdir = logdir
        self.modeldir = modeldir
        self.epochs = 20

        from dataset.kitti.parser import Parser
        self.parser = Parser(root=self.datadir,
                                        train_sequences=self.DATA["split"]["train"],
                                        valid_sequences=self.DATA["split"]["valid"],
                                        test_sequences=None,
                                        labels=self.DATA["labels"],
                                        color_map=self.DATA["color_map"],
                                        learning_map=self.DATA["learning_map"],
                                        learning_map_inv=self.DATA["learning_map_inv"],
                                        sensor=self.ARCH["dataset"]["sensor"],
                                        max_points=self.ARCH["dataset"]["max_points"],
                                        batch_size=self.ARCH["train"]["batch_size"],
                                        workers=self.ARCH["train"]["workers"],
                                        gt=True,
                                        shuffle_train=False)
        self.num_classes = self.parser.get_n_classes() 
        epsilon_w = self.ARCH["train"]["epsilon_w"]
        content = torch.zeros(self.parser.get_n_classes(), dtype=torch.float)
        for cl, freq in DATA["content"].items():
            x_cl = self.parser.to_xentropy(cl)  # map actual class to xentropy class
            content[x_cl] += freq
        self.loss_w = 1 / (content + epsilon_w)  # get weights
        for x_cl, w in enumerate(self.loss_w):  # ignore the ones necessary to ignore
            if DATA["learning_ignore"][x_cl]:
                # don't weigh
                self.loss_w[x_cl] = 0
        print("Loss weights from content: ", self.loss_w.data)

        # build model and criterion
        # self.model = model
        # concatenate the encoder and the head
        self.model = set_model(ARCH, modeldir, 'rp', 0, 0, self.num_classes, self.device)
        print(self.parser.get_n_classes())
        self.post = None
        if self.ARCH["post"]["KNN"]["use"]:
            self.post = KNN(self.ARCH["post"]["KNN"]["params"],
                            self.parser.get_n_classes())
        print(self.parser.get_n_classes())

        # GPU?
        self.gpu = False
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Infering in device: ", self.device)
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            cudnn.benchmark = True
            cudnn.fastest = True
            self.gpu = True
            self.model.cuda()

    def start(self):
        print("Starting training with the HDC online learning:")
        self.model.eval()
        self.ignore_class = []
        for i, w in enumerate(self.loss_w):
            if w < 1e-10:
                self.ignore_class.append(i)
                print("Ignoring class ", i, " in IoU evaluation")
        self.evaluator = iouEval(self.parser.get_n_classes(),
                                 self.device, self.ignore_class)
        for e in range(1, 2):
            time1 = time.time()
            self.train(self.parser.get_train_set(), self.model, self.logger)
            time2 = time.time()
            print('train epoch {}, total time {:.2f}'.format(e, time2 - time1))
            # acc = self.validate(self.parser.get_valid_set(), self.model, self.evaluator)
            # print('Stream final acc: {}'.format(acc))
        for epoch in range(1, self.epochs + 1):
            # train for one epoch
            time1 = time.time()
            self.retrain(self.parser.get_train_set(), self.model, epoch, self.logger)

            time2 = time.time()
            print('retrain epoch {}, total time {:.2f}'.format(epoch, time2 - time1))

            # final validation
            acc = self.validate(self.parser.get_valid_set(), self.model, self.evaluator)
            print('Stream final acc: {}'.format(acc))

    def train(self, train_loader, model, logger):  # task_list
        """Training on single-pass of data"""
        # Set validation frequency
        batchs_per_class = np.floor(len(train_loader) / self.num_classes).astype('int')
        if self.gpu:
            torch.cuda.empty_cache()
        with torch.no_grad():
            # for idx, (images, labels) in enumerate(train_loader):
            idx = 0  # batch index
            cur_class = -1
            self.mask = None
            train_time = []
            self.is_wrong_list = [None] * len(train_loader)  # store the wrong classification for each batch
            for i, (proj_in, proj_mask, proj_labels, unproj_labels, path_seq, path_name, p_x, p_y, proj_range, unproj_range, _, _, _, _, npoints) in enumerate(tqdm(train_loader, desc="Training")):
                # print(labels.detach().cpu().tolist())
                # print(images.shape, labels.shape)
                # if i > 10: # for debug
                #     break
                # proj_range = proj_range[0, :npoints]
                # unproj_range = unproj_range[0, :npoints]
                path_seq = path_seq[0]
                path_name = path_name[0]

                if self.gpu:
                    proj_in = proj_in.cuda()
                    proj_mask = proj_mask.cuda()
                    # if self.post:
                    #     proj_range = proj_range.cuda()
                    #     unproj_range = unproj_range.cuda()

                # samples_hv = self.model.encode(proj_in, self.mask) # (bsz*size, hd_dim)
                start = time.time()
                samples_hv, indices, _ = self.model.encode(proj_in, self.mask, PERCENTAGE=0.05)
                samples_hv = samples_hv.to(model.classify_weights.dtype)

                #proj_labels shape: torch.Size([1, 64, 512])
                proj_labels = proj_labels.view(-1)  # shape: (btsz*64*512, 1)
                proj_labels = proj_labels.to(self.device)
                proj_labels = proj_labels[indices]
                
                model.classify_weights.index_add_(0, proj_labels, samples_hv)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                res = time.time() - start
                train_time.append(res)
                start = time.time()

                predictions =self.model.get_predictions(samples_hv)
                # print("predictions: ", predictions) #torch.Size([32768, 20])
                argmax = predictions.argmax(dim=1) # (bsz*size, 1)
                # self.is_wrong_list[i] = proj_labels != argmax

                is_wrong = proj_labels != argmax
                proj_labels = proj_labels[is_wrong]
                argmax = argmax[is_wrong]
                samples_hv = samples_hv[is_wrong]
                samples_hv = samples_hv.to(model.classify_weights.dtype)

                # # Pick the loss by the wrong
                # self.is_wrong_list[i] = proj_labels != argmax
                # loss = nn.CrossEntropyLoss(weight=self.loss_w.to(self.device))(predictions, proj_labels)
                # self.is_wrong_list[i] *= loss

                # Pick the loss by the Wy^X - WyX
                # Compute dot products
                # true_scores = torch.sum(model.classify_weights[proj_labels] * samples_hv, dim=1)      # shape: [wrong_size]
                # wrong_scores = torch.sum(model.classify_weights[argmax] * samples_hv, dim=1) # shape: [wrong_size]
                true_scores = predictions[is_wrong, proj_labels]  # shape: [wrong_size]
                wrong_scores = predictions[is_wrong, argmax]  # shape: [wrong_size]
                # # losses = wrong_scores - true_scores  # shape: [wrong_size]
                losses = wrong_scores - true_scores  # shape: [wrong_size]
                # predictions 100:C
                # is_wrong 100:1 -> 
                # predictions[is_wrong]  50:C
                # proj_labels[is_wrong] 50:1

                # losses = true_scores - wrong_scores          # shape: [wrong_size]
                # if losses.sum().item() < 0:
                # print("Warning: negative losses detected, this is not expected")
                # print("proj_labels: ", proj_labels)
                # print("argmax: ", argmax)
                # print("samples_hv: ", samples_hv)
                # print("true_scores: ", true_scores)
                # print("wrong_scores: ", wrong_scores)
                # print("losses: ", losses)
                # print("Check the is_wrong shape", is_wrong.shape)
                # print("Check the losses shape", losses.shape)
                # print("Check the self.is_wrong_list[i] shape", self.is_wrong_list[i].shape)
                # assert self.is_wrong_list[i].shape == is_wrong.shape
                # Initialize if needed — make sure it's a FloatTensor
                # if self.is_wrong_list[i] is None or self.is_wrong_list[i].shape != is_wrong.shape:
                #     self.is_wrong_list[i] = torch.zeros_like(is_wrong, dtype=losses.dtype)
                # self.is_wrong_list[i][is_wrong] = losses
                # print("is_wrong shape: ", is_wrong.shape)
                self.is_wrong_list[i] = is_wrong
                # print(losses.min(), losses.max())


            model.classify.weight[:] = F.normalize(model.classify_weights)
            print("sum of is_wrong_list: ", sum([x.sum().item() for x in self.is_wrong_list if x is not None]))
            print("Mean HDC training time:{}\t std:{}".format(np.mean(train_time), np.std(train_time)))
            # print("Finish one batch, update classify weights")
    
    def retrain(self, train_loader, model, epoch, logger):  # task_list
        """Training of one epoch on single-pass of data"""
        # Set validation frequency
        batchs_per_class = np.floor(len(train_loader) / self.num_classes).astype('int')
        if self.gpu:
            torch.cuda.empty_cache()
        with torch.no_grad():
            # for idx, (images, labels) in enumerate(train_loader):
            idx = 0  # batch index
            cur_class = -1
            total_miss = 0
            retrain_time = []
            for i, (proj_in, proj_mask, proj_labels, unproj_labels, path_seq, path_name, p_x, p_y, proj_range, unproj_range, _, _, _, _, npoints) in enumerate(tqdm(train_loader, desc="Retraining")):
                # print(labels.detach().cpu().tolist())
                # print(images.shape, labels.shape)
                # if i > 10: # for debug
                #     break
                # proj_range = proj_range[0, :npoints]
                # unproj_range = unproj_range[0, :npoints]
                path_seq = path_seq[0]
                path_name = path_name[0]

                if self.gpu:
                    proj_in = proj_in.cuda()
                    proj_mask = proj_mask.cuda()
                    # if self.post:
                    #     proj_range = proj_range.cuda()
                    #     unproj_range = unproj_range.cuda()
                start = time.time()
                model.classify.weight[:] = F.normalize(model.classify_weights)
                # print("Number of wrongs:", self.is_wrong_list[i].sum().item())
                predictions, samples_hv, indices, self.is_wrong_list[i] = model(proj_in, self.mask, 0.05, self.is_wrong_list[i])
                argmax = predictions.argmax(dim=1) # (bsz*size, 1)
                # #proj_labels shape: torch.Size([1, 64, 512])
                proj_labels = proj_labels.view(-1)  # shape: (btsz*64*512, 1) 
                proj_labels = proj_labels.to(self.device)
                proj_labels = proj_labels[indices]  # map to the sampled hypervectors

                is_wrong = proj_labels != argmax
                # self.is_wrong_list[i][indices[is_wrong]] = True
                
                if is_wrong.sum().item() == 0:
                    continue

                # Check wrong classification number here and update the classify weights
                total_miss += is_wrong.sum().item()
                proj_labels = proj_labels[is_wrong]
                argmax = argmax[is_wrong]
                samples_hv = samples_hv[is_wrong]
                samples_hv = samples_hv.to(model.classify_weights.dtype)
                # n
                # Y(c) = a(hd*c)x(hd) + b
                # a (c*hd)[prj_labedls] -> n*1*hd
                # x = n*hd
                # Y = n * (sum(hd))
                # true_scores = torch.sum(model.classify_weights[proj_labels] * samples_hv, dim=1)      # shape: [wrong_size]
                # wrong_scores = torch.sum(model.classify_weights[argmax] * samples_hv, dim=1) # shape: [wrong_size]
                true_scores = predictions[is_wrong, proj_labels]  # shape: [wrong_size]
                wrong_scores = predictions[is_wrong, argmax]  # shape: [wrong_size
                losses = wrong_scores - true_scores  # shape: [wrong_size]
                if losses.sum().item() < 0:
                    print("Warning: negative losses detected, this is not expected")
                    print("proj_labels: ", proj_labels)
                    print("argmax: ", argmax)
                    print("samples_hv: ", samples_hv)
                    print("true_scores: ", true_scores)
                    print("wrong_scores: ", wrong_scores)
                    print("losses: ", losses)

                # losses = true_scores - wrong_scores          # shape: [wrong_size]
                # print(losses.min(), losses.max())
                # self.is_wrong_list[i][is_wrong] = losses
                wrong_indices_within_selected = is_wrong.nonzero(as_tuple=False).squeeze()
                actual_wrong_indices = indices[wrong_indices_within_selected]
                # self.is_wrong_list[i][actual_wrong_indices] = losses.to(self.is_wrong_list[i].dtype)
                # print("actual_wrong_indices: ", actual_wrong_indices.shape)
                # print("wrong_indices_within_selected: ", wrong_indices_within_selected.shape)
                # print("is_wrong: ", is_wrong.shape)
                # print("self.is_wrong_list[i]: ", self.is_wrong_list[i].shape)
                # print("indices: ", indices.shape)
                # # print("is_wrong_list number of wrong: ", self.is_wrong_list[i].nonzero(as_tuple=False).squeeze().sum().item())
                # print("Number of wrongs:", self.is_wrong_list[i].sum().item())
                self.is_wrong_list[i][actual_wrong_indices] = True
                # print("is_wrong_list number of wrong: ", self.is_wrong_list[i].nonzero(as_tuple=False).squeeze().sum().item())
                # print("Number of wrongs:", self.is_wrong_list[i].sum().item())


                model.classify_weights.index_add_(0, proj_labels, samples_hv)
                model.classify_weights.index_add_(0, proj_labels, samples_hv)
                model.classify_weights.index_add_(0, argmax, -samples_hv)
                model.classify_weights.index_add_(0, argmax, -samples_hv)
                # model.classify.weight[:] = F.normalize(model.classify_weights)

                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                res = time.time() - start
                retrain_time.append(res)
                start = time.time()

                # print("Finish one batch, update classify weights")
            print("total_miss: ", total_miss)
            print("sum of is_wrong_list: ", sum([x.sum().item() for x in self.is_wrong_list if x is not None]))
            print("Mean HDC retraining time:{}\t std:{}".format(np.mean(retrain_time), np.std(retrain_time)))

    def validate(self, val_loader, model, evaluator):  # task_list
        """Validation, evaluate linear classification accuracy and kNN accuracy"""
        # return 0
        losses = AverageMeter()
        jaccs = AverageMeter()
        wces = AverageMeter()
        acc = AverageMeter()
        iou = AverageMeter()
        rand_imgs = []
        evaluator.reset()
        validation_time = []
        class_func=self.parser.get_xentropy_class_string,
        with torch.no_grad():
            for i, (proj_in, proj_mask, proj_labels, unproj_labels, path_seq, path_name, p_x, p_y, proj_range, unproj_range, _, _, _, _, npoints) in enumerate(tqdm(val_loader, desc="Validation")):
                # p_x = p_x[0, :npoints]
                # p_y = p_y[0, :npoints]
                # proj_range = proj_range[0, :npoints]
                # unproj_range = unproj_range[0, :npoints]
                path_seq = path_seq[0]
                path_name = path_name[0]
                B, C, H, W = proj_in.shape[0], proj_in.shape[1], proj_in.shape[2], proj_in.shape[3]

                # print("labels import correct: ", proj_labels) #torch.Size([1, 64, 512])

                if self.gpu:
                    proj_in = proj_in.cuda()
                    # p_x = p_x.cuda()
                    # p_y = p_y.cuda()
                    # if self.post:
                    #     proj_range = proj_range.cuda()
                    #     unproj_range = unproj_range.cuda()
                start = time.time()
                # print("proj_in shape: ", proj_in.shape) #torch.Size([1, 5, 64, 512])
                predictions, _, _, _ = model(proj_in, self.mask)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                res = time.time() - start
                validation_time.append(res)
                start = time.time()
                # print("predictions shape: ", predictions.shape) #torch.Size([32768, 20])
                #print('outputs', outputs)

                predictions = predictions.view(B, H, W, self.num_classes)  # (1, H, W, C)
                predictions = predictions.permute(0, 3, 1, 2)        # → (1, C, H, W)
                # print("predictions shape: ", predictions.shape) #torch.Size([1, 20, 64, 512])
                argmax = predictions.argmax(dim=1)
                argmax = argmax.squeeze(0) 
                # print("argmax shape: ", argmax.shape) #torch.Size([1, 64, 512])
                proj_labels = proj_labels.to(self.device)
                # print("proj_labels shape: ", proj_labels.shape) #torch.Size([1, 64, 512])
                # print("argmax shape: ", argmax.shape) #torch.Size([64, 512])
                # print("proj_labels: ", proj_labels)
                # print("argmax: ", argmax)
                evaluator.addBatch(argmax, proj_labels)

                # if torch.cuda.is_available():
                #     torch.cuda.synchronize()
                # if self.post:
                #     # knn postproc
                #     unproj_argmax = self.post(proj_range,
                #                             unproj_range,
                #                             argmax,
                #                             p_x,
                #                             p_y)
                # else:
                #     # put in original pointcloud using indexes
                #     unproj_argmax = argmax[p_y, p_x]

                # # measure elapsed time
                # if torch.cuda.is_available():
                #     torch.cuda.synchronize()
                # # save scan
                # # get the first scan in batch and project scan
                # pred_np = unproj_argmax.cpu().numpy()
                # pred_np = pred_np.reshape((-1)).astype(np.int32)

                # # map to original label
                # pred_np = to_orig_fn(pred_np)
                # # save scan
                # path = os.path.join(self.logdir, "sequences",
                #                     path_seq, "predictions", path_name)
                # pred_np.tofile(path)
            print("Mean HDC validation time:{}\t std:{}".format(np.mean(validation_time), np.std(validation_time)))
        accuracy = evaluator.getacc()
        jaccard, class_jaccard = evaluator.getIoU()
        acc.update(accuracy.item(), proj_in.size(0))
        iou.update(jaccard.item(), proj_in.size(0))

        print('Validation set:\n'
                'Time avg per batch xxx\n'
                'Loss avg {loss.avg:.4f}\n'
                'Jaccard avg {jac.avg:.4f}\n'
                'WCE avg {wces.avg:.4f}\n'
                'Acc avg {acc.avg:.3f}\n'
                'IoU avg {iou.avg:.3f}'.format(
                                                loss=losses,
                                                jac=jaccs,
                                                wces=wces,
                                                acc=acc, iou=iou))
        
        print('Class Jaccard: ', class_jaccard)
        return iou.avg
    
class UQTrainer():
    def __init__(self, ARCH, DATA, datadir, logdir, modeldir, logger, bipolar_prototypes=False, bipolar_subclusters=False):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger = logger

        self.ARCH = ARCH
        self.DATA = DATA
        self.datadir = datadir
        self.logdir = logdir
        self.modeldir = modeldir
        self.epochs = 20

        from dataset.kitti.parser import Parser
        self.parser = Parser(root=self.datadir,
                                        train_sequences=self.DATA["split"]["train"],
                                        valid_sequences=self.DATA["split"]["valid"],
                                        test_sequences=None,
                                        labels=self.DATA["labels"],
                                        color_map=self.DATA["color_map"],
                                        learning_map=self.DATA["learning_map"],
                                        learning_map_inv=self.DATA["learning_map_inv"],
                                        sensor=self.ARCH["dataset"]["sensor"],
                                        max_points=self.ARCH["dataset"]["max_points"],
                                        batch_size=self.ARCH["train"]["batch_size"],
                                        workers=self.ARCH["train"]["workers"],
                                        gt=True,
                                        shuffle_train=False)
        self.num_classes = self.parser.get_n_classes() 
        epsilon_w = self.ARCH["train"]["epsilon_w"]
        content = torch.zeros(self.parser.get_n_classes(), dtype=torch.float)
        for cl, freq in DATA["content"].items():
            x_cl = self.parser.to_xentropy(cl)
            content[x_cl] += freq
        self.loss_w = 1 / (content + epsilon_w)
        for x_cl, w in enumerate(self.loss_w):
            if DATA["learning_ignore"][x_cl]:
                self.loss_w[x_cl] = 0
        print("Loss weights from content: ", self.loss_w.data)

        self.model = set_uq_model(ARCH, modeldir, 'rp', 0, 0, self.num_classes, self.device, subcluster_type='bipolar' if bipolar_subclusters else 'continuous')
        print(self.parser.get_n_classes())
        self.post = None
        if self.ARCH["post"]["KNN"]["use"]:
            self.post = KNN(self.ARCH["post"]["KNN"]["params"],
                            self.parser.get_n_classes())
        print(self.parser.get_n_classes())

        self.gpu = False
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Infering in device: ", self.device)
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            cudnn.benchmark = True
            cudnn.fastest = True
            self.gpu = True
            self.model.cuda()

        self.bipolar_prototypes = bipolar_prototypes

    def start(self):
        print("Starting training with the HDC online learning:")
        self.model.eval()
        self.ignore_class = []
        for i, w in enumerate(self.loss_w):
            if w < 1e-10:
                self.ignore_class.append(i)
                print("Ignoring class ", i, " in IoU evaluation")
        self.evaluator = iouEval(self.parser.get_n_classes(),
                                 self.device, self.ignore_class)
        for e in range(1, 2):
            time1 = time.time()
            self.train(self.parser.get_train_set(), self.model, self.logger)
            time2 = time.time()
            print('train epoch {}, total time {:.2f}'.format(e, time2 - time1))
        for epoch in range(1, self.epochs + 1):
            time1 = time.time()
            self.retrain(self.parser.get_train_set(), self.model, epoch, self.logger)

            time2 = time.time()
            print('retrain epoch {}, total time {:.2f}'.format(epoch, time2 - time1))

            acc = self.validate(self.parser.get_valid_set(), self.model, self.evaluator)
            print('Stream final acc: {}'.format(acc))

    def train(self, train_loader, model, logger):  # task_list
        """Training on single-pass of data"""
        batchs_per_class = np.floor(len(train_loader) / self.num_classes).astype('int')
        if self.gpu:
            torch.cuda.empty_cache()
        with torch.no_grad():
            idx = 0  # batch index
            cur_class = -1
            self.mask = None
            train_time = []
            self.is_wrong_list = [None] * len(train_loader)  # store the wrong classification for each batch
            for i, (proj_in, proj_mask, proj_labels, unproj_labels, path_seq, path_name, p_x, p_y, proj_range, unproj_range, _, _, _, _, npoints) in enumerate(tqdm(train_loader, desc="Training")):
                path_seq = path_seq[0]
                path_name = path_name[0]

                if self.gpu:
                    proj_in = proj_in.cuda()
                    proj_mask = proj_mask.cuda()

                start = time.time()
                samples_hv, _, _ = self.model.encode(proj_in, self.mask)
                samples_hv = samples_hv.to(model.classify_weights.dtype)

                proj_labels = proj_labels.view(-1)  # shape: (btsz*64*512, 1)
                proj_labels = proj_labels.to(self.device)

                model.classify_weights.index_add_(0, proj_labels, samples_hv)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                res = time.time() - start
                train_time.append(res)
                start = time.time()

                predictions =self.model.get_predictions(samples_hv)
                # print("predictions: ", predictions) #torch.Size([32768, 20])
                argmax = predictions.argmax(dim=1) # (bsz*size, 1)
                # self.is_wrong_list[i] = proj_labels != argmax

                is_wrong = proj_labels != argmax
                proj_labels = proj_labels[is_wrong]
                argmax = argmax[is_wrong]
                samples_hv = samples_hv[is_wrong]
                samples_hv = samples_hv.to(model.classify_weights.dtype)

                true_scores = predictions[is_wrong, proj_labels]  # shape: [wrong_size]
                wrong_scores = predictions[is_wrong, argmax]  # shape: [wrong_size]
                # # losses = wrong_scores - true_scores  # shape: [wrong_size]
                losses = wrong_scores - true_scores  # shape: [wrong_size]

                self.is_wrong_list[i] = is_wrong

            if self.bipolar_prototypes:
                # making prototypes bipolar instead of continuous
                with torch.no_grad():
                    model.classify_weights.data = torch.sign(model.classify_weights.data)

                    zero_mask = model.classify_weights.data == 0
                    if torch.any(zero_mask):
                        print(f"Warning: Found {zero_mask.sum().item()} zeros after sign(), replacing with -1")
                        model.classify_weights.data[zero_mask] = -1.0
                    
                    model.classify.weight.data = model.classify_weights.data.clone()
            else:
                model.classify.weight[:] = F.normalize(model.classify_weights)

    
            print("sum of is_wrong_list: ", sum([x.sum().item() for x in self.is_wrong_list if x is not None]))
            print("Mean HDC training time:{}\t std:{}".format(np.mean(train_time), np.std(train_time)))
            # print("Finish one batch, update classify weights")
    
    def retrain(self, train_loader, model, epoch, logger):  # task_list
        """Training of one epoch on single-pass of data"""
        batchs_per_class = np.floor(len(train_loader) / self.num_classes).astype('int')
        if self.gpu:
            torch.cuda.empty_cache()
        with torch.no_grad():
            # for idx, (images, labels) in enumerate(train_loader):
            idx = 0  # batch index
            cur_class = -1
            total_miss = 0
            retrain_time = []
            for i, (proj_in, proj_mask, proj_labels, unproj_labels, path_seq, path_name, p_x, p_y, proj_range, unproj_range, _, _, _, _, npoints) in enumerate(tqdm(train_loader, desc="Retraining")):
                path_seq = path_seq[0]
                path_name = path_name[0]

                if self.gpu:
                    proj_in = proj_in.cuda()
                    proj_mask = proj_mask.cuda()

                start = time.time()

                if self.bipolar_prototypes:
                    # making prototypes bipolar instead of 
                    with torch.no_grad():
                        model.classify_weights.data = torch.sign(model.classify_weights.data)

                        zero_mask = model.classify_weights.data == 0
                        if torch.any(zero_mask):
                            print(f"Warning: Found {zero_mask.sum().item()} zeros after sign(), replacing with -1")
                            model.classify_weights.data[zero_mask] = -1.0
                        
                        model.classify.weight.data = model.classify_weights.data.clone()
                else:
                    model.classify.weight[:] = F.normalize(model.classify_weights)

                # print("Number of wrongs:", self.is_wrong_list[i].sum().item())
                predictions, samples_hv, indices, self.is_wrong_list[i] = model(proj_in, self.mask, 0.05, self.is_wrong_list[i])
                argmax = predictions.argmax(dim=1) # (bsz*size, 1)
                # #proj_labels shape: torch.Size([1, 64, 512])
                proj_labels = proj_labels.view(-1)  # shape: (btsz*64*512, 1) 
                proj_labels = proj_labels.to(self.device)
                proj_labels = proj_labels[indices]  # map to the sampled hypervectors

                is_wrong = proj_labels != argmax
                # self.is_wrong_list[i][indices[is_wrong]] = True
                
                if is_wrong.sum().item() == 0:
                    continue

                # Check wrong classification number here and update the classify weights
                total_miss += is_wrong.sum().item()
                proj_labels = proj_labels[is_wrong]
                argmax = argmax[is_wrong]
                samples_hv = samples_hv[is_wrong]
                samples_hv = samples_hv.to(model.classify_weights.dtype)

                true_scores = predictions[is_wrong, proj_labels]  # shape: [wrong_size]
                wrong_scores = predictions[is_wrong, argmax]  # shape: [wrong_size
                losses = wrong_scores - true_scores  # shape: [wrong_size]
                if losses.sum().item() < 0:
                    print("Warning: negative losses detected, this is not expected")
                    print("proj_labels: ", proj_labels)
                    print("argmax: ", argmax)
                    print("samples_hv: ", samples_hv)
                    print("true_scores: ", true_scores)
                    print("wrong_scores: ", wrong_scores)
                    print("losses: ", losses)

                wrong_indices_within_selected = is_wrong.nonzero(as_tuple=False).squeeze()
                actual_wrong_indices = indices[wrong_indices_within_selected]

                self.is_wrong_list[i][actual_wrong_indices] = True

                model.classify_weights.index_add_(0, proj_labels, samples_hv)
                # model.classify_weights.index_add_(0, proj_labels, samples_hv)
                model.classify_weights.index_add_(0, argmax, -samples_hv)
                # model.classify_weights.index_add_(0, argmax, -samples_hv)

                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                res = time.time() - start
                retrain_time.append(res)
                start = time.time()

                # print("Finish one batch, update classify weights")
            print("total_miss: ", total_miss)
            print("sum of is_wrong_list: ", sum([x.sum().item() for x in self.is_wrong_list if x is not None]))
            print("Mean HDC retraining time:{}\t std:{}".format(np.mean(retrain_time), np.std(retrain_time)))

    def validate(self, val_loader, model, evaluator):  # task_list
        """Validation, evaluate linear classification accuracy and kNN accuracy"""
        # return 0
        losses = AverageMeter()
        jaccs = AverageMeter()
        wces = AverageMeter()
        acc = AverageMeter()
        iou = AverageMeter()
        rand_imgs = []
        evaluator.reset()
        validation_time = []
        class_func=self.parser.get_xentropy_class_string,
        with torch.no_grad():
            for i, (proj_in, proj_mask, proj_labels, unproj_labels, path_seq, path_name, p_x, p_y, proj_range, unproj_range, _, _, _, _, npoints) in enumerate(tqdm(val_loader, desc="Validation")):

                path_seq = path_seq[0]
                path_name = path_name[0]
                B, C, H, W = proj_in.shape[0], proj_in.shape[1], proj_in.shape[2], proj_in.shape[3]

                if self.gpu:
                    proj_in = proj_in.cuda()

                start = time.time()
                predictions, _, _, _ = model(proj_in, self.mask)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                res = time.time() - start
                validation_time.append(res)
                start = time.time()

                predictions = predictions.view(B, H, W, self.num_classes)  # (1, H, W, C)
                predictions = predictions.permute(0, 3, 1, 2)        # → (1, C, H, W)
                # print("predictions shape: ", predictions.shape) #torch.Size([1, 20, 64, 512])
                argmax = predictions.argmax(dim=1)
                argmax = argmax.squeeze(0) 
                # print("argmax shape: ", argmax.shape) #torch.Size([1, 64, 512])
                proj_labels = proj_labels.to(self.device)

                evaluator.addBatch(argmax, proj_labels)

            print("Mean HDC validation time:{}\t std:{}".format(np.mean(validation_time), np.std(validation_time)))
        accuracy = evaluator.getacc()
        jaccard, class_jaccard = evaluator.getIoU()
        acc.update(accuracy.item(), proj_in.size(0))
        iou.update(jaccard.item(), proj_in.size(0))

        print('Validation set:\n'
                'Time avg per batch xxx\n'
                'Loss avg {loss.avg:.4f}\n'
                'Jaccard avg {jac.avg:.4f}\n'
                'WCE avg {wces.avg:.4f}\n'
                'Acc avg {acc.avg:.3f}\n'
                'IoU avg {iou.avg:.3f}'.format(
                                                loss=losses,
                                                jac=jaccs,
                                                wces=wces,
                                                acc=acc, iou=iou))
        
        print('Class Jaccard: ', class_jaccard)
        return iou.avg