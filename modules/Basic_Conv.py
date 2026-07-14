import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tqdm import tqdm
from postproc.KNN import KNN
from common.avgmeter import *
from dataset.kitti.parser import Parser
import torch.backends.cudnn as cudnn
from modules.ioueval import *
from modules.scheduler.warmupLR import warmupLR
from modules.losses.Lovasz_Softmax import Lovasz_softmax
from modules.scheduler.cosine import CosineAnnealingWarmUpRestarts

def load_conv_model(ARCH, modeldir, num_classes, device):
    # Load encoder (frozen)
    with torch.no_grad():
        torch.nn.Module.dump_patches = True
        if ARCH["train"]["pipeline"] == "hardnet":
            from modules.network.HarDNet import HarDNet
            net = HarDNet(num_classes, ARCH["train"]["aux_loss"])

        if ARCH["train"]["pipeline"] == "res":
            from modules.network.ResNet import ResNet_34
            net = ResNet_34(num_classes, ARCH["train"]["aux_loss"])

            def convert_relu_to_softplus(model, act):
                for child_name, child in model.named_children():
                    if isinstance(child, nn.LeakyReLU):
                        setattr(model, child_name, act)
                    else:
                        convert_relu_to_softplus(child, act)

            if ARCH["train"]["act"] == "Hardswish":
                convert_relu_to_softplus(net, nn.Hardswish())
            elif ARCH["train"]["act"] == "SiLU":
                convert_relu_to_softplus(net, nn.SiLU())

        if ARCH["train"]["pipeline"] == "fid":
            from modules.network.Fid import ResNet_34
            net = ResNet_34(num_classes, ARCH["train"]["aux_loss"])

            if ARCH["train"]["act"] == "Hardswish":
                convert_relu_to_softplus(net, nn.Hardswish())
            elif ARCH["train"]["act"] == "SiLU":
                convert_relu_to_softplus(net, nn.SiLU())
    w_dict = torch.load(modeldir + "/SENet_valid_best",
                        map_location=lambda storage, loc: storage)
    net.load_state_dict(w_dict['state_dict'], strict=True)
    net.eval()

    # Trainable classifier
    classifier = nn.Conv2d(128, num_classes, kernel_size=1).to(device)

    return net, classifier

class BasicConv():
    def __init__(self, ARCH, DATA, datadir, logdir, modeldir, logger):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger = logger

        self.ARCH = ARCH
        self.DATA = DATA
        self.datadir = datadir
        self.logdir = logdir
        self.modeldir = modeldir
        self.epochs = ARCH["train"]["max_epochs"]

        self.parser = Parser(
            root=self.datadir,
            train_sequences=self.DATA["split"]["train"],
            valid_sequences=self.DATA["split"]["valid"],
            test_sequences=self.DATA["split"]["test"],
            labels=self.DATA["labels"],
            color_map=self.DATA["color_map"],
            learning_map=self.DATA["learning_map"],
            learning_map_inv=self.DATA["learning_map_inv"],
            sensor=self.ARCH["dataset"]["sensor"],
            max_points=self.ARCH["dataset"]["max_points"],
            batch_size=self.ARCH["train"]["batch_size"],
            workers=self.ARCH["train"]["workers"],
            gt=True,
            shuffle_train=False
        )

        self.num_classes = self.parser.get_n_classes()
        epsilon_w = self.ARCH["train"]["epsilon_w"]
        content = torch.zeros(self.num_classes, dtype=torch.float)
        for cl, freq in DATA["content"].items():
            x_cl = self.parser.to_xentropy(cl)
            content[x_cl] += freq
        self.loss_w = 1 / (content + epsilon_w)
        for x_cl, w in enumerate(self.loss_w):
            if DATA["learning_ignore"][x_cl]:
                self.loss_w[x_cl] = 0
        print("Loss weights from content: ", self.loss_w.data)

        self.encoder, self.semantic_output = load_conv_model(ARCH, modeldir, self.num_classes, self.device)

        if self.ARCH["post"]["KNN"]["use"]:
            self.post = KNN(self.ARCH["post"]["KNN"]["params"], self.num_classes)
        else:
            self.post = None

        self.criterion = nn.CrossEntropyLoss(weight=self.loss_w.to(self.device))
        
        if self.ARCH["train"]["scheduler"] == "consine":
            length = self.parser.get_train_size()
            dict = self.ARCH["train"]["consine"]
            self.optimizer = optim.SGD(self.semantic_output.parameters(),
                                       lr=dict["min_lr"],
                                       momentum=self.ARCH["train"]["momentum"],
                                       weight_decay=self.ARCH["train"]["w_decay"])
            self.scheduler = CosineAnnealingWarmUpRestarts(optimizer=self.optimizer,
                                                           T_0=dict["first_cycle"] * length, T_mult=dict["cycle"],
                                                           eta_max=dict["max_lr"],
                                                           T_up=dict["wup_epochs"]*length, gamma=dict["gamma"])

        else:
            self.optimizer = optim.SGD(self.semantic_output.parameters(),
                                       lr=self.ARCH["train"]["decay"]["lr"],
                                       momentum=self.ARCH["train"]["momentum"],
                                       weight_decay=self.ARCH["train"]["w_decay"])
            steps_per_epoch = self.parser.get_train_size()
            up_steps = int(self.ARCH["train"]["decay"]["wup_epochs"] * steps_per_epoch)
            final_decay = self.ARCH["train"]["decay"]["lr_decay"] ** (1 / steps_per_epoch)
            self.scheduler = warmupLR(optimizer=self.optimizer,
                                      lr=self.ARCH["train"]["decay"]["lr"],
                                      warmup_steps=up_steps,
                                      momentum=self.ARCH["train"]["momentum"],
                                      decay=final_decay)
        # GPU?
        self.gpu = False
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Infering in device: ", self.device)
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            cudnn.benchmark = True
            cudnn.fastest = True
            self.gpu = True
            self.encoder.cuda()
            self.semantic_output.cuda()

    def train(self, train_loader, encoder, semantic_output, criterion, optimizer, epoch, evaluator, scheduler):
        losses = AverageMeter()
        acc = AverageMeter()
        iou = AverageMeter()
        update_ratio_meter = AverageMeter()

        # empty the cache to train now
        if self.gpu:
            torch.cuda.empty_cache()
            
        encoder.eval()
        semantic_output.train()
        scaler = torch.amp.GradScaler('cuda')
        train_time = []
        for i, (proj_in, proj_mask, proj_labels, unproj_labels, path_seq, path_name, p_x, p_y, proj_range, unproj_range, _, _, _, _, npoints) in enumerate(tqdm(train_loader, desc="Training")):
            proj_in = proj_in.to(self.device)
            proj_mask = proj_mask.to(self.device)

            if self.gpu:
                proj_labels = proj_labels.cuda().long()
            start = time.time()
            with torch.amp.autocast('cuda'):
                features = self.encoder(proj_in, only_feat=True)
                logits = self.semantic_output(features)
                loss = criterion(logits, proj_labels)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            if torch.cuda.is_available():
                    torch.cuda.synchronize()
            res = time.time() - start
            train_time.append(res)
            start = time.time()
            with torch.no_grad():
                evaluator.reset()
                argmax = logits.argmax(dim=1)
                evaluator.addBatch(argmax, proj_labels)
                accuracy = evaluator.getacc()
                jaccard, class_jaccard = evaluator.getIoU()
            losses.update(loss.item(), proj_in.size(0))
            acc.update(accuracy.item(), proj_in.size(0))
            iou.update(jaccard.item(), proj_in.size(0))
            scheduler.step()
        print("Mean Conv training time:{}\t std:{}".format(np.mean(train_time), np.std(train_time)))
        return iou.avg

    def validate(self, val_loader, encoder, semantic_output, criterion, evaluator):
        losses = AverageMeter()
        jaccs = AverageMeter()
        wces = AverageMeter()
        acc = AverageMeter()
        iou = AverageMeter()
        validation_time = []
        
        encoder.eval()
        semantic_output.eval()
        evaluator.reset()
        # empty the cache to infer in high res
        if self.gpu:
            torch.cuda.empty_cache()
        for i, (proj_in, proj_mask, proj_labels, unproj_labels, path_seq, path_name, p_x, p_y, proj_range, unproj_range, _, _, _, _, npoints) in enumerate(tqdm(val_loader, desc="Validation")):
            if self.gpu:
                proj_in = proj_in.cuda()
                proj_mask = proj_mask.cuda()
            if self.gpu:
                proj_labels = proj_labels.cuda(non_blocking=True).long()
            start = time.time()
            with torch.no_grad():
                features = self.encoder(proj_in, only_feat=True)
                logits = self.semantic_output(features)
                predictions = logits.argmax(dim=1)
            if torch.cuda.is_available():
                    torch.cuda.synchronize()
            res = time.time() - start
            validation_time.append(res)
            start = time.time()

            evaluator.addBatch(predictions, proj_labels)
        accuracy = evaluator.getacc()
        jaccard, class_jaccard = evaluator.getIoU()
        acc.update(accuracy.item(), proj_in.size(0))
        iou.update(jaccard.item(), proj_in.size(0))
        print("Mean Conv validation time:{}\t std:{}".format(np.mean(validation_time), np.std(validation_time)))
        return iou.avg

    def start(self):
        print("Starting training with the Conv online learning:")
        self.ignore_class = []
        for i, w in enumerate(self.loss_w):
            if w < 1e-10:
                self.ignore_class.append(i)
                print("Ignoring class ", i, " in IoU evaluation")
        self.evaluator = iouEval(self.parser.get_n_classes(),
                                 self.device, self.ignore_class)

        optimizer = torch.optim.Adam(self.semantic_output.parameters(), weight_decay=self.ARCH["train"]["w_decay"])

        best_miou = 0.0
        for epoch in range(self.epochs):
            print(f"Epoch {epoch+1}/{self.epochs}")
            train_miou = self.train(train_loader=self.parser.get_train_set(),
                                    encoder=self.encoder,
                                    semantic_output=self.semantic_output,
                                    criterion=self.criterion,
                                    optimizer=self.optimizer,
                                    epoch=epoch,
                                    evaluator=self.evaluator,
                                    scheduler=self.scheduler)
            val_miou = self.validate(val_loader=self.parser.get_valid_set(),
                                    encoder=self.encoder,
                                    semantic_output=self.semantic_output,
                                    criterion=self.criterion,
                                    evaluator=self.evaluator)
            print(f"Train mIou: {train_miou:.4f}, Val mIoU: {val_miou:.4f}")

            if val_miou > best_miou:
                print("Saving best model...")
                os.makedirs(self.logdir, exist_ok=True)
                torch.save(self.semantic_output.state_dict(), os.path.join(self.logdir, "best_conv_classifier.pth"))
                best_miou = val_miou