import os
import logging
import numpy as np
import torch
import yaml
import matplotlib.pyplot as plt

from dataset.kitti.parser import Parser
from modules.HDC_utils import Model, UQModel
from modules.trainer import Trainer
from modules.Basic_HD import UQTrainer

MODEL_DIR = "logs"
NU_DATA_DIR = "/mnt/alpha/jmfleming/HyperLidar_dataset/nuscenes_all"
DATA_DIR = "/mnt/alpha/jmfleming/nuscenes_kitti"
LOG_DIR = "logs"
NUM_CLASSES = 17 

MAX_HDC_EPOCHS = 10
FEATURE_EXTRACTOR_EPOCHS = 80

HD_DIM = 10000

HDC_SAVE_PATH = "logs/hdc.pth"
HDC_SUB_PATH = "logs/hdc_sub.pth"

def setup_logger(log_file):
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logger = logging.getLogger() # Get root logger
    logger.setLevel(logging.INFO)
    
    # Remove all existing handlers
    if logger.hasHandlers():
        logger.handlers.clear()
        
    fh = logging.FileHandler(log_file, mode='w')
    fh.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger

def save_graphic(save_path, title, data):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.figure()
    if isinstance(data, dict):
        for label, values in data.items():
            plt.plot(values, label=label)
        plt.legend()
    else:
        plt.plot(data)
    plt.title(title)
    plt.xlabel('Steps')
    plt.ylabel('Metric')
    plt.savefig(save_path)
    plt.close()

def extract_metrics_from_conf_matrix(conf_matrix):
    tp = torch.diag(conf_matrix)
    union = conf_matrix.sum(dim=1) + conf_matrix.sum(dim=0) - tp
    iou_per_class = tp / (union + 1e-6)
    
    # Exclude class 0 (unlabeled/ignore)
    valid_classes = union > 0 
    valid_classes[0] = False
    
    miou = iou_per_class[valid_classes].mean().item()
    
    # SemanticKITTI groupings (mapping: 2:bike, 3:bus, 4:car, 6:moto, 7:person, 10:truck, 11:driveable, 12:other_flat, 13:sidewalk, 14:terrain, 15:manmade, 16:vegetation)
    head_mask = torch.zeros_like(valid_classes)
    head_mask[[11, 13, 14, 15, 16]] = True
    head_mask = head_mask & valid_classes
    
    mid_mask = torch.zeros_like(valid_classes)
    mid_mask[[4, 12]] = True
    mid_mask = mid_mask & valid_classes
    
    tail_mask = torch.zeros_like(valid_classes)
    tail_mask[[2, 3, 6, 7, 10]] = True
    tail_mask = tail_mask & valid_classes
    
    head_miou = iou_per_class[head_mask].mean().item() if head_mask.any() else 0.0
    mid_miou = iou_per_class[mid_mask].mean().item() if mid_mask.any() else 0.0
    tail_miou = iou_per_class[tail_mask].mean().item() if tail_mask.any() else 0.0
    
    # Calculate overall accuracy excluding class 0
    total_correct_valid = tp[1:].sum().item()
    total_samples_valid = conf_matrix[1:, :].sum().item()
    overall_acc = total_correct_valid / (total_samples_valid + 1e-6)
    
    return miou, head_miou, mid_miou, tail_miou, overall_acc, iou_per_class.cpu().tolist()

def load_hdc_model(path):
    print(f"Loading pretrained HDC model from {path}...")
    from modules.HDC_utils import UQModel
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ARCH = yaml.safe_load(open("config/arch/senet-2048p.yml", 'r'))
    import os
    modeldir = os.path.dirname(path)
    weights_path = os.path.join(modeldir, "SENet_valid_best")
    if os.path.exists(weights_path):
        tmp_dict = torch.load(weights_path, map_location='cpu')
        NUM_CLASSES = tmp_dict['state_dict']['semantic_output.bias'].shape[0]
    else:
        NUM_CLASSES = 13 # Fallback
    model = UQModel(ARCH, modeldir, 'rp', 0, 0, NUM_CLASSES, device, subcluster_type='continuous')
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    return model


def train_extractor(ARCH, DATA, epochs=FEATURE_EXTRACTOR_EPOCHS, data_dir=None, return_trainer=False, resume_path=None):
    trainer = Trainer(ARCH, DATA, data_dir if data_dir else DATA_DIR, LOG_DIR, path=resume_path) # saves in "/logs/SENet_..."
    trainer.train(epochs=epochs)

    if return_trainer:
        return trainer

def train_hdc(ARCH, DATA, epochs=MAX_HDC_EPOCHS, data_dir=None, return_extractor=False) -> UQModel:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    parser = Parser(root=data_dir if data_dir else DATA_DIR,
                        train_sequences=DATA["split"]["train"], # self.DATA["split"]["valid"] + self.DATA["split"]["train"] if finetune with valid
                        valid_sequences=DATA["split"]["valid"],
                        test_sequences=None,
                        labels=DATA["labels"],
                        color_map=DATA["color_map"],
                        learning_map=DATA["learning_map"],
                        learning_map_inv=DATA["learning_map_inv"],
                        sensor=ARCH["dataset"]["sensor"],
                        max_points=ARCH["dataset"]["max_points"],
                        batch_size=ARCH["train"]["batch_size"],
                        workers=ARCH["train"]["workers"],
                        gt=True,
                        shuffle_train=True)
    
    dataloader = parser.get_train_set()
    val_loader = parser.get_valid_set() # val_loader is empty???

    ignore = []
    for cl, ign in DATA['learning_ignore'].items():
        if ign:
            x_cl = int(cl)
            ignore.append(x_cl)

    trainer = UQTrainer(ARCH, DATA, DATA_DIR, LOG_DIR, MODEL_DIR, None)

    trainer.train(dataloader, trainer.model, None)

    for i in range(epochs - 1):
        trainer.retrain(dataloader, trainer.model, i+1, None)
        # Save checkpoint after each epoch so training can be picked up if interrupted
        torch.save(trainer.model.state_dict(), HDC_SAVE_PATH)

    model: UQModel = trainer.model
    torch.save(model.state_dict(), HDC_SAVE_PATH)

    if return_extractor: return model, trainer

    return model

def test_hdc_model(model, dataloader, return_detailed=False) -> None:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    all_accuracies = []
    class_correct = torch.zeros(model.num_classes, device=device)
    class_total = torch.zeros(model.num_classes, device=device)

    class_intersection = torch.zeros(model.num_classes, device=device)
    class_union = torch.zeros(model.num_classes, device=device)
    
    global_correct = 0
    global_total = 0
    model.eval()
    
    with torch.no_grad():
        for _, batch_data in enumerate(dataloader):
            proj_in = batch_data[0].to(device)
            proj_labels = batch_data[2].to(device)
            logits, _, indices, _ = model(proj_in, PERCENTAGE=None, is_wrong=None)
            
            predictions = torch.argmax(logits, dim=1)
            proj_labels_flat = proj_labels.view(-1)
            selected_labels = proj_labels_flat[indices]
            
            batch_correct = ((predictions == selected_labels) & (selected_labels > 0)).sum().item()
            batch_total = (selected_labels > 0).sum().item()
            batch_accuracy = batch_correct / batch_total if batch_total > 0 else 0
            all_accuracies.append(batch_accuracy)
            global_correct += batch_correct
            global_total += batch_total
            
            valid_eval_mask = (selected_labels > 0)
            for class_id in range(model.num_classes):
                class_mask = (selected_labels == class_id)
                pred_mask = (predictions == class_id) & valid_eval_mask
                
                if class_mask.any():
                    class_correct[class_id] += (predictions[class_mask] == class_id).sum().item()
                    class_total[class_id] += class_mask.sum().item()

                intersection = (class_mask & pred_mask).sum().item()
                union = (class_mask | pred_mask).sum().item()
                
                class_intersection[class_id] += intersection
                class_union[class_id] += union
    
    global_accuracy = global_correct / global_total if global_total > 0 else 0
    mean_batch_accuracy = np.mean(all_accuracies) if all_accuracies else 0
    
    per_class_accuracy = {}
    per_class_iou = {}
    valid_ious = []
    
    for class_id in range(model.num_classes):
        if class_total[class_id] > 0:
            per_class_accuracy[class_id] = (class_correct[class_id] / class_total[class_id]).item()
        else:
            per_class_accuracy[class_id] = 0.0
        
        # Calculate IoU for each class (Exclude Class 0 which is typically the ignored/unlabeled class)
        if class_union[class_id] > 0:
            iou = (class_intersection[class_id] / class_union[class_id]).item()
            per_class_iou[class_id] = iou
            if class_id > 0:
                valid_ious.append(iou)
        else:
            per_class_iou[class_id] = 0.0

    miou = np.mean(valid_ious) if valid_ious else 0.0
    
    print(f"\n{'='*60}")
    print("Training Set Accuracy Results")
    print(f"{'='*60}")
    print(f"Global Accuracy: {global_accuracy:.4f} ({global_correct}/{global_total})")
    print(f"Mean Batch Accuracy: {mean_batch_accuracy:.4f}")
    print(f"mIOU: {miou:.4f}")
    print()
    print("Per-Class Accuracies:")
    for class_id in sorted(range(model.num_classes)):
        if class_total[class_id] > 0:
            acc = per_class_accuracy[class_id]
            iou = per_class_iou[class_id]
            correct = int(class_correct[class_id].item())
            total = int(class_total[class_id].item())
            print(f"  Class {class_id}: Acc={acc:.4f} ({correct}/{total}), IoU={iou:.4f}")
        else:
            print(f"  Class {class_id}: No samples")

    if return_detailed:
        detailed_stats = {
            "per_class_acc": per_class_accuracy,
            "per_class_iou": per_class_iou,
            "class_total": {i: int(class_total[i].item()) for i in range(model.num_classes)},
            "class_correct": {i: int(class_correct[i].item()) for i in range(model.num_classes)},
            "class_intersection": {i: int(class_intersection[i].item()) for i in range(model.num_classes)},
            "class_union": {i: int(class_union[i].item()) for i in range(model.num_classes)}
        }
        return global_accuracy, miou, detailed_stats

    return global_accuracy, miou