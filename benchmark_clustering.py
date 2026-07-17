import os
import time
import torch
import numpy as np
from tqdm import tqdm
from sklearn.cluster import KMeans, MiniBatchKMeans, DBSCAN, BisectingKMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
import yaml

from dataset.kitti.parser import Parser
from modules.HDC_utils import set_uq_model

def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def run_benchmarks():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load configs
    arch_cfg = load_config('config/arch/senet-2048p.yml')
    data_cfg = load_config('config/labels/semantic-kitti-all.yaml')
    
    data_dir = '/mnt/alpha/jmfleming/KITTI'
    
    print(f"Initializing Dataset...")
    parser = Parser(root=data_dir,
                    train_sequences=data_cfg["split"]["train"],
                    valid_sequences=data_cfg["split"]["valid"],
                    test_sequences=None,
                    labels=data_cfg["labels"],
                    color_map=data_cfg.get("color_map", {}),
                    learning_map=data_cfg["learning_map"],
                    learning_map_inv=data_cfg["learning_map_inv"],
                    sensor=arch_cfg["dataset"]["sensor"],
                    max_points=arch_cfg["dataset"]["max_points"],
                    batch_size=1,
                    workers=4,
                    gt=True,
                    shuffle_train=True) 
    
    dataloader = torch.utils.data.DataLoader(parser.trainloader.dataset, batch_size=1, shuffle=True, num_workers=4)
    
    print("Loading Pretrained Model...")
    model_dir = 'logs/kitti_pretrain'
    hd_encoder = 'HDnn'
    num_levels = 100
    randomness = 'gaussian'
    num_classes = 20
    
    model = set_uq_model(arch_cfg, model_dir, hd_encoder, num_levels, randomness, num_classes, device)
    model.load_state_dict(torch.load(os.path.join(model_dir, 'hdc_sub.pth'), map_location=device), strict=False)
    model.eval()
    
    # We will collect embeddings for a few specific classes to benchmark
    # 9 = road (common), 10 = parking (rare), 1 = car (common), 4 = pedestrian (rare)
    target_classes = [9, 10, 1, 4]
    collected_features = {c: [] for c in target_classes}
    max_features_per_class = 20000
    
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(tqdm(dataloader, desc="Extracting Features")):
            if batch_idx > 200:
                break
            proj_in = batch_data[0].to(device)
            proj_labels = batch_data[2].to(device).view(-1)
            
            valid_mask = (proj_labels > 0) & (proj_labels < num_classes)
            if not valid_mask.any():
                continue
                
            latent_x = model.net(proj_in)
            latent_x = latent_x.permute(0, 2, 3, 1).reshape(-1, 128)
            import torch.nn.functional as F
            latent_x = F.normalize(latent_x, dim=1)
            
            valid_latent = latent_x[valid_mask]
            valid_labels = proj_labels[valid_mask]
            
            for c in target_classes:
                c_mask = valid_labels == c
                if c_mask.any():
                    feats = valid_latent[c_mask].cpu().numpy()
                    collected_features[c].append(feats)
                    
            # Stop early if we have enough data for all target classes
            all_full = True
            for c in target_classes:
                if sum(len(f) for f in collected_features[c]) < max_features_per_class:
                    all_full = False
            if all_full:
                break
                
    print("\n--- Clustering Benchmarks ---")
    K = 10
    
    for c in target_classes:
        if len(collected_features[c]) == 0:
            print(f"Skipping Class {c}: Not enough data found.")
            continue
            
        data = np.concatenate(collected_features[c], axis=0)
        # Downsample if too many
        if data.shape[0] > max_features_per_class:
            indices = np.random.choice(data.shape[0], max_features_per_class, replace=False)
            data = data[indices]
            
        print(f"\nEvaluating Class {c} (N = {data.shape[0]})")
        
        # 1. Standard KMeans
        start = time.time()
        kmeans = KMeans(n_clusters=K, n_init='auto', random_state=42)
        labels_kmeans = kmeans.fit_predict(data)
        time_kmeans = time.time() - start
        
        try:
            score_kmeans = silhouette_score(data, labels_kmeans, sample_size=10000, metric='cosine')
        except ValueError:
            score_kmeans = -1
        
        print(f"  [KMeans]         Time: {time_kmeans:.3f}s | Silhouette: {score_kmeans:.3f} | Clusters: {K}")
        
        # 2. MiniBatch KMeans
        start = time.time()
        mb_kmeans = MiniBatchKMeans(n_clusters=K, n_init='auto', batch_size=1024, random_state=42)
        labels_mb = mb_kmeans.fit_predict(data)
        time_mb = time.time() - start
        
        try:
            score_mb = silhouette_score(data, labels_mb, sample_size=10000, metric='cosine')
        except ValueError:
            score_mb = -1
            
        print(f"  [MiniBatch]      Time: {time_mb:.3f}s | Silhouette: {score_mb:.3f} | Clusters: {K}")
        
        # 3. DBSCAN
        start = time.time()
        # Cosine distance eps: max distance is 2.0, so 0.05 is a tight cluster
        dbscan = DBSCAN(eps=0.05, min_samples=20, metric='cosine', n_jobs=-1)
        labels_dbscan = dbscan.fit_predict(data)
        time_dbscan = time.time() - start
        num_clusters_dbscan = len(set(labels_dbscan)) - (1 if -1 in labels_dbscan else 0)
        
        try:
            if num_clusters_dbscan > 1:
                score_dbscan = silhouette_score(data, labels_dbscan, sample_size=10000, metric='cosine')
            else:
                score_dbscan = -1
        except ValueError:
            score_dbscan = -1
            
        print(f"  [DBSCAN]         Time: {time_dbscan:.3f}s | Silhouette: {score_dbscan:.3f} | Clusters: {num_clusters_dbscan}")

        # 4. Bisecting KMeans
        start = time.time()
        b_kmeans = BisectingKMeans(n_clusters=K, random_state=42)
        labels_b = b_kmeans.fit_predict(data)
        time_b = time.time() - start
        
        try:
            score_b = silhouette_score(data, labels_b, sample_size=10000, metric='cosine')
        except ValueError:
            score_b = -1
            
        print(f"  [Bisect KMeans]  Time: {time_b:.3f}s | Silhouette: {score_b:.3f} | Clusters: {K}")
        
        # 5. Gaussian Mixture Model
        start = time.time()
        gmm = GaussianMixture(n_components=K, random_state=42, covariance_type='diag')
        labels_gmm = gmm.fit_predict(data)
        time_gmm = time.time() - start
        
        try:
            score_gmm = silhouette_score(data, labels_gmm, sample_size=10000, metric='cosine')
        except ValueError:
            score_gmm = -1
            
        print(f"  [GMM]            Time: {time_gmm:.3f}s | Silhouette: {score_gmm:.3f} | Clusters: {K}")

if __name__ == "__main__":
    run_benchmarks()
