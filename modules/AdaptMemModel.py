import torch
import torch.nn as nn
import torch.nn.functional as F

class AdaptiveMemoryBank(nn.Module):
    """
    Adaptive Memory Bank for Test-Time Adaptation operating natively in the HDC space.
    Uses a global un-partitioned memory bank with Reservoir Sampling and Density-Adaptive k-NN.
    """
    def __init__(self, hd_dim=10000, num_classes=17, memory_capacity=10000, k=10, purity_threshold=0.8):
        super().__init__()
        self.hd_dim = hd_dim
        self.num_classes = num_classes
        self.k = k
        self.purity_threshold = purity_threshold
        self.capacity = memory_capacity
        
        # Global unpartitioned memory bank
        self.register_buffer("keys", torch.empty((self.capacity, hd_dim), dtype=torch.float32))
        self.register_buffer("values", torch.zeros(self.capacity, dtype=torch.int64))
        self.register_buffer("is_valid", torch.zeros(self.capacity, dtype=torch.bool))
        self.register_buffer("ptr", torch.tensor(0, dtype=torch.int64))
        self.register_buffer("reserved_slots", torch.tensor(0, dtype=torch.int64))
        
    def query(self, features):
        """
        Query via Density-Adaptive Hamming Distance against the global memory bank.
        """
        valid_mask = self.is_valid
        if not valid_mask.any():
            return torch.zeros(features.size(0), dtype=torch.int64, device=features.device), \
                   torch.zeros(features.size(0), dtype=torch.float32, device=features.device)
                   
        flat_keys = self.keys[valid_mask]
        flat_values = self.values[valid_mask]
        
        # Calculate Internal Density (Class Frequency in the global reservoir)
        class_counts = torch.bincount(flat_values, minlength=self.num_classes).float()
        
        # Binarize incoming query
        bin_features = torch.sign(features).to(torch.float32)
        bin_features[bin_features == 0] = 1.0 
        
        k = min(self.k, flat_keys.size(0))
        predictions = []
        purity = []
        
        # Adaptive Metric tuning parameter (Margin Penalty)
        # Using Extreme Value Theory (EVT) for the max of N Gaussian variables.
        # The dot product of two random 10,000D bipolar vectors is Gaussian with sigma = 100.
        # The expected maximum of N such variables grows as sigma * sqrt(2 * ln(N)).
        # We subtract this exact statistical advantage to make the similarity density-blind.
        target_penalties = 100.0 * torch.sqrt(2.0 * torch.log(class_counts + 1.0))
        target_penalties = target_penalties[flat_values]
        
        # Process in chunks using float16 for massive speedup
        chunk_size = 50000
        for i in range(0, bin_features.size(0), chunk_size):
            chunk = bin_features[i:i+chunk_size]
            sims = torch.mm(chunk.half(), flat_keys.t().half()).float()
            
            adaptive_sims = sims - target_penalties.unsqueeze(0)
            
            topk_sims, topk_idx = adaptive_sims.topk(k=k, dim=1)
            neighbor_sem = flat_values[topk_idx]
            
            pred_chunk = torch.mode(neighbor_sem, dim=1).values
            predictions.append(pred_chunk)
            # Calculate Relative Manifold Cohesion natively in 10,000D space
            # S is the sum vector of the k nearest neighbors
            S = torch.zeros((chunk.size(0), self.hd_dim), device=chunk.device, dtype=torch.float32)
            for j in range(k):
                S += flat_keys[topk_idx[:, j]]
                
            # Average internal neighbor distance
            internal_sim_sum = (S * S).sum(dim=1)
            avg_internal_sim = (internal_sim_sum - k * 10000.0) / (k * (k - 1) + 1e-8)
            D_int = 1.0 - (avg_internal_sim / 10000.0)
            
            # Structural Variance Prior (prevent division by zero from exact duplicates)
            # A typical clean cluster in HDC space has ~0.05 variance (cos sim 0.9).
            D_int_clamped = torch.clamp(D_int, min=0.05)
            
            # Average query-to-neighbor distance
            q_sim = (chunk.float() * S).sum(dim=1)
            D_q = 1.0 - (q_sim / (k * 10000.0))
            
            if i == 0:
                print(f"DEBUG: D_q mean = {D_q.mean().item():.4f}, D_int mean = {D_int.mean().item():.4f}, D_int_clamped mean = {D_int_clamped.mean().item():.4f}, Ratio mean = {(D_q / (D_int_clamped + 1e-8)).mean().item():.4f}")
            
            # Cohesion Ratio (D_q / D_int)
            cohesion_ratio = D_q / (D_int_clamped + 1e-8)
            
            # Convert to confidence score (1.0 = perfect fit, < 0.8 = outlier)
            cohesion_conf = 1.0 / (cohesion_ratio + 1e-8)
            purity.append(cohesion_conf)
            
        predictions = torch.cat(predictions, dim=0)
        purity = torch.cat(purity, dim=0)
        
        return predictions, purity

    def update(self, features, pseudo_labels, confidence, true_labels=None):
        """
        Graph-Pruned Update with Reservoir Sampling (Fixed Probability Replacement).
        """
        valid_mask = confidence >= self.purity_threshold
        admission_rate = valid_mask.float().mean().item()
        
        if not valid_mask.any():
            return admission_rate, -1.0
            
        valid_features = features[valid_mask]
        valid_labels = pseudo_labels[valid_mask]
        
        # Binarize incoming features BEFORE storing them, otherwise we corrupt the Float16 dot products
        bin_features = torch.sign(valid_features).to(torch.float32)
        bin_features[bin_features == 0] = 1.0
        
        # Track memory bank semantic purity (diagnostic only)
        purity_err = -1.0
        if true_labels is not None:
            valid_true = true_labels[valid_mask]
            # Ignore ignore_index (-1 or 255) in purity calculation
            eval_mask = (valid_true >= 0) & (valid_true < self.num_classes)
            if eval_mask.any():
                incorrect = (valid_labels[eval_mask] != valid_true[eval_mask]).float().sum()
                purity_err = (incorrect / eval_mask.float().sum()).item()
        
        n_in = bin_features.size(0)
        ptr = self.ptr.item()
        cap = self.capacity
        
        # If queue is not yet full, fill it sequentially
        if not self.is_valid.all():
            available = cap - ptr
            if n_in <= available:
                self.keys[ptr:ptr+n_in] = bin_features
                self.values[ptr:ptr+n_in] = valid_labels
                self.is_valid[ptr:ptr+n_in] = True
                self.ptr.fill_((ptr + n_in) % cap)
            else:
                self.keys[ptr:] = bin_features[:available]
                self.values[ptr:] = valid_labels[:available]
                self.is_valid[ptr:] = True
                
                # The remainder undergo Reservoir Sampling in the non-reserved space
                remainder_features = bin_features[available:]
                remainder_labels = valid_labels[available:]
                
                # Replace with probability P = 0.01 (Slower Momentum-based write policy to prevent rapid flushing)
                prob_mask = torch.rand(remainder_features.size(0), device=features.device) < 0.01
                replace_features = remainder_features[prob_mask]
                replace_labels = remainder_labels[prob_mask]
                
                if replace_features.size(0) > 0:
                    replace_idx = torch.randint(self.reserved_slots.item(), cap, (replace_features.size(0),), device=features.device)
                    self.keys[replace_idx] = replace_features
                    self.values[replace_idx] = replace_labels
                    
                self.ptr.fill_(0) # Pointer meaning is lost after full, but we keep it at 0
        else:
            # Queue is full, use global Reservoir Sampling replacement in non-reserved space
            # Using P = 0.01 to prevent rapid flushing of the memory bank
            prob_mask = torch.rand(n_in, device=features.device) < 0.01
            replace_features = bin_features[prob_mask]
            replace_labels = valid_labels[prob_mask]
            
            if replace_features.size(0) > 0:
                replace_idx = torch.randint(self.reserved_slots.item(), cap, (replace_features.size(0),), device=features.device)
                self.keys[replace_idx] = replace_features
                self.values[replace_idx] = replace_labels
                
        return admission_rate, purity_err
        
    def recover_geometry(self, features):
        pass

    def forward(self, features):
        pass
