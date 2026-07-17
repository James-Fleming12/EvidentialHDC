from torchhd import functional
from torchhd import embeddings

import numpy as np
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

class Model(nn.Module):
    def __init__(self, ARCH, modeldir, hd_encoder, num_levels, randomness, num_classes, device):
        super(Model, self).__init__()

        self.device = device

        # Record the current number of class hypervectors
        self.num_classes = num_classes      # Used in supervised HD
        self.hd_dim = 10000
        self.temperature = 0.01

        self.flatten = torch.nn.Flatten()

        # set the input dimension
        self.input_dim = 128
        self.ARCH = ARCH

        with torch.no_grad():
            torch.nn.Module.dump_patches = True
            if self.ARCH["train"]["pipeline"] == "hardnet":
                from modules.network.HarDNet import HarDNet
                self.net = HarDNet(self.num_classes, self.ARCH["train"]["aux_loss"])

            if self.ARCH["train"]["pipeline"] == "res":
                from modules.network.ResNet import ResNet_34
                self.net = ResNet_34(self.num_classes, self.ARCH["train"]["aux_loss"])

                def convert_relu_to_softplus(model, act):
                    for child_name, child in model.named_children():
                        if isinstance(child, nn.LeakyReLU):
                            setattr(model, child_name, act)
                        else:
                            convert_relu_to_softplus(child, act)

                if self.ARCH["train"]["act"] == "Hardswish":
                    convert_relu_to_softplus(self.net, nn.Hardswish())
                elif self.ARCH["train"]["act"] == "SiLU":
                    convert_relu_to_softplus(self.net, nn.SiLU())

            if self.ARCH["train"]["pipeline"] == "fid":
                from modules.network.Fid import ResNet_34
                self.net = ResNet_34(self.parser.get_n_classes(), self.ARCH["train"]["aux_loss"])

                if self.ARCH["train"]["act"] == "Hardswish":
                    convert_relu_to_softplus(self.net, nn.Hardswish())
                elif self.ARCH["train"]["act"] == "SiLU":
                    convert_relu_to_softplus(self.net, nn.SiLU())
        w_dict = torch.load(modeldir + "/SENet_valid_best",
                            map_location=lambda storage, loc: storage)
        self.net.load_state_dict(w_dict['state_dict'], strict=True)
        self.net.eval()
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            self.gpu = True
            self.net.cuda()

        self.hd_encoder = hd_encoder
        if self.hd_encoder == 'rp':  # Random projection encoding
            # Generate a random projection matrix
            self.projection = embeddings.Projection(self.input_dim, self.hd_dim)

        elif self.hd_encoder == 'idlevel':  # ID-level encoding
            # Generate id-level value hv for each floating value
            self.value = embeddings.Level(num_levels, self.hd_dim, 
                                          randomness=randomness)
            print("self.value", self.value.weight.shape)  # cifar10: [100, 10000] # num_levels * hd_dim
            # Create a random hv for each position, for binding with the value hv
            self.position = embeddings.Random(self.input_dim, self.hd_dim)
            print("self.position", self.position.weight.shape)  # cifar10: [1280, 10000]  #bsz x num_features

        elif self.hd_encoder == 'nonlinear':  # Nonlinear encoding
            self.nonlinear_projection = embeddings.Sinusoid(self.input_dim, self.hd_dim)
        
        else:  # No encoder, use raw samples
            self.hd_dim = self.input_dim

        # Set classify
        self.classify = nn.Linear(self.hd_dim, self.num_classes, bias=False)
        self.classify_sample_cnt = torch.zeros((self.num_classes, 1)).to(self.device)

        self.classify.weight.data.fill_(0.0)

        # self.classify_weights is the sum of all hypervectors, so its scale
        # accounts the number of samples in this class/cluster
        self.classify_weights = nn.Parameter(self.classify.weight.data.clone()).to(device)
        # print(self.classify_weights.shape)  # size num_class x HD dim

    def encode(self, x, mask=None, PERCENTAGE=None, is_wrong=None):
        if mask is None:
            mask = torch.ones(self.hd_dim, device=self.device).type(torch.bool)
        # print("x.shape", x.shape)  # torch.Size([1, 5, 64, 512])

        with torch.cuda.amp.autocast(enabled=True):
            x = self.net(x, True)
        
        # print("x.shape", x.shape)  # torch.Size([1, 128, 64, 512])
        # x = self.flatten(x)
        x = x.permute(0, 2, 3, 1)  # shape: (1, 64, 512, 128)
        x = x.reshape(-1, 128)     # shape: (1*64*512, 128) = (32768, 128)
        # sample_hv = torch.zeros((x.shape[0], self.hd_dim), device=self.device)
        # print("x.shape", x.shape)  # torch.Size([32768, 128])
        if PERCENTAGE is not None:
            num_samples = int(x.shape[0] * PERCENTAGE)  # Calculate the number of samples to select
            
            if is_wrong is not None:
                # # Pick by the wrong and keep the PERCENTAGE
                wrong_indices = torch.nonzero(is_wrong, as_tuple=False).squeeze()
                
                if wrong_indices.numel() >= num_samples:
                    # If there are enough wrong samples, randomly select from them
                    selected_indices = wrong_indices[torch.randperm(wrong_indices.shape[0], device=x.device)[:num_samples]]
                    is_wrong[selected_indices] = False # Mark the selected indices as used
                else:
                    # If there are not enough wrong samples, fill the rest with random samples
                    non_wrong_indices = torch.nonzero(~is_wrong, as_tuple=False).squeeze()
                    remaining = num_samples - wrong_indices.numel()
                    fill_indices = non_wrong_indices[torch.randperm(non_wrong_indices.shape[0], device=x.device)[:remaining]]
    
                    selected_indices = torch.cat([wrong_indices, fill_indices], dim=0)
                    is_wrong[selected_indices] = False # Mark the selected indices as used
            else:
                selected_indices = torch.randperm(x.shape[0], device=x.device)[:num_samples]

            selected_indices, _ = selected_indices.sort()  # Optional: sort to preserve order
            # print("selected_indices", selected_indices.shape)  # e.g., torch.Size([1638])
            x = x[selected_indices]  # shape: (~PERCENTAGE * 32768, 128)
            assert x.shape[0] == num_samples, f"Expected {num_samples} samples, got {x.shape[0]}"

            # Pick by loss: 
            # num_samples = int(x.shape[0] * PERCENTAGE)
            # num_wrongdata = 0
            # sorted_loss, sorted_indices = torch.sort(is_wrong, descending=True)
            # top_indices = sorted_indices[:num_wrongdata]

            # all_indices = torch.arange(is_wrong.shape[0], device=x.device)
            # temp = torch.ones_like(is_wrong, dtype=torch.bool)
            # temp[top_indices] = False
            # remaining_indices = all_indices[temp]

            # remaining = num_samples - num_wrongdata
            # if remaining_indices.numel() >= remaining:
            #     random_fill_indices = remaining_indices[torch.randperm(remaining_indices.shape[0])[:remaining]]
            # else:
            #     # If not enough remaining, take all of them
            #     random_fill_indices = remaining_indices
            
            # selected_indices = torch.cat([top_indices, random_fill_indices], dim=0)
            # is_wrong[selected_indices] = 0 # Mark the selected indices as used

            # Get top losses and their indices (descending sort)
            # sorted_loss, sorted_indices = torch.sort(is_wrong, descending=True)
            # selected_indices = sorted_indices[:num_samples]  # pick top N
            # is_wrong[selected_indices] = 0.0

            # Filter your data
            # x = x[selected_indices]
            # print("x after selection", x.shape)  # e.g., torch.Size([1638, 128])
            # print("x", x[0])  # e.g., torch.Size([1638])

        else:
            selected_indices = torch.arange(x.shape[0], device=x.device)  # use all data
        sample_hv = torch.zeros((x.shape[0], self.hd_dim), device=self.device, dtype=x.dtype)

        if self.hd_encoder == 'rp':
            if x.dtype != self.projection.weight.dtype:
                self.projection = self.projection.to(x.dtype).to(self.device)
            sample_hv[:, mask] = self.projection(x)[:, mask]

        elif self.hd_encoder == 'idlevel':
            # print("Encode bind value: ", self.value(x)[:, :, mask].shape)  # btz*size x num_features * hd_dim
            # print("Encode position value: ", self.position.weight[:, mask].shape)  # num_features * hd_dim
            tmp_hv = functional.bind(self.position.weight[:, mask],
                                     self.value(x)[:, :, mask])  # bsz*size x num_features x hd_dim
            sample_hv[:, mask] = functional.multiset(tmp_hv)  # bsz*size x hd_dim

        elif self.hd_encoder == 'nonlinear':
            sample_hv[:, mask] = self.nonlinear_projection(x)[:, mask]
        else:  # None encoder, just use the raw sample
            return x

        sample_hv[:, mask] = functional.hard_quantize(sample_hv[:, mask])
        # print("sample_hv.shape", sample_hv.shape)  # (bsz*size, 1000)
        return sample_hv, selected_indices, is_wrong

    def forward(self, x, mask=None, PERCENTAGE=None, is_wrong=None):
        if mask is None:
            mask = torch.ones(self.hd_dim, device=self.device).type(torch.bool)

        # Get logits output
        enc, indices, is_wrong_left = self.encode(x, mask, PERCENTAGE, is_wrong)
        # Compute the cosine distance between normalized hypervectors
        if enc.dtype != self.classify.weight.dtype:
            self.classify = self.classify.to(enc.dtype)
        logits = self.classify(F.normalize(enc, dim=1))

        #logits = torch.div(logits, self.temperature)
        #softmax_logits = F.log_softmax(logits, dim=1)

        return logits, F.normalize(enc, dim=1), indices, is_wrong_left # enc is still hd_dim, but some elements are 0

    def get_predictions(self, enc):
        # Compute the cosine distance between normalized hypervectors
        if enc.dtype != self.classify.weight.dtype:
            self.classify = self.classify.to(enc.dtype)
        logits = self.classify(F.normalize(enc, dim=1))
        return logits

    def extract_class_hv(self, mask=None):
        if mask is None:
            mask = torch.ones(self.hd_dim, device=self.device).type(torch.bool)

        if self.method == 'LifeHD':
            class_hv = self.classify.weight[:self.cur_classes, mask]
        else:  # self.method == 'BasicHD'
            #class_hv = self.classify_weights / self.classify_sample_cnt
            class_hv = self.classify.weight[:, mask]
        return class_hv.detach().cpu().numpy()
    
    def extract_pair_simil(self, mask=None):
        if mask is None:
            mask = torch.ones(self.hd_dim, device=self.device).type(torch.bool)

        if self.method == 'LifeHD' or self.method == 'LifeHDsemi':
            class_hv = self.classify.weight[:self.cur_classes, mask]
        elif self.method == 'BasicHD':
            class_hv = self.classify.weight[:, mask]
        else:
            raise ValueError('method not supported: {}'.format(self.method))
        pair_simil = class_hv @ class_hv.T

        if self.method == 'LifeHDsemi':
            pair_simil[:self.num_classes, :self.num_classes] = torch.eye(self.num_classes)
        return pair_simil.detach().cpu().numpy(), class_hv.detach().cpu().numpy()

def set_model(ARCH, modeldir, hd_encoder, num_levels, randomness, num_classes, device):
    return Model(ARCH, modeldir, hd_encoder, num_levels, randomness, num_classes, device)

class UQModel(nn.Module):
    def __init__(self, ARCH, modeldir, hd_encoder, num_levels, randomness, num_classes, device, gauss_rp=True, use_adaptor=True):
        super(UQModel, self).__init__()

        self.device = device
        self.use_adaptor = use_adaptor

        self.num_classes = num_classes
        self.hd_dim = 10000
        self.temperature = 0.01

        self.flatten = torch.nn.Flatten()

        self.input_dim = 128
        self.ARCH = ARCH

        with torch.no_grad():
            torch.nn.Module.dump_patches = True
            if self.ARCH["train"]["pipeline"] == "hardnet":
                from modules.network.HarDNet import HarDNet
                self.net = HarDNet(self.num_classes, self.ARCH["train"]["aux_loss"])

            if self.ARCH["train"]["pipeline"] == "res":
                from modules.network.ResNet import ResNet_34
                self.net = ResNet_34(self.num_classes, self.ARCH["train"]["aux_loss"], use_adaptor=self.use_adaptor)

                def convert_relu_to_softplus(model, act):
                    for child_name, child in model.named_children():
                        if isinstance(child, nn.LeakyReLU):
                            setattr(model, child_name, act)
                        else:
                            convert_relu_to_softplus(child, act)

                if self.ARCH["train"]["act"] == "Hardswish":
                    convert_relu_to_softplus(self.net, nn.Hardswish())
                elif self.ARCH["train"]["act"] == "SiLU":
                    convert_relu_to_softplus(self.net, nn.SiLU())

            if self.ARCH["train"]["pipeline"] == "fid":
                from modules.network.Fid import ResNet_34
                self.net = ResNet_34(self.num_classes, self.ARCH["train"]["aux_loss"])

                if self.ARCH["train"]["act"] == "Hardswish":
                    convert_relu_to_softplus(self.net, nn.Hardswish())
                elif self.ARCH["train"]["act"] == "SiLU":
                    convert_relu_to_softplus(self.net, nn.SiLU())
            
            if self.ARCH["train"]["pipeline"] == "pointpillar":
                from modules.HDC_cl import PointPillarEncoder

                class _PointPillarEncoder4D(PointPillarEncoder):
                    def forward(self, batch, only_feat=False):
                        return super().forward(batch).unsqueeze(-1).unsqueeze(-1)

                self.net = _PointPillarEncoder4D(
                    in_channels=self.ARCH["train"].get("pointpillar_in_channels", 4),
                    bev_shape=tuple(self.ARCH["train"].get("pointpillar_bev_shape", [512, 512])),
                )

        if self.ARCH["train"]["pipeline"] != "pointpillar":
            w_dict = torch.load(modeldir + "/SENet_valid_best", map_location=lambda storage, loc: storage)
            
            state_dict = w_dict['state_dict']
            model_state = self.net.state_dict()
            for k in list(state_dict.keys()):
                if k in model_state and state_dict[k].shape != model_state[k].shape:
                    del state_dict[k]
                    
            self.net.load_state_dict(state_dict, strict=False)
            self.net.eval()
            if torch.cuda.is_available() and torch.cuda.device_count() > 0:
                self.gpu = True
                self.net.cuda()
        self.hd_encoder = hd_encoder
        if self.hd_encoder == 'rp':  # Random projection encoding
            torch_rng_state = torch.get_rng_state()
            numpy_rng_state = np.random.get_state()
            if torch.cuda.is_available():
                cuda_rng_state = torch.cuda.get_rng_state()

            torch.manual_seed(42) # setting fixed seed for projection initialization (removes saved model randomness)
            np.random.seed(42)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(42)
                torch.cuda.manual_seed_all(42)

            if not gauss_rp:
                # self.projection = embeddings.Projection(self.input_dim, self.hd_dim)

                self.projection = nn.Linear(self.input_dim, self.hd_dim, bias=False)
                with torch.no_grad():
                    gaussian_matrix = torch.randn(self.hd_dim, self.input_dim) 
                    self.projection.weight.copy_(gaussian_matrix / np.sqrt(self.input_dim))
            else:
                self.projection = nn.Linear(self.input_dim, self.hd_dim, bias=False)
                with torch.no_grad():
                    gaussian_matrix = torch.randn(self.hd_dim, self.input_dim)
                    q, _ = torch.linalg.qr(gaussian_matrix)
                    self.projection.weight.copy_(q * torch.sqrt(torch.tensor(self.hd_dim))) # Scale by the square root of the dimension to preserve variance (Johnson-Lindenstrauss)

            torch.set_rng_state(torch_rng_state) # set back to random
            np.random.set_state(numpy_rng_state)
            if torch.cuda.is_available():
                torch.cuda.set_rng_state(cuda_rng_state)

        elif self.hd_encoder == 'idlevel':  # ID-level encoding
            # Generate id-level value hv for each floating value
            self.value = embeddings.Level(num_levels, self.hd_dim,  randomness=randomness)
            print("self.value", self.value.weight.shape)  # cifar10: [100, 10000] # num_levels * hd_dim
            # Create a random hv for each position, for binding with the value hv
            self.position = embeddings.Random(self.input_dim, self.hd_dim)
            print("self.position", self.position.weight.shape)  # cifar10: [1280, 10000]  #bsz x num_features

        elif self.hd_encoder == 'nonlinear':  # Nonlinear encoding
            self.nonlinear_projection = embeddings.Sinusoid(self.input_dim, self.hd_dim)
        else:
            self.hd_dim = self.input_dim

        self.classify = nn.Linear(self.hd_dim, self.num_classes, bias=False)
        self.classify_sample_cnt = torch.zeros((self.num_classes, 1)).to(self.device)

        self.classify.weight.data.fill_(0.0)

        self.classify_weights = nn.Parameter(self.classify.weight.data.clone()).to(device)
        self.gauss_rp = gauss_rp

        self.register_buffer('proto_momentum', torch.zeros_like(self.classify.weight.data)) # EMA momentum

    def encode(self, x, mask=None, PERCENTAGE=None, is_wrong=None, chunk_idx=None):
        if mask is None:
            mask = torch.ones(self.hd_dim, device=self.device).type(torch.bool)

        with torch.amp.autocast('cuda', enabled=True):
            x = self.net(x, only_feat=True)

        x = x.permute(0, 2, 3, 1)
        x = x.reshape(-1, 128)

        if chunk_idx is not None:
            start, end = chunk_idx
            x = x[start:end]

        if PERCENTAGE is not None:
            wrong_indices = torch.nonzero(is_wrong, as_tuple=False).squeeze()
            num_samples = int(x.shape[0] * PERCENTAGE)  # Calculate the number of samples to select

            if wrong_indices.numel() >= num_samples:
                selected_indices = wrong_indices[torch.randperm(wrong_indices.shape[0], device=x.device)[:num_samples]]
                is_wrong[selected_indices] = False
            else:
                non_wrong_indices = torch.nonzero(~is_wrong, as_tuple=False).squeeze()
                remaining = num_samples - wrong_indices.numel()
                fill_indices = non_wrong_indices[torch.randperm(non_wrong_indices.shape[0], device=x.device)[:remaining]]

                selected_indices = torch.cat([wrong_indices, fill_indices], dim=0)
                is_wrong[selected_indices] = False

            selected_indices, _ = selected_indices.sort()
            x = x[selected_indices]
            assert x.shape[0] == num_samples, f"Expected {num_samples} samples, got {x.shape[0]}"
        else:
            selected_indices = torch.arange(x.shape[0], device=x.device)  # use all data
        sample_hv = torch.zeros((x.shape[0], self.hd_dim), device=self.device, dtype=x.dtype)

        if self.hd_encoder == 'rp':
            if x.dtype != self.projection.weight.dtype:
                self.projection = self.projection.to(x.dtype).to(self.device)
            sample_hv[:, mask] = self.projection(x)[:, mask]

        elif self.hd_encoder == 'idlevel':
            tmp_hv = functional.bind(self.position.weight[:, mask],
                                     self.value(x)[:, :, mask])
            sample_hv[:, mask] = functional.multiset(tmp_hv)

        elif self.hd_encoder == 'nonlinear':
            sample_hv[:, mask] = self.nonlinear_projection(x)[:, mask]
        else:
            return x

        sample_hv[:, mask] = functional.hard_quantize(sample_hv[:, mask])
        return sample_hv, selected_indices, is_wrong

    def forward(self, x, mask=None, PERCENTAGE=None, is_wrong=None):
        if mask is None:
            mask = torch.ones(self.hd_dim, device=self.device).type(torch.bool)

        enc, indices, is_wrong_left = self.encode(x, mask, PERCENTAGE, is_wrong)
        if enc.dtype != self.classify.weight.dtype:
            self.classify = self.classify.to(enc.dtype)
        logits = self.classify(F.normalize(enc, dim=1))

        return logits, F.normalize(enc, dim=1), indices, is_wrong_left

    def get_predictions(self, enc):
        if enc.dtype != self.classify.weight.dtype:
            self.classify = self.classify.to(enc.dtype)
        logits = self.classify(F.normalize(enc, dim=1))
        return logits

    def extract_class_hv(self, mask=None):
        if mask is None:
            mask = torch.ones(self.hd_dim, device=self.device).type(torch.bool)

        if self.method == 'LifeHD':
            class_hv = self.classify.weight[:self.cur_classes, mask]
        else:
            class_hv = self.classify.weight[:, mask]
        return class_hv.detach().cpu().numpy()
    
    def extract_pair_simil(self, mask=None):
        if mask is None:
            mask = torch.ones(self.hd_dim, device=self.device).type(torch.bool)

        if self.method == 'LifeHD' or self.method == 'LifeHDsemi':
            class_hv = self.classify.weight[:self.cur_classes, mask]
        elif self.method == 'BasicHD':
            class_hv = self.classify.weight[:, mask]
        else:
            raise ValueError('method not supported: {}'.format(self.method))
        pair_simil = class_hv @ class_hv.T

        if self.method == 'LifeHDsemi':
            pair_simil[:self.num_classes, :self.num_classes] = torch.eye(self.num_classes)
        return pair_simil.detach().cpu().numpy(), class_hv.detach().cpu().numpy()
    
    @torch.no_grad()
    def get_confidence(self, enc, preds=None, method='hybrid'):
        """
        Master method to compute the confidence score for gating.
        Routes to the appropriate underlying uncertainty methods based on the requested string.
        """
        pass

    @torch.no_grad()
    def online_update(self, x, proj_xyz=None, update_method='hybrid_balanced'):
        """
        The primary entrypoint for test-time adaptation. 
        Calls `get_confidence`, applies thresholds, consults the subcluster ledger (if balanced), 
        and updates the prototype weights `self.classify.weight`.
        """
        pass

    @torch.no_grad()
    def _get_epistemic_uncertainty(self, x, enc):
        """
        Pillar 1(b): Network Uncertainty.
        Estimates uncertainty before the softmax.
        Candidates to implement here:
        - Multi-RP ensemble (multiple random projections)
        - Feature-space density
        - Evidential deep learning mass
        
        Returns a score in [0, 1] where 1 is highly reliable.
        """
        pass

    @torch.no_grad()
    def _get_spatial_consistency(self, enc, preds, proj_xyz):
        """
        Pillar 1(c): Spatial/Temporal Consistency.
        Uses `proj_xyz` (3D coordinates) to check if a point's predicted label 
        agrees with its physical neighbors. Acts as a hard veto against fog/noise artifacts.
        
        Returns a binary mask or soft score in [0, 1].
        """
        pass

    @torch.no_grad()
    def _get_geometric_confidence(self, enc, preds):
        """
        Pillar 1(a): HD Space Geometry.
        The baseline standard prototype cosine similarity.
        
        Returns similarity score in [-1, 1].
        """
        pass

    @torch.no_grad()
    def _fuse_uncertainties(self, epistemic, consistency, geometric):
        """
        Combines the independent uncertainty scores into a single gating metric.
        Must be calibrated so that a failure in one source (e.g. geometric drift) 
        can be overridden by another (e.g. epistemic rejection).
        """
        pass

    @torch.no_grad()
    def _initialize_subcluster_ledger(self, num_clusters=10):
        """
        Initializes K subclusters per class using KMeans on source-domain density.
        Initializes a counter array `self.subcluster_update_counts = zeros(NUM_CLASSES, K)`.
        """
        from sklearn.cluster import KMeans
        import numpy as np
        
        self.num_clusters = num_clusters
        self.subcluster_centroids = torch.zeros(self.num_classes, num_clusters, 128, device=self.device)
        self.subcluster_update_counts = torch.zeros(self.num_classes, num_clusters, device=self.device)
        self.actual_k_per_class = torch.zeros(self.num_classes, dtype=torch.long, device=self.device)
        
        # Check if latents were collected
        if not hasattr(self, 'class_latents_for_clustering'):
            raise RuntimeError("_initialize_subcluster_ledger requires self.class_latents_for_clustering to be populated")
            
        # Find the minimum number of samples across all valid classes
        class_sizes = []
        for c in range(self.num_classes):
            if len(self.class_latents_for_clustering[c]) > 0:
                size = sum(x.shape[0] for x in self.class_latents_for_clustering[c])
                class_sizes.append(size)
                
        if len(class_sizes) == 0:
            return
            
        # Target samples per class is the minimum size across all classes
        target_samples = min(class_sizes)
        print(f"Downsampling all classes to exactly {target_samples} samples for KMeans...")
            
        for c in range(self.num_classes):
            if len(self.class_latents_for_clustering[c]) == 0:
                continue
                
            data = torch.cat(self.class_latents_for_clustering[c], dim=0).numpy()
            
            # Uniformly downsample to target_samples
            if data.shape[0] > target_samples:
                indices = np.random.choice(data.shape[0], target_samples, replace=False)
                data = data[indices]
            
            # If a class has fewer points than K, reduce K for that class
            K = min(num_clusters, data.shape[0])
            if K == 0: continue
            self.actual_k_per_class[c] = K
            
            kmeans = KMeans(n_clusters=K, n_init='auto', random_state=42)
            kmeans.fit(data)
            
            centroids = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32, device=self.device)
            # Normalize centroids back to hypersphere
            centroids = torch.nn.functional.normalize(centroids, dim=1)
            
            self.subcluster_centroids[c, :K] = centroids
    @torch.no_grad()
    def _consult_budget_ledger(self, latent_x_valid, preds, update_weights, budget_margin=50):
        """
        Takes the continuous soft weights (`update_weights`) of candidate points.
        1. Maps each candidate point (latent_x_valid) to its nearest subcluster centroid for its predicted class.
        2. Checks `self.subcluster_update_counts`. If a subcluster has exceeded its 
           relative budget (e.g., it has N more updates than the minimum sibling), 
           its update weight is zeroed out.
        3. Returns refined weights where saturated points are dropped.
        4. Increments the ledger counts for the points that are ultimately admitted.
        """
        if latent_x_valid.shape[0] == 0:
            return update_weights
            
        latent_x_norm = torch.nn.functional.normalize(latent_x_valid, dim=1)
        
        # We need to find the closest subcluster centroid for each point based on its predicted class
        # preds is shape (N,)
        refined_weights = update_weights.clone()
        
        # We process class by class to avoid massive broadcasting
        unique_classes = torch.unique(preds)
        for c in unique_classes:
            c_mask = (preds == c)
            if not c_mask.any(): continue
            
            K = self.actual_k_per_class[c].item()
            if K == 0:
                # No subclusters were initialized for this class (very rare), just allow all updates
                continue
                
            # shape: (N_c, 128)
            pts = latent_x_norm[c_mask]
            
            # shape: (K, 128)
            centroids = self.subcluster_centroids[c, :K]
            
            # cosine similarity: (N_c, K)
            sims = torch.matmul(pts, centroids.T)
            
            # nearest subcluster indices: (N_c,)
            nearest_subclusters = torch.argmax(sims, dim=1)
            
            # Check budgets
            # We want to freeze subclusters that have significantly more updates than the least-updated subcluster
            current_counts = self.subcluster_update_counts[c, :K]
            min_count = current_counts.min()
            
            # For each point, check if its subcluster is saturated
            subcluster_counts_for_pts = current_counts[nearest_subclusters]
            
            # Saturated if it exceeds the minimum count by more than the budget_margin
            saturated_mask = (subcluster_counts_for_pts - min_count) > budget_margin
            
            # Zero out the weights for saturated points
            refined_weights[c_mask] = refined_weights[c_mask] * (~saturated_mask).float()
            
            # Update the ledger for points that were NOT dropped (i.e., non-zero weights)
            # We count an update if the refined weight > 0
            admitted_pts_mask = (refined_weights[c_mask] > 0)
            if admitted_pts_mask.any():
                admitted_subclusters = nearest_subclusters[admitted_pts_mask]
                # Increment counts using bincount
                updates = torch.bincount(admitted_subclusters, minlength=K)
                self.subcluster_update_counts[c, :K] += updates

        return refined_weights

def set_uq_model(ARCH, modeldir, hd_encoder, num_levels, randomness, num_classes, device, subcluster_type='bipolar'):
    return UQModel(ARCH, modeldir, hd_encoder, num_levels, randomness, num_classes, device)