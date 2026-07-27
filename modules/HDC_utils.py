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

class DualGateModel(nn.Module):
    def __init__(self, ARCH, modeldir, hd_encoder, num_levels, randomness, num_classes, device, gauss_rp=True, use_adaptor=True):
        super(DualGateModel, self).__init__()

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
    def get_confidence(self, enc, preds=None, method='soft_dual_weight', uncertainty=None, z_score=None, logits=None, active_mu_cos=None, active_sigma_cos=None, dynamic_geom=True, view_preds=None, view_var=None, **kwargs):
        """
        Master method to compute the confidence score for gating.
        Returns (base_weights, uncertainty, z_score).
        """
        if uncertainty is None:
            uncertainty = self._get_epistemic_uncertainty(enc, logits=logits, active_mu_cos=active_mu_cos, active_sigma_cos=active_sigma_cos)
        if z_score is None and preds is not None:
            z_score = self._get_geometric_confidence(enc, preds, dynamic_geom=dynamic_geom)
        elif z_score is None:
            z_score = torch.zeros_like(uncertainty)
            
        base_weights = self._fuse_uncertainties(uncertainty, None, z_score, method=method)
        return base_weights, uncertainty, z_score

    @torch.no_grad()
    def online_update(self, enc, preds, update_weights, update_method='bm_ic4', ic_method='ic4', uncertainty=None, update_lr=0.005, normalize_weights=False, view_preds=None, **kwargs):
        """
        The primary entrypoint for test-time adaptation. 
        Applies active learning multipliers (IC4), filters candidate points,
        and updates prototype weights using Bayesian Momentum.
        """
        if ic_method == 'ic4' and uncertainty is not None:
            update_weights = update_weights * uncertainty
            
        fired_mask = update_weights > 0
        if not fired_mask.any():
            return fired_mask
            
        valid_enc = enc[fired_mask]
        labels_fired = preds[fired_mask]
        weights_fired = update_weights[fired_mask]
        
        sums_c = torch.zeros((self.num_classes, valid_enc.shape[1]), dtype=valid_enc.dtype, device=self.device)
        weighted_enc = (valid_enc * weights_fired.unsqueeze(1)).to(valid_enc.dtype)
        sums_c.index_add_(0, labels_fired, weighted_enc)
        
        counts_c = torch.bincount(labels_fired, minlength=self.num_classes).float()
        weights_sum_c = torch.bincount(labels_fired, weights=weights_fired.float(), minlength=self.num_classes)
        
        valid_c = (counts_c > 0)
        if valid_c.any():
            c_update_norm = torch.nn.functional.normalize(sums_c, p=2, dim=1)
            step_mags = update_lr * (weights_sum_c / torch.clamp(counts_c, min=1.0))
            
            update_delta = torch.where(valid_c.unsqueeze(1), step_mags.unsqueeze(1) * c_update_norm.to(self.classify.weight.dtype), torch.zeros_like(self.classify.weight))
            self.classify.weight.data += update_delta
            if normalize_weights:
                self.classify.weight.data = torch.nn.functional.normalize(self.classify.weight.data, p=2, dim=1)
                
            if not hasattr(self, 'class_update_counts'):
                self.class_update_counts = torch.zeros(self.num_classes, device=self.device)
            self.class_update_counts += valid_c.long()
            if not hasattr(self, '_update_magnitude_log'):
                self._update_magnitude_log = []
            self._update_magnitude_log.extend(step_mags[valid_c].detach().cpu().tolist())
            
        return fired_mask

    @torch.no_grad()
    def _get_epistemic_uncertainty(self, enc, logits=None, active_mu_cos=None, active_sigma_cos=None):
        """
        Pillar 1(b): Network Uncertainty (Dirichlet Evidence Decay).
        Computes epistemic certainty in [0, 1] derived from Dirichlet evidential density.
        """
        if active_mu_cos is None:
            active_mu_cos = getattr(self, 'source_mu_cos', None)
        if active_sigma_cos is None:
            active_sigma_cos = getattr(self, 'source_sigma_cos', None)
            
        if active_mu_cos is not None and active_sigma_cos is not None:
            if logits is None:
                norm_enc = torch.nn.functional.normalize(enc.float(), dim=1)
                norm_weights = torch.nn.functional.normalize(self.classify.weight.float(), dim=1)
                logits = torch.nn.functional.linear(norm_enc, norm_weights)
            z = (logits - active_mu_cos) / (active_sigma_cos + 1e-8)
            ev = torch.nn.functional.softplus(5.0 * z)
            S_val = torch.sum(ev + 1.0, dim=1)
            return self.num_classes / S_val
        elif logits is not None:
            probs = torch.softmax(logits, dim=1)
            conf, _ = torch.max(probs, dim=1)
            return conf
        return torch.ones(enc.shape[0], dtype=torch.float32, device=self.device)

    @torch.no_grad()
    def _get_spatial_consistency(self, enc, preds, proj_xyz=None, **kwargs):
        """
        Pillar 1(c): Spatial/Temporal Consistency.
        Returns a binary mask or soft score in [0, 1].
        """
        return torch.ones_like(preds, dtype=torch.bool)

    @torch.no_grad()
    def _get_geometric_confidence(self, enc, preds, dynamic_geom=True):
        """
        Pillar 1(a): HD Space Geometry (Mahalanobis Z-Score Density).
        Measures physical dispersion on the 128D hypersphere using uncentred Euclidean distance Z-score.
        """
        if hasattr(self, 'class_latent_means') and self.class_latent_means is not None and hasattr(self, 'source_density_std') and self.source_density_std is not None:
            if not hasattr(self, 'source_density_mean') or self.source_density_mean is None:
                raise ValueError("CRITICAL ERROR: self.source_density_mean is missing or None! Stale checkpoint or cache would cause uncentred 128D geometric distance saturation.")
            pred_means = self.class_latent_means[preds]
            dist = torch.norm(enc.float() - pred_means.float(), p=2, dim=1)
            
            if dynamic_geom:
                if not hasattr(self, 'running_density_mean') or self.running_density_mean is None:
                    self.running_density_mean = self.source_density_mean.clone().to(self.device)
                if not hasattr(self, 'running_density_std') or self.running_density_std is None:
                    self.running_density_std = self.source_density_std.clone().to(self.device)
                self.running_density_mean = self.running_density_mean.to(self.device)
                self.running_density_std = self.running_density_std.to(self.device)
                
                counts = torch.bincount(preds, minlength=self.num_classes).float()
                sum_dist = torch.bincount(preds, weights=dist, minlength=self.num_classes)
                sum_sq_dist = torch.bincount(preds, weights=dist**2, minlength=self.num_classes)
                
                valid_c = (counts > 1)
                batch_means = torch.where(valid_c, sum_dist / torch.clamp(counts, min=1.0), torch.zeros_like(sum_dist))
                batch_vars = torch.where(valid_c, (sum_sq_dist - sum_dist**2 / torch.clamp(counts, min=1.0)) / torch.clamp(counts - 1.0, min=1.0), torch.zeros_like(sum_sq_dist))
                batch_stds = torch.sqrt(torch.clamp(batch_vars, min=0.0))
                
                self.running_density_mean = torch.where(valid_c, 0.95 * self.running_density_mean + 0.05 * batch_means, self.running_density_mean)
                valid_std = valid_c & (batch_stds > 0)
                self.running_density_std = torch.where(valid_std, 0.95 * self.running_density_std + 0.05 * batch_stds, self.running_density_std)
                
                mean_c = self.running_density_mean[preds]
                std_c = self.running_density_std[preds] + 1e-8
            else:
                self.source_density_mean = self.source_density_mean.to(self.device)
                self.source_density_std = self.source_density_std.to(self.device)
                mean_c = self.source_density_mean[preds]
                std_c = self.source_density_std[preds] + 1e-8
                
            return (dist - mean_c) / std_c
        else:
            raise ValueError("CRITICAL ERROR: Geometric confidence (Mahalanobis Z-Score) requires self.class_latent_means and self.source_density_std to be populated from source pretraining statistics! Cannot compute Z-score.")

    @torch.no_grad()
    def _fuse_uncertainties(self, epistemic, consistency, geometric, method='soft_dual_weight'):
        """
        Combines the independent uncertainty scores into a single gating metric.
        """
        if method == 'soft_dual_weight':
            u_excess = torch.relu(epistemic - 0.5)
            z_excess = torch.relu(geometric - 0.5)
            return torch.exp(-1.5 * u_excess - 1.0 * z_excess)
        elif method == 'ellipsoid_gate':
            u_excess = torch.relu(epistemic - 0.5)
            z_excess = torch.relu(geometric - 0.5)
            return torch.exp(-2.0 * (u_excess**2 + 0.5 * (z_excess**2)))
        elif method == 'rescue_gate':
            return torch.where((epistemic < 0.5) & (geometric >= 0.8), geometric, epistemic)
        elif method == 'epistemic':
            return torch.exp(-2.0 * torch.relu(epistemic - 0.1))
        elif method == 'geometric':
            return torch.exp(-2.0 * torch.relu(geometric - 0.5))
        elif method in ['and_gate', 'or_gate']:
            e_dec = torch.exp(-2.0 * torch.relu(epistemic - 0.1))
            g_dec = torch.exp(-2.0 * torch.relu(geometric - 0.5))
            return torch.min(e_dec, g_dec) if method == 'and_gate' else torch.max(e_dec, g_dec)
        return torch.ones_like(epistemic)

def set_uq_model(ARCH, modeldir, hd_encoder, num_levels, randomness, num_classes, device, subcluster_type='bipolar'):
    return DualGateModel(ARCH, modeldir, hd_encoder, num_levels, randomness, num_classes, device)

UQModel = DualGateModel

class MV_TTAModel(DualGateModel):
    """
    Multi-View TTA variant of DualGateModel.
    Incorporates multi-view 3D spatial augmentation consensus (veto_disagree) and cross-view
    softmax probability variance gating (view_var_gate) into the confidence gating and online adaptation loop.
    """
    def __init__(self, ARCH, modeldir, hd_encoder, num_levels, randomness, num_classes, device, gauss_rp=True, use_adaptor=True, mv_tta='veto_disagree'):
        super(MV_TTAModel, self).__init__(ARCH, modeldir, hd_encoder, num_levels, randomness, num_classes, device, gauss_rp=gauss_rp, use_adaptor=use_adaptor)
        self.mv_tta = mv_tta

    @torch.no_grad()
    def _get_spatial_consistency(self, enc, preds, proj_xyz=None, view_preds=None, view_var=None, **kwargs):
        """
        Pillar 1(c): Multi-View Spatial Consistency & Disagreement Veto.
        Checks if the predicted label agrees across 3D spatial augmentations (view_preds).
        """
        if view_preds is not None and len(view_preds) >= 2:
            pred_m1, pred_m2 = view_preds[0], view_preds[1]
            view_disagreement = (preds != pred_m1) | (preds != pred_m2)
            return ~view_disagreement
        return super()._get_spatial_consistency(enc, preds, proj_xyz=proj_xyz, **kwargs)

    @torch.no_grad()
    def get_confidence(self, enc, preds=None, method='soft_dual_weight', uncertainty=None, z_score=None, view_preds=None, view_var=None, **kwargs):
        """
        Extends DualGateModel.get_confidence with multi-view disagreement veto and cross-view variance gating.
        """
        base_weights, uncertainty, z_score = super().get_confidence(enc, preds=preds, method=method, uncertainty=uncertainty, z_score=z_score, view_preds=view_preds, view_var=view_var, **kwargs)
        
        if method == 'view_var_gate' and view_var is not None:
            view_var_decay = torch.exp(-2.0 * torch.relu(view_var - 0.05))
            base_weights = base_weights * torch.min(torch.exp(-2.0 * torch.relu(uncertainty - 0.5)), view_var_decay)
            
        if self.mv_tta == 'veto_disagree' and view_preds is not None:
            agree_mask = self._get_spatial_consistency(enc, preds, view_preds=view_preds)
            base_weights = base_weights * agree_mask.float()
            
        return base_weights, uncertainty, z_score

    @torch.no_grad()
    def online_update(self, enc, preds, update_weights, update_method='bm_ic4', ic_method='ic4', uncertainty=None, update_lr=0.005, normalize_weights=False, view_preds=None, **kwargs):
        """
        Applies multi-view disagreement veto to filter candidate points before invoking prototype momentum updates.
        """
        if self.mv_tta == 'veto_disagree' and view_preds is not None:
            agree_mask = self._get_spatial_consistency(enc, preds, view_preds=view_preds)
            update_weights = update_weights * agree_mask.float()
            
        return super().online_update(enc, preds, update_weights, update_method=update_method, ic_method=ic_method, uncertainty=uncertainty, update_lr=update_lr, normalize_weights=normalize_weights, view_preds=view_preds, **kwargs)

MVTTAModel = MV_TTAModel

def set_dual_gate_model(ARCH, modeldir, hd_encoder, num_levels, randomness, num_classes, device, subcluster_type='bipolar'):
    return DualGateModel(ARCH, modeldir, hd_encoder, num_levels, randomness, num_classes, device)

def set_mv_tta_model(ARCH, modeldir, hd_encoder, num_levels, randomness, num_classes, device, mv_tta='veto_disagree'):
    return MV_TTAModel(ARCH, modeldir, hd_encoder, num_levels, randomness, num_classes, device, mv_tta=mv_tta)