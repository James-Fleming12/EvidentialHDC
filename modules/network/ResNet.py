import torch.nn as nn
import torch
from torch.nn import functional as F
import numpy as np

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class ScaleOnlyInstanceNorm2d(nn.Module):
    """Iteration C8 lever 2: scale-only internal InstanceNorm. Normalizes each
    (sample, channel) by its own std over spatial dims but does NOT subtract the mean,
    so the per-dimension offset structure survives. Affine scale/bias still applied
    after division, mirroring InstanceNorm2d's affine=True behavior."""
    def __init__(self, num_features, eps=1e-5, affine=True, momentum=None, track_running_stats=False):
        super(ScaleOnlyInstanceNorm2d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if affine:
            self.weight = nn.Parameter(torch.ones(num_features))
            self.bias = nn.Parameter(torch.zeros(num_features))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

    def forward(self, x):
        B, C, H, W = x.shape
        # mean per (sample, channel); x is uncentered so mean survives
        mu = x.mean(dim=(2, 3), keepdim=True)
        var = x.var(dim=(2, 3), keepdim=True, unbiased=False)
        std = (var + self.eps).sqrt()
        x = x / std
        if self.affine:
            x = x * self.weight.view(1, C, 1, 1) + self.bias.view(1, C, 1, 1)
        return x

def norm_layer_for(norm, scale_in=False):
    """Map a norm string to the layer class. 'bn' -> BatchNorm2d, 'in' ->
    InstanceNorm2d, 'in_scale' -> ScaleOnlyInstanceNorm2d. scale_in=True forces the
    scale-only variant regardless of the string (used by the C8 lever-2 micro)."""
    if norm == 'in_scale' or scale_in:
        return ScaleOnlyInstanceNorm2d
    if norm == 'in':
        return nn.InstanceNorm2d
    return nn.BatchNorm2d

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, relu=True,
                 norm='bn', scale_in=False):
        super(BasicConv2d, self).__init__()
        self.relu = relu
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = norm_layer_for(norm, scale_in)(out_planes)
        if self.relu:
            self.relu = nn.LeakyReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.relu:
            x = self.relu(x)
        return x

class Final_Model(nn.Module):

    def __init__(self, backbone_net, semantic_head):
        super(Final_Model, self).__init__()
        self.backend = backbone_net
        self.semantic_head = semantic_head

    def forward(self, x):
        middle_feature_maps = self.backend(x)

        semantic_output = self.semantic_head(middle_feature_maps)

        return semantic_output

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, if_BN=None, use_adaptor=False, norm_layer=None):
        super(BasicBlock, self).__init__()
        self.if_BN = if_BN
        if self.if_BN and norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        if self.if_BN:
            self.bn1 = norm_layer(planes)
        self.relu = nn.LeakyReLU()
        self.conv2 = conv3x3(planes, planes)
        if self.if_BN:
            self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

        self.adaptor = Adaptor(planes) if use_adaptor else None

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        if self.if_BN:
            out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        if self.if_BN:
            out = self.bn2(out)

        if self.adaptor is not None:
            out = out + self.adaptor(out)

        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out

class Adaptor(nn.Module):
    """
    Lightweight parallel adaptor for a ResNet block (paper Sec 3.2).
    Down-projects channels by ratio r, applies ReLU, up-projects back.
    Up-projection is zero-initialised so the adaptor is a no-op at
    the start of test-time adaptation.
    """
    def __init__(self, channels: int, r: int = 32):
        super().__init__()
        bottleneck = max(1, channels // r)
        self.down = nn.Conv2d(channels, bottleneck, kernel_size=1, bias=False)
        self.relu = nn.ReLU()
        self.up   = nn.Conv2d(bottleneck, channels, kernel_size=1, bias=False)
        nn.init.zeros_(self.up.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(self.relu(self.down(x)))

class ResNet_34(nn.Module):
    def __init__(self, nclasses, aux, block=BasicBlock, layers=[3, 4, 6, 3], if_BN=True, zero_init_residual=False,
                 norm_layer=None, groups=1, width_per_group=64, use_adaptor=True,
                 corr_dim=0, corr_mode='ind', inv_dim=128, norm='bn', input_in=False,
                 norm_channels=None, scale_only=False, norm_scope='all', scale_in=False,
                 geoid_head=False):
        super(ResNet_34, self).__init__()
        if norm_layer is None:
            norm_layer = norm_layer_for(norm, scale_in)
        self._norm_layer = norm_layer
        # norm_scope (Iteration C8 lever 1): 'all' = current behavior (every stage uses
        # `norm`); 'in_late' = InstanceNorm only in the LATE stages (layer3/layer4 + the
        # bottleneck conv_1/conv_2) while the early geometry blocks (conv1-3, layer1/2)
        # keep BatchNorm. The C8 diagnostic showed the healthy-condition ceiling loss is
        # CONTINUOUS (survives every decoding), and the hypothesis is that InstanceNorm
        # throughout the whole backbone erases the early-stage per-dimension anisotropy
        # the healthy conditions recover through; scoping it to the late bottleneck
        # keeps the fog/crosstalk covariate-shift robustness where it acts.
        self.norm_scope = norm_scope
        # scale_in (Iteration C8 lever 2): use a SCALE-ONLY internal InstanceNorm that
        # divides by the per-scan per-channel std but does NOT center. Full InstanceNorm
        # forces every healthy scan's channels to zero-mean, erasing the per-dimension
        # offset structure (the packing loss); keeping the mean preserves that
        # anisotropy while still absorbing the magnitude shift.
        self.scale_in = scale_in
        early_norm = 'bn' if norm_scope == 'in_late' else norm
        late_norm = norm if norm_scope == 'all' else ('in' if norm_scope == 'in_late' else norm)

        self.conv1 = BasicConv2d(5, 64, kernel_size=3, padding=1, norm=early_norm, scale_in=scale_in)
        self.conv2 = BasicConv2d(64, 128, kernel_size=3, padding=1, norm=early_norm, scale_in=scale_in)
        self.conv3 = BasicConv2d(128, 128, kernel_size=3, padding=1, norm=early_norm, scale_in=scale_in)

        self.inplanes = 128

        self.groups = groups
        self.base_width = width_per_group
        self.use_adaptor = use_adaptor
        self.if_BN = if_BN
        self.dilation = 1
        self.aux = aux

        self.layer1 = self._make_layer(block, 128, layers[0], use_adaptor=use_adaptor, norm=early_norm)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, use_adaptor=use_adaptor, norm=early_norm)
        self.layer3 = self._make_layer(block, 128, layers[2], stride=2, use_adaptor=use_adaptor, norm=late_norm)
        self.layer4 = self._make_layer(block, 128, layers[3], stride=2, use_adaptor=use_adaptor, norm=late_norm)

        self.conv_1 = BasicConv2d(640, 256, kernel_size=3, padding=1, norm=late_norm, scale_in=scale_in)
        self.conv_2 = BasicConv2d(256, inv_dim, kernel_size=3, padding=1, norm=late_norm, scale_in=scale_in)
        # Decoupling branch (Iteration-15 shortlist): a SECOND bottleneck head with its
        # own capacity. The invariant head (conv_2) keeps the full inv_dim and carries
        # GMSIFC+LSCC+SupCon; the corruption head (conv_corr) is either an independent
        # branch (mode='ind') or an additive residual inv + delta (mode='res'). The
        # decoder reads the concatenation [inv, corr], so the HDC oracle has access to
        # the retained shifted direction.
        self.corr_dim = corr_dim
        self.corr_mode = corr_mode
        self.inv_dim = inv_dim
        self.input_in = input_in
        self.input_in_prob = None  # set by the trainer for conditional-input-IN
        # norm_channels: if set, per-scan input normalization applies ONLY to these
        # channel indices (e.g. (0,4) = range+remission) and leaves the rest (xyz
        # geometry) untouched -- the Iteration-19.11.2 condition-aware fix.
        self.norm_channels = norm_channels
        # scale_only: divide by per-scan std WITHOUT subtracting the mean (preserves
        # direction -- the Iteration-19.12 candidate for the cleaner general fix).
        self.scale_only = scale_only
        if corr_dim > 0:
            self.conv_corr = BasicConv2d(256, corr_dim, kernel_size=3, padding=1, norm=late_norm, scale_in=scale_in)
            self.semantic_output = nn.Conv2d(inv_dim + corr_dim, nclasses, 1)
        else:
            self.conv_corr = None
            self.semantic_output = nn.Conv2d(inv_dim, nclasses, 1)

        if self.aux:
            self.aux_head1 = nn.Conv2d(128, nclasses, 1)
            self.aux_head2 = nn.Conv2d(128, nclasses, 1)
            self.aux_head3 = nn.Conv2d(128, nclasses, 1)

        # GeoID inlier-discrimination head (port of exp_geoid.py final_cls): a 1x1
        # conv on the bottleneck output -> 1 channel (logit of "real inlier" vs
        # "synthetic displaced outlier"). Trained with BCE alongside the seg loss;
        # the synthetic points are injected by GenTrainer's get_augmented_view for
        # the geo_inlier methods. Enabled via input_in-like flag (geoid_head).
        self.geoid_head = None
        if geoid_head:
            self.geoid_head = nn.Conv2d(inv_dim, 1, 1)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False, use_adaptor=False, norm=None):
        norm_layer = self._norm_layer if norm is None else norm_layer_for(norm, self.scale_in)
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            if self.if_BN:
                downsample = nn.Sequential(
                    conv1x1(self.inplanes, planes * block.expansion, stride),
                    norm_layer(planes * block.expansion),
                )
            else:
                downsample = nn.Sequential(
                    conv1x1(self.inplanes, planes * block.expansion, stride)
                )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, if_BN=self.if_BN, use_adaptor=use_adaptor, norm_layer=norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups, base_width=self.base_width, dilation=self.dilation, if_BN=self.if_BN, use_adaptor=use_adaptor, norm_layer=norm_layer))
            
        return nn.Sequential(*layers)

    def _input_instancenorm(self, in_vol):
        """Per-scan input normalization (Iteration-19.10 level-1 covariate shift). The
        parser normalizes the 5-channel input by FIXED clean-data img_means/img_stds;
        under fog/crosstalk the network then receives inputs scaled against clean
        statistics. This re-normalizes each scan's valid channels by its OWN per-scan
        statistics (over valid points only), the training-side mirror of the
        BN-statistic alignment TTA lever. Applied inside forward so it holds at BOTH
        train and eval time (the diagnostics call model() directly).
        Design modes (Iteration-19.12):
          - default (norm_channels=None): full per-scan mean+std on all channels.
            (Verified: this ERASES fog's recoverable direction -- dir_retention -> 1.)
          - norm_channels=(0,4): mean+std on range+remission only, xyz left in clean
            stats. (Verified: fog recovers, crosstalk up -- the 19.12 winner.)
          - scale_only=True: divide by per-scan std on ALL channels but DO NOT subtract
            the mean -- absorbs the magnitude statistics shift (crosstalk) while
            preserving the direction (fog). The cleaner general version of the
            channel-restricted fix, testable as a micro before the medium run."""
        if self.scale_only:
            valid = (in_vol[:, 0:1, :, :] > 0).float()
            x = in_vol * valid
            denom = valid.sum(dim=(2, 3), keepdim=True).clamp(min=1)
            var = ((x - x.sum(dim=(2, 3), keepdim=True) / denom).pow(2) * valid
                   ).sum(dim=(2, 3), keepdim=True) / denom
            std = var.clamp(min=1e-6).sqrt()
            return (x / std) * valid
        if self.norm_channels is not None:
            # normalize only the listed channels; keep the rest unchanged
            x = in_vol.clone()
            sub = x[:, self.norm_channels]
            valid = (x[:, 0:1, :, :] > 0).float()
            xv = sub * valid
            denom = valid.sum(dim=(2, 3), keepdim=True).clamp(min=1)
            mu = xv.sum(dim=(2, 3), keepdim=True) / denom
            var = ((xv - mu).pow(2) * valid).sum(dim=(2, 3), keepdim=True) / denom
            std = var.clamp(min=1e-6).sqrt()
            x[:, self.norm_channels] = ((xv - mu) / std) * valid
            return x
        valid = (in_vol[:, 0:1, :, :] > 0).float()
        x = in_vol * valid
        denom = valid.sum(dim=(2, 3), keepdim=True).clamp(min=1)
        mu = x.sum(dim=(2, 3), keepdim=True) / denom
        var = ((x - mu).pow(2) * valid).sum(dim=(2, 3), keepdim=True) / denom
        std = var.clamp(min=1e-6).sqrt()
        out = (x - mu) / std
        return out * valid

    def forward(self, x, only_feat=False, return_enc=False, return_stage4=False):
        if self.input_in:
            if self.input_in_prob is not None and self.training:
                # Stochastic input-IN (conditional-input-IN training): apply the
                # per-scan normalization to a RANDOM SUBSET of the batch's scans
                # at train time, so the network learns to produce good features
                # under BOTH normalized and raw inputs. At eval input_in_prob is
                # ignored -> the gate (on/off) is a clean choice the weights
                # support (unlike the failed eval-only gate, whose weights were
                # trained with input-IN always on).
                keep = (torch.rand(x.size(0), device=x.device) < self.input_in_prob)
                x_n = self._input_instancenorm(x)
                x = torch.where(keep.view(-1, 1, 1, 1), x_n, x)
            else:
                x = self._input_instancenorm(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        x_1 = self.layer1(x)  # 1/1
        x_2 = self.layer2(x_1)  # 1/2
        x_3 = self.layer3(x_2)  # 1/4
        x_4 = self.layer4(x_3)  # 1/8

        res_2 = F.interpolate(x_2, size=x.size()[2:], mode='bilinear', align_corners=True)
        res_3 = F.interpolate(x_3, size=x.size()[2:], mode='bilinear', align_corners=True)
        res_4 = F.interpolate(x_4, size=x.size()[2:], mode='bilinear', align_corners=True)

        res = [x, x_1, res_2, res_3, res_4]
        feat_map = torch.cat(res, dim=1) 
        
        out = self.conv_1(feat_map)
        out_inv = self.conv_2(out)

        if self.corr_dim > 0:
            out_corr = self.conv_corr(out)
            if self.corr_mode == 'res':
                out_corr = out_inv[:, :self.corr_dim] + out_corr
            out = torch.cat([out_inv, out_corr], dim=1)
        else:
            out = out_inv

        if only_feat:
            return out

        logits = self.semantic_output(out)
        pred = F.softmax(logits, dim=1)

        if self.aux:
            aux2 = F.softmax(self.aux_head1(res_2), dim=1)
            aux3 = F.softmax(self.aux_head2(res_3), dim=1)
            aux4 = F.softmax(self.aux_head3(res_4), dim=1)
            if self.geoid_head is not None:
                geoid_logits = self.geoid_head(out)
                if return_stage4:
                    return pred, [aux2, aux3, aux4], out, x_4, geoid_logits
                if return_enc:
                    return pred, [aux2, aux3, aux4], out, feat_map, geoid_logits
                return pred, [aux2, aux3, aux4], out, geoid_logits
            if return_stage4:
                return pred, [aux2, aux3, aux4], out, x_4
            if return_enc:
                return pred, [aux2, aux3, aux4], out, feat_map
            return pred, [aux2, aux3, aux4], out

        if return_stage4:
            return pred, out, x_4
        if return_enc:
            return pred, out, feat_map

        return pred, out
    
    def adaptor_parameters(self):
        """Yield only the adaptor parameters (what gets updated at test time)."""
        for module in self.modules():
            if isinstance(module, Adaptor):
                yield from module.parameters()

    @torch.enable_grad()
    def test_time_adapt(
        self,
        x: torch.Tensor,
        mu_tr: torch.Tensor,
        sigma_tr: torch.Tensor,
        mu_te_prev: torch.Tensor,
        alpha: float = 0.01,
        lr: float = 1e-3,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Freezes all backbone params; updates only adaptor weights via
        image-level KL divergence loss between training and (EMA-updated)
        test feature distributions.

        Args:
            x:           Current test batch  [B, C, H, W]
            mu_tr:       Pre-computed training feature mean  [C]
            sigma_tr:    Pre-computed training feature variance  [C]
            mu_te_prev:  EMA mean from the previous step  [C]
            alpha:       EMA momentum for test mean (default 0.01)
            lr:          SGD learning rate for the adaptor

        Returns:
            pred:        Softmax prediction on x  [B, nclasses, H, W]
            mu_te_new:   Updated EMA test mean for the next step  [C]
        """
        for p in self.parameters():
            p.requires_grad_(False)
        adaptor_params = list(self.adaptor_parameters())
        for p in adaptor_params:
            p.requires_grad_(True)

        optimizer = torch.optim.SGD(adaptor_params, lr=lr)

        self.train()
        pred, feat = self(x)[:2]

        mu_te_curr = feat.mean(dim=[0, 2, 3]).detach()
        mu_te_new = (1 - alpha) * mu_te_prev + alpha * mu_te_curr

        eps = 1e-6
        loss_img = (0.5 * ((mu_tr - mu_te_new) ** 2) / (sigma_tr + eps)).sum()

        optimizer.zero_grad()
        loss_img.backward()
        optimizer.step()

        self.eval()
        with torch.no_grad():
            pred, _ = self(x)[:2]

        return pred, mu_te_new.detach()

if __name__ == "__main__":
    import time
    model = ResNet_34(20).cuda()
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Number of parameters: ", pytorch_total_params / 1000000, "M")
    time_train = []
    for i in range(20):
        inputs = torch.randn(1, 5, 64, 2048).cuda()
        model.eval()
        with torch.no_grad():
          start_time = time.time()
          outputs = model(inputs)
        torch.cuda.synchronize()  # wait for cuda to finish (cuda is asynchronous!)
        fwt = time.time() - start_time
        time_train.append(fwt)
        print ("Forward time per img: %.3f (Mean: %.3f)" % (
          fwt / 1, sum(time_train) / len(time_train) / 1))
        time.sleep(0.15)

