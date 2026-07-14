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


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, relu=True):
        super(BasicConv2d, self).__init__()
        self.relu = relu
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
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
                 base_width=64, dilation=1, if_BN=None, use_adaptor=False):
        super(BasicBlock, self).__init__()
        self.if_BN = if_BN
        if self.if_BN:
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
                 norm_layer=None, groups=1, width_per_group=64, use_adaptor=True):
        super(ResNet_34, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.if_BN = if_BN
        self.dilation = 1
        self.aux = aux

        self.groups = groups
        self.base_width = width_per_group

        self.conv1 = BasicConv2d(5, 64, kernel_size=3, padding=1)
        self.conv2 = BasicConv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = BasicConv2d(128, 128, kernel_size=3, padding=1)

        self.inplanes = 128

        self.use_adaptor = use_adaptor

        self.layer1 = self._make_layer(block, 128, layers[0], use_adaptor=use_adaptor)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, use_adaptor=use_adaptor)
        self.layer3 = self._make_layer(block, 128, layers[2], stride=2, use_adaptor=use_adaptor)
        self.layer4 = self._make_layer(block, 128, layers[3], stride=2, use_adaptor=use_adaptor)

        self.conv_1 = BasicConv2d(640, 256, kernel_size=3, padding=1)
        self.conv_2 = BasicConv2d(256, 128, kernel_size=3, padding=1)
        self.semantic_output = nn.Conv2d(128, nclasses, 1)

        if self.aux:
            self.aux_head1 = nn.Conv2d(128, nclasses, 1)
            self.aux_head2 = nn.Conv2d(128, nclasses, 1)
            self.aux_head3 = nn.Conv2d(128, nclasses, 1)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False, use_adaptor=False):
        norm_layer = self._norm_layer
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
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, if_BN=self.if_BN, use_adaptor=use_adaptor))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups, base_width=self.base_width, dilation=self.dilation, if_BN=self.if_BN, use_adaptor=use_adaptor))
            
        return nn.Sequential(*layers)

    def forward(self, x, only_feat=False, return_enc=False):
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
        out = self.conv_2(out)
        
        if only_feat:
            return out

        logits = self.semantic_output(out)
        pred = F.softmax(logits, dim=1)

        if self.aux:
            aux2 = F.softmax(self.aux_head1(res_2), dim=1)
            aux3 = F.softmax(self.aux_head2(res_3), dim=1)
            aux4 = F.softmax(self.aux_head3(res_4), dim=1)
            if return_enc:
                return pred, [aux2, aux3, aux4], out, feat_map
            return pred, [aux2, aux3, aux4], out

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




