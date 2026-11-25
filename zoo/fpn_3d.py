import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from functools import partial


# from model.backbone import build_backbone


# def build_backbone(back_bone):
#     if back_bone == "resnet101":
#         return ResNet101(pretrained=True)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class FPN_3D(nn.Module):

    def __init__(self, depth, in_channels, out_classes):
        super(FPN_3D, self).__init__()
        self.depth = depth
        self.in_channels = in_channels
        self.out_classes = out_classes

        self.in_planes = 64
        self.back_bone = resnet18(in_channels)

        # torch.Size([64, 512, 2, 3, 3]) torch.Size([64, 256, 4, 6, 6]) torch.Size([64, 128, 7, 12, 12]) torch.Size([64, 64, 13, 24, 24])

        # Top layer
        self.toplayer = nn.Conv3d(512, 512, kernel_size=1, stride=1, padding=0)  # Reduce channels
        # Lateral layers
        self.latlayer1 = nn.Conv3d(256, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv3d(128, 128, kernel_size=1, stride=1, padding=0)
        self.latlayer3 = nn.Conv3d(64, 64, kernel_size=1, stride=1, padding=0)

        # Addendum layers to reduce channels
        self.sumlayer1 = nn.Conv3d(512, 256, kernel_size=1, stride=1, padding=0)  # Reduce channels
        self.sumlayer2 = nn.Conv3d(256, 128, kernel_size=1, stride=1, padding=0)
        self.sumlayer3 = nn.Conv3d(128, 64, kernel_size=1, stride=1, padding=0)

        # Top-down
        # torch.Size([64, 512, 2, 3, 3]) torch.Size([64, 256, 4, 6, 6]) torch.Size([64, 128, 7, 12, 12]) torch.Size([64, 64, 13, 24, 24])
        a, b, c, d = 2, 4, 8, 16

        self.conv2_3d_p5 = nn.Conv3d(512, 256, kernel_size=(a + 2, 3, 3), stride=1, padding=1)
        self.conv2_3d_p4 = nn.Conv3d(256, 256, kernel_size=(b + 2, 3, 3), stride=1, padding=1)
        self.conv2_3d_p3 = nn.Conv3d(128, 128, kernel_size=(c + 2, 3, 3), stride=1, padding=1)
        self.conv2_3d_p2 = nn.Conv3d(64, 128, kernel_size=(d + 2, 3, 3), stride=1, padding=1)

        self.semantic_branch_2d = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.conv2_2d = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(128, 64, kernel_size=1, stride=1, padding=0)
        self.conv4out = nn.Conv2d(64, self.depth, kernel_size=3, stride=1, padding=1)
        self.conv5out = nn.Conv2d(self.depth, self.out_classes, kernel_size=3, stride=1, padding=1)
        self.gn1 = nn.GroupNorm(128, 128)
        self.gn2 = nn.GroupNorm(256, 256)

    def _upsample3d(self, x, d, h, w):
        return F.interpolate(x, size=(d, h, w), mode='trilinear', align_corners=True)

    def _upsample2d(self, x, h, w):
        return F.interpolate(x, size=(h, w), mode='bilinear', align_corners=True)

    def _make_layer(self, Bottleneck, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(Bottleneck(self.in_planes, planes, stride))
            self.in_planes = planes * Bottleneck.expansion
        return nn.Sequential(*layers)

    def _upsample_add(self, x, y):
        '''Upsample and add two feature maps.
        Args:
          x: (Variable) top feature map to be upsampled.
          y: (Variable) lateral feature map.
        Returns:
          (Variable) added feature map.
        Note in PyTorch, when input size is odd, the upsampled feature map
        with `F.upsample(..., scale_factor=2, mode='nearest')`
        maybe not equal to the lateral feature map size.
        e.g.
        original input size: [N,_,15,15] ->
        conv2d feature map size: [N,_,8,8] ->
        upsampled feature map size: [N,_,16,16]
        So we choose bilinear upsample which supports arbitrary output sizes.
        '''
        _, _, D, H, W = y.size()
        return F.interpolate(x, size=(D, H, W), mode='trilinear', align_corners=True) + y

    def forward(self, x):
        # Bottom-up using backbone
        low_level_features = self.back_bone(x)
        # c1 = low_level_features[0]
        c2 = low_level_features[1]
        c3 = low_level_features[2]
        c4 = low_level_features[3]
        c5 = low_level_features[4]

        # Top-down
        p5 = self.toplayer(c5)
        p4 = self._upsample_add(
            torch.relu(self.sumlayer1(p5)), torch.relu(self.latlayer1(c4)))  # p5 interpolation to the size of c4
        p3 = self._upsample_add(
            torch.relu(self.sumlayer2(p4)), torch.relu(self.latlayer2(c3)))
        p2 = self._upsample_add(
            torch.relu(self.sumlayer3(p3)), torch.relu(self.latlayer3(c2)))

        # Semantic
        _, _, _, h, w = p2.size()
        # 256->256
        s5 = self.conv2_3d_p5(p5)
        s5 = torch.squeeze(s5, 2)  # squeeze only dim 2 to avoid to remove the batch dimension
        s5 = self._upsample2d(torch.relu(self.gn2(s5)), h, w)
        # 256->256 [32, 256, 24, 24]
        s5 = self._upsample2d(torch.relu(self.gn2(self.conv2_2d(s5))), h, w)
        # 256->128 [32, 128, 24, 24]
        s5 = self._upsample2d(torch.relu(self.gn1(self.semantic_branch_2d(s5))), h, w)

        # 256->256 p4:[32, 256, 4, 6, 6] -> s4:[32, 256, 1, 6, 6]
        s4 = self.conv2_3d_p4(p4)
        s4 = torch.squeeze(s4, 2)  # s4:[32, 256, 6, 6]
        s4 = self._upsample2d(torch.relu(self.gn2(s4)), h, w)  # s4:[32, 256, 24, 24]
        # 256->128  s4:[32, 128, 24, 24]
        s4 = self._upsample2d(torch.relu(self.gn1(self.semantic_branch_2d(s4))), h, w)

        # 256->128
        s3 = self.conv2_3d_p3(p3)
        s3 = torch.squeeze(s3, 2)
        s3 = self._upsample2d(torch.relu(self.gn1(s3)), h, w)

        s2 = self.conv2_3d_p2(p2)
        s2 = torch.squeeze(s2, 2)
        s2 = self._upsample2d(torch.relu(self.gn1(s2)), h, w)

        out = self._upsample2d(self.conv3(s2 + s3 + s4 + s5), 2 * h, 2 * w)
        # introducing MSELoss on NDVI signal
        out_cai = torch.sigmoid(self.conv4out(out))  # for Class Activation Interval
        out_cls = self.conv5out(out_cai)  # for Classification

        return out_cls


def resnet18(in_channels):
    """Constructs a ResNet-18 model.
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], in_channels)
    return model


class ResNet(nn.Module):

    def __init__(self, block, layers, in_channels):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv3d(
            in_channels,
            64,
            kernel_size=(7, 3, 3),  # orig: 7
            stride=(1, 1, 1),  # orig: (1, 2, 2)
            padding=(3, 1, 1),  # orig: (3, 3, 3)
            bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], 'B')
        self.layer2 = self._make_layer(block, 128, layers[1], 'B', stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], 'B', stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], 'B', stride=2)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight = nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(
                    downsample_basic_block,
                    planes=planes * block.expansion,
                    stride=stride)
            else:
                downsample = nn.Sequential(
                    nn.Conv3d(
                        self.inplanes,
                        planes * block.expansion,
                        kernel_size=1,
                        stride=stride,
                        bias=False), nn.BatchNorm3d(planes * block.expansion))

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        low_level_feat1 = x

        x = self.layer1(x)
        low_level_feat2 = x
        x = self.layer2(x)
        low_level_feat3 = x
        x = self.layer3(x)
        low_level_feat4 = x
        x = self.layer4(x)
        low_level_feat5 = x
        return [low_level_feat1, low_level_feat2, low_level_feat3, low_level_feat4, low_level_feat5]


def get_fine_tuning_parameters(model, ft_begin_index):
    if ft_begin_index == 0:
        return model.parameters()

    ft_module_names = []
    for i in range(ft_begin_index, 5):
        ft_module_names.append('layer{}'.format(i))
    ft_module_names.append('fc')

    parameters = []
    for k, v in model.named_parameters():
        for ft_module in ft_module_names:
            if ft_module in k:
                parameters.append({'params': v})
                break
        else:
            parameters.append({'params': v, 'lr': 0.0})

    return parameters


def conv3x3x3(in_planes, out_planes, stride=1):
    # 3x3x3 convolution with padding
    return nn.Conv3d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False)


def downsample_basic_block(x, planes, stride):
    out = F.avg_pool3d(x, kernel_size=1, stride=stride)
    zero_pads = torch.Tensor(
        out.size(0), planes - out.size(1), out.size(2), out.size(3),
        out.size(4)).zero_()
    if isinstance(out.data, torch.cuda.FloatTensor):
        zero_pads = zero_pads.cuda()

    out = Variable(torch.cat([out.data, zero_pads], dim=1))

    return out


class BasicBlock(nn.Module):
    expansion = 1

    # Try with pre-act ResNet
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = nn.Conv3d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out