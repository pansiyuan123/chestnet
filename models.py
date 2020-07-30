import torch
import torch.nn as nn
import torchvision.models as models

from chestx.pooling import WildcatPool2d, ClassWisePool


class _SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(_SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                nn.Linear(channel, channel // reduction),
                nn.ReLU(inplace=True),
                nn.Linear(channel // reduction, channel),
                nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)

        return x * y.expand_as(x)


class DenseNetChest(nn.Module):

    def __init__(self, model, num_classes, pooling=WildcatPool2d(), dense=False):
        super(DenseNetChest, self).__init__()

        self.dense = dense

        # self.features = nn.Sequential(
        #     model.features.conv0,
        #     model.features.norm0,
        #     model.features.relu0,
        #     model.features.pool0,
        #     model.features.denseblock1,
        #     model.features.transition1,
        #     _SELayer(model.features.transition1.conv.out_channels),
        #     model.features.denseblock2,
        #     model.features.transition2,
        #     _SELayer(model.features.transition2.conv.out_channels),
        # )
        # num_features1 = model.features.transition2.conv.out_channels
        # self.classifier1 = nn.Sequential(
        #     nn.Conv2d(num_features1, num_classes, kernel_size=1, stride=1, padding=0, bias=True))
        #
        # self.features1 = nn.Sequential(
        #     model.features.denseblock3,
        #     model.features.transition3,
        #     _SELayer(model.features.transition3.conv.out_channels)
        # )
        # num_features2 = model.features.transition3.conv.out_channels
        # self.classifier2 = nn.Sequential(
        #     nn.Conv2d(num_features2, num_classes, kernel_size=1, stride=1, padding=0, bias=True))
        #
        # self.features2 = nn.Sequential(
        #     model.features.denseblock4,
        #     model.features.norm5,
        #     _SELayer(model.features.norm5.num_features)
        # )

        self.features = nn.Sequential(
            model.features.conv0,
            model.features.norm0,
            model.features.relu0,
            model.features.pool0,
            model.features.denseblock1,
            model.features.transition1,
            _SELayer(model.features.transition1.conv.out_channels),
            model.features.denseblock2,
            model.features.transition2,
            _SELayer(model.features.transition2.conv.out_channels),
            model.features.denseblock3,
            model.features.transition3,
            _SELayer(model.features.transition3.conv.out_channels),
            model.features.denseblock4,
            model.features.norm5,
            _SELayer(model.features.norm5.num_features),
        )
        # self.features = model.features

        # classification layer
        num_features = model.features.norm5.num_features
        self.classifier = nn.Sequential(
            nn.Conv2d(num_features, num_classes, kernel_size=1, stride=1, padding=0, bias=True))

        self.spatial_pooling = pooling


    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        if not self.dense:
            x = self.spatial_pooling(x)

        return torch.sigmoid(x)

    def get_config_optim(self, lr, lrp):
        return [{'params': self.features.parameters(), 'lr': lr * lrp},
                {'params': self.classifier.parameters()},
                {'params': self.spatial_pooling.parameters()}]


def densenet121_chest(num_classes, pretrained=True, kmax=1, kmin=None, alpha=1, num_maps=1):
    model = models.densenet121(pretrained)
    pooling = nn.Sequential()
    pooling.add_module('class_wise', ClassWisePool(num_maps))
    pooling.add_module('spatial', WildcatPool2d(kmax, kmin, alpha))
    return DenseNetChest(model, num_classes * num_maps, pooling=pooling)


