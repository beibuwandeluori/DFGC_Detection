import torch
import torch.nn as nn
from torch.nn import init
import torchvision
from efficientnet_pytorch import EfficientNet

# fc layer weight init
def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in') # For old pytorch, you may use kaiming_normal.
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        init.constant_(m.bias.data, 0.0)

    elif classname.find('BatchNorm1d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)

# 当in_channel != 3 时，初始化模型的第一个Conv的weight， 把之前的通道copy input_chaneel/3 次
def init_imagenet_weight(_conv_stem_weight, input_channel=3):
    for i in range(input_channel//3):
        if i == 0:
            _conv_stem_weight_new = _conv_stem_weight
        else:
            _conv_stem_weight_new = torch.cat([_conv_stem_weight_new, _conv_stem_weight], axis=1)

    return torch.nn.Parameter(_conv_stem_weight_new)


class TransferModel(nn.Module):
    """
    Simple transfer learning model that takes an imagenet pretrained model with
    a fc layer as base model and retrains a new fc layer for num_out_classes
    """
    def __init__(self, modelchoice, num_out_classes=2, dropout=0.0):
        super(TransferModel, self).__init__()
        self.modelchoice = modelchoice
        if modelchoice == 'efficientnet-b7' or modelchoice == 'efficientnet-b6'\
                or modelchoice == 'efficientnet-b5' or modelchoice == 'efficientnet-b4'\
                or modelchoice == 'efficientnet-b3' or modelchoice == 'efficientnet-b2'\
                or modelchoice == 'efficientnet-b1' or modelchoice == 'efficientnet-b0':
            # self.model = EfficientNet.from_name(modelchoice, override_params={'num_classes': num_out_classes})
            self.model = get_efficientnet(model_name=modelchoice, num_classes=num_out_classes)
        elif modelchoice == 'resnet18' or modelchoice == 'resnet50':
            if modelchoice == 'resnet50':
                self.model = torchvision.models.resnet50(pretrained=True)
            if modelchoice == 'resnet18':
                self.model = torchvision.models.resnet18(pretrained=True)
            # Replace fc
            num_ftrs = self.model.fc.in_features
            if not dropout:
                self.model.fc = nn.Linear(num_ftrs, num_out_classes)
                init.normal_(self.model.fc.weight.data, std=0.001)
                init.constant_(self.model.fc.bias.data, 0.0)
            else:
                self.model.fc = nn.Sequential(
                    nn.Linear(num_ftrs, 256),
                    nn.Dropout(p=dropout),
                    nn.Linear(256, num_out_classes)
                )
                init.normal_(self.model.fc[2].weight.data, std=0.001)
                init.constant_(self.model.fc[2].bias.data, 0.0)
        else:
            raise Exception('Choose valid model, e.g. resnet50')

    def forward(self, x):

        x = self.model(x)
        return x


def model_selection(modelname, num_out_classes, dropout=None):
    """
    :param modelname:
    :return: model, image size, pretraining<yes/no>, input_list
    """
    if modelname == 'resnet18' or modelname == 'resnet50' :
        return TransferModel(modelchoice=modelname, dropout=dropout,
                             num_out_classes=num_out_classes), \
               224, True, ['image'], None
    elif modelname == 'efficientnet-b7' or modelname == 'efficientnet-b6'\
            or modelname == 'efficientnet-b5' or modelname == 'efficientnet-b4' \
            or modelname == 'efficientnet-b3' or modelname == 'efficientnet-b2' \
            or modelname == 'efficientnet-b1' or modelname == 'efficientnet-b0':
        return TransferModel(modelchoice=modelname, dropout=dropout,
                             num_out_classes=num_out_classes), \
               224, True, ['image'], None

    else:
        raise NotImplementedError(modelname)


def get_efficientnet(model_name='efficientnet-b0', num_classes=2):
    net = EfficientNet.from_pretrained(model_name)
    # net = EfficientNet.from_name(model_name)
    in_features = net._fc.in_features
    net._fc = nn.Linear(in_features=in_features, out_features=num_classes, bias=True)

    return net


if __name__ == '__main__':
    model, image_size, *_ = model_selection('efficientnet-b0', num_out_classes=2)

    model = model.to(torch.device('cpu'))
    from torchsummary import summary
    input_s = (3, image_size, image_size)
    print(summary(model, input_s, device='cpu'))

    # print(model._modules.items())
    # print(model)


    pass

