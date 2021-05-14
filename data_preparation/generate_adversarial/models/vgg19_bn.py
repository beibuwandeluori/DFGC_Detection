import torch
import torchvision
import torch.nn as nn
from torch.nn import init
from torch.autograd import Variable


######################################################################
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


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal_(m.weight.data, std=0.001)
        init.constant_(m.bias.data, 0.0)


# Defines the new fc layer and classification layer
# |--Linear--|--bn--|--relu--|--Linear--|
class ClassBlock(nn.Module):
    def __init__(self, input_dim, class_num, droprate, relu=False, bnorm=True,
                 num_bottleneck=512, linear=True, return_f=False):
        super(ClassBlock, self).__init__()
        self.return_f = return_f
        add_block = []
        if linear:
            add_block += [nn.Linear(input_dim, num_bottleneck)]
        else:
            num_bottleneck = input_dim
        if bnorm:
            add_block += [nn.BatchNorm1d(num_bottleneck)]
        if relu:
            add_block += [nn.LeakyReLU(0.1)]
        if droprate>0:
            add_block += [nn.Dropout(p=droprate)]
        add_block = nn.Sequential(*add_block)
        add_block.apply(weights_init_kaiming)

        classifier = []
        classifier += [nn.Linear(num_bottleneck, class_num)]
        classifier = nn.Sequential(*classifier)
        classifier.apply(weights_init_classifier)

        self.add_block = add_block
        self.classifier = classifier

    def forward(self, x):
        x = self.add_block(x)
        if self.return_f:
            f = x
            x = self.classifier(x)
            return x, f
        else:
            x = self.classifier(x)
            return x

# Define the VGG19_BN-based Model
class VGG19_BN(nn.Module):
    def __init__(self, class_num, droprate=0.5, return_f=False):
        super(VGG19_BN, self).__init__()
        self.class_num = class_num
        self.vgg = torchvision.models.vgg19_bn(pretrained=False)

        self.avgpool_global = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier_global = ClassBlock(512, self.class_num, droprate=droprate,
                                            relu=False, bnorm=True,
                                            num_bottleneck=256, return_f=return_f)

    def forward(self, x):
        fea = self.vgg.features(x)
        avg_glo = self.avgpool_global(fea)

        glo = avg_glo.view(avg_glo.size(0), avg_glo.size(1))
        # print('global_shape:' + str(glo.size()))

        if self.classifier_global.return_f:
            y, y_fea = self.classifier_global(glo)
            return y, y_fea
        else:
            y = self.classifier_global(glo)
            return y



def test():
    dummy_input = torch.randn(2, 3, 256, 256)

    model = ResNet34(2)
    out, ca_act_reg = model(dummy_input)

    print(model)
    print('\nModel input shape :', dummy_input.size())
    print('Model output shape :', out.size())
    print('ca_act_reg :', ca_act_reg)


if __name__ == '__main__':
    test()





















