import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms


def lpips_loss(lpips, images, adv_images):
    batch_size = images.size(0)
    normal_layer = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    alex_loss = 0
    for k in range(batch_size):
        alex_loss += lpips(normal_layer(images[k]), normal_layer(adv_images[k])).view(1)
    loss = alex_loss / batch_size

    return loss[0]


def input_diversity(image, div_prob=0.9, low=1.0, high=1.2):
    if random.random() > div_prob:
        return image
    # rnd = random.randint(low, high)
    assert low>0 and high>low
    rnd =  random.uniform(low, high)
    B,C,H,W = image.size()
    new_H,new_W =int(H*rnd),int(W*rnd)
    max_H,max_W = int(H*high),int(W*high)
    rescaled = F.interpolate(image, size=[new_H, new_W], mode='bilinear')
    h_rem = max_H - new_H
    w_rem = max_W - new_W
    pad_top = random.randint(0, h_rem)
    pad_bottom = h_rem - pad_top
    pad_left = random.randint(0, w_rem)
    pad_right = w_rem - pad_left
    padded = F.pad(rescaled, [pad_top, pad_bottom, pad_left, pad_right], 'constant', 0)
    return padded


class AttackerTPGD:
    def __init__(self, eps=8/255.0, alpha=2/255.0, steps=40, low=0.8, high=1.2, div_prob=0.9, device=torch.device('cuda')):
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.low = low
        self.high = high
        self.div_prob = div_prob
        self.device = device

    def attack(self, model, images, labels=None,mask=None):
        r"""
        Overridden.
        """
        images = images.clone().detach().to(self.device)
        logit_ori = model(images).detach()

        adv_images = images + 0.001 * torch.randn_like(images)
        adv_images = torch.clamp(adv_images, min=0, max=1).detach()

        loss = nn.KLDivLoss(reduction='sum')

        steps=np.random.randint(int(self.steps//10),self.steps)
        eps=random.uniform(1/255,self.eps)

        for i in range(steps):

            adv_images.requires_grad = True
            adv_div = input_diversity(adv_images, div_prob=self.div_prob, low=self.low, high=self.high)
            logit_adv = model(adv_div)

            cost = loss(F.log_softmax(logit_adv, dim=1),
                        F.softmax(logit_ori, dim=1))

            grad = torch.autograd.grad(cost, adv_images,
                                       retain_graph=False, create_graph=False)[0]
            if mask is not None:
                grad = grad * mask
            adv_images = adv_images.detach() + self.alpha * grad.sign()
            delta = torch.clamp(adv_images - images, min=-eps, max=eps)
            adv_images = torch.clamp(images + delta, min=0, max=1).detach()

        return adv_images


class AttackerMIFGSM:
    def __init__(self, eps=8 / 255, steps=5, decay=1.0, low=0.8, high=1.2, div_prob=0.9, lpips=None, beta=1.0,
                 is_sigmoid=False, targeted=False, device=torch.device('cuda')):
        self.eps = eps
        self.steps = steps
        self.decay = decay
        self.alpha = self.eps / self.steps
        self.low = low
        self.high = high
        self.div_prob = div_prob
        self.lpips = lpips
        self.beta = beta
        self.is_sigmoid = is_sigmoid
        self._targeted = 1.0 if targeted else -1.0
        self.device = device

    def attack(self,model, images, labels, mask=None):
        r"""
        Overridden.
        """
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)
        if self.is_sigmoid:
            loss = nn.BCEWithLogitsLoss()
        else:
            loss = nn.CrossEntropyLoss()
        momentum = torch.zeros_like(images).detach().to(self.device)

        adv_images = images.clone().detach()

        steps = np.random.randint(int(self.steps // 10), self.steps)
        eps = random.uniform(1 / 255, self.eps)

        for i in range(steps):
            adv_images.requires_grad = True
            adv_div = input_diversity(adv_images, div_prob=self.div_prob, low=self.low, high=self.high)
            outputs = model(adv_div)
            if self.is_sigmoid:
                cost = self._targeted * loss(torch.squeeze(outputs, dim=1), labels.float())
            else:
                cost = self._targeted * loss(outputs, labels)
            if self.lpips is not None:
                cost += lpips_loss(self.lpips, images, adv_images) * self.beta

            grad = torch.autograd.grad(cost, adv_images,
                                       retain_graph=False, create_graph=False)[0]

            grad_norm = torch.norm(nn.Flatten()(grad), p=1, dim=1)
            grad = grad / grad_norm.view([-1] + [1] * (len(grad.shape) - 1))
            grad = grad + momentum * self.decay
            momentum = grad
            if mask is not None:
                grad = grad * mask
            adv_images = adv_images.detach() - self.alpha * grad.sign()
            # delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
            delta = torch.clamp(adv_images - images, min=-eps, max=eps)
            adv_images = torch.clamp(images + delta, min=0, max=1).detach()

        return adv_images

