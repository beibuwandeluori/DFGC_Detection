from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2


def build_transforms(height, width, max_pixel_value=255.0, norm_mean=[0.485, 0.456, 0.406],
                     norm_std=[0.229, 0.224, 0.225], **kwargs):
    """Builds train and test transform functions.

    Args:
        height (int): target image height.
        width (int): target image width.E
        norm_mean (list or None, optional): normalization mean values. Default is ImageNet means.
        norm_std (list or None, optional): normalization standard deviation values. Default is
            ImageNet standard deviation values.
        max_pixel_value (float): max pixel value
    """

    if norm_mean is None or norm_std is None:
        norm_mean = [0.485, 0.456, 0.406] # imagenet mean
        norm_std = [0.229, 0.224, 0.225] # imagenet std
    normalize = transforms.Normalize(mean=norm_mean, std=norm_std)

    train_transform = A.Compose([
        # A.ChannelShuffle(p=0.1),
        A.HorizontalFlip(),
        A.GaussNoise(p=0.1),
        A.GaussianBlur(p=0.1),
        # A.HueSaturationValue(p=0.1),
        # A.IAAAdditiveGaussianNoise(p=0.1),
        # A.IAASharpen(p=0.5),
        # A.ISONoise(p=0.3),
        # A.RandomBrightness(p=0.8),
        # A.RandomBrightnessContrast(p=0.2),
        # A.ToSepia(p=0.1),
        A.Resize(height, width),
        A.Normalize(mean=norm_mean, std=norm_std, max_pixel_value=max_pixel_value),
        ToTensorV2(),
    ])

    test_transform = A.Compose([
        A.Resize(height, width),
        A.Normalize(mean=norm_mean, std=norm_std, max_pixel_value=max_pixel_value),
        ToTensorV2(),
    ])

    return train_transform, test_transform






