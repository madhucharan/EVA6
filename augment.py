import albumentations as A
from albumentations.pytorch import ToTensorV2



def get_train_transform(mu, sigma):

    """
    Args:
        tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
    Returns:
        Tensor: Normalized image.
    """
    train_transform = A.Compose([
                                A.ShiftScaleRotate(),
                                A.CoarseDropout(max_holes=1, 
                                                max_height=16, 
                                                max_width=16, 
                                                min_holes=1, 
                                                min_height=16,
                                                min_width=16,
                                                fill_value=mu),
                                A.Normalize(mean=mu, 
                                            std=sigma),
                                ToTensorV2(),
    ])

    return(train_transform)



def get_test_transform(mu, sigma):
    """
    Args:
        tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
    Returns:
        Tensor: Normalized image.
        """
    test_transform = A.Compose([
                                A.Normalize(mean=(mu), 
                                            std=sigma),
                                ToTensorV2(),
    ])
    return(test_transform)


def no_transform():
    return(A.Compose([A.Normalize()]))