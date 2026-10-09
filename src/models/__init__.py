from src.models.lenet import MNISTLeNet5


def get_model(dataset: str, model_name: str):
    """The paper's architectures (app:exp_spectrum): LeNet-5 on MNIST, WRN-16-4 (GroupNorm) on CIFAR-10,
    ResNet-18 (GroupNorm) on CelebA and ImageNette."""
    if dataset == "mnist" and model_name == "lenet5":
        return MNISTLeNet5()
    if dataset == "cifar10" and model_name == "wrn16_4_gn":
        from src.models.resnet_gn import wrn16_4_gn
        return wrn16_4_gn(10)
    if dataset == "celeba" and model_name == "resnet18gn":
        from src.models.resnet_gn import resnet18_gn
        return resnet18_gn(2)                 # binary Male attribute
    if dataset == "imagenette" and model_name == "resnet18gn":
        from src.models.resnet_gn import resnet18_gn
        return resnet18_gn(10)
    raise ValueError(f"unknown (dataset, model): ({dataset!r}, {model_name!r})")
