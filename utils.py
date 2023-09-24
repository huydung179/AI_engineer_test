import numpy as np
import torch
from torchvision import models


def freeze_layers(model: torch.nn.Module) -> None:
    """
    Freeze all parameters of the model.
    
    :param model: PyTorch model whose parameters will be frozen.
    """
    for param in model.parameters():
        param.requires_grad = False


def initialize_mobilenet(model_name: str,
                         pretrained: bool,
                         training: bool,
                         num_output: int) -> (torch.nn.Module, torch.nn.parameter.Parameter):

    model = torch.hub.load('pytorch/vision:v0.10.0',
                           model_name, pretrained=pretrained)
    if training:
        freeze_layers(model)
    model.classifier[1] = torch.nn.Linear(
        model.classifier[1].in_features, num_output)
    return model, model.classifier.parameters()


def initialize_resnet(model_name: str,
                      pretrained: bool,
                      training: bool,
                      num_output: int) -> (torch.nn.Module, torch.nn.parameter.Parameter):

    model = torch.hub.load('pytorch/vision:v0.10.0',
                           model_name, pretrained=pretrained)
    if training:
        freeze_layers(model)
    model.fc = torch.nn.Linear(model.fc.in_features, num_output)
    return model, model.fc.parameters()


def initialize_squeezenet(model_name: str,
                          pretrained: bool,
                          training: bool,
                          num_output: int) -> (torch.nn.Module, torch.nn.parameter.Parameter):

    model = torch.hub.load('pytorch/vision:v0.10.0',
                           model_name, pretrained=pretrained)
    if training:
        freeze_layers(model)
    model.classifier[1] = torch.nn.Conv2d(
        512, num_output, kernel_size=(1, 1), stride=(1, 1))
    return model, model.classifier.parameters()


def initialize_effnet(model_name: str,
                      pretrained: bool,
                      training: bool,
                      num_output: int
                      ) -> (torch.nn.Module, torch.nn.parameter.Parameter):

    model = torch.hub.load(
        'NVIDIA/DeepLearningExamples:torchhub', model_name, pretrained=pretrained)
    if training:
        freeze_layers(model)
    model.classifier.fc = torch.nn.Linear(
        model.classifier.fc.in_features, num_output)
    return model, model.classifier.fc.parameters()


def initialize_from_torchvision(model_name: str,
                                weights: str,
                                training: bool,
                                num_output: int,
                                update_layer: str) -> (torch.nn.Module, torch.nn.parameter.Parameter):

    model = getattr(models, model_name)(weights=weights)
    if training:
        freeze_layers(model)

    # Split the update_layer into a list of attribute names
    update_layers = update_layer.split('.')

    # Use a loop to access nested attributes
    current_layer = model
    for attr_name in update_layers[:-1]:
        current_layer = getattr(current_layer, attr_name)

    # Replace the attribute with a new Linear layer
    setattr(current_layer, update_layers[-1], torch.nn.Linear(
        getattr(current_layer, update_layers[-1]).in_features, num_output))

    return model, getattr(current_layer, update_layers[-1]).parameters()


def load_model(model_name: str, num_output: int, pretrained: bool = True, training: bool = True) -> (torch.nn.Module, torch.nn.parameter.Parameter):
    """
    Load a specific model with the option for pretrained weights and for a specified output size.
    
    :param model_name: Name of the model.
    :param num_output: Number of output features or classes.
    :param pretrained: Boolean, whether to use pretrained weights.
    :param training: Boolean, whether the model is being used for training.
    :return: Loaded model and parameters to optimize.
    """

    models_dict = {
        "mobilenet_v2": initialize_mobilenet,
        "squeezenet1_0": initialize_squeezenet,
        "squeezenet1_1": initialize_squeezenet,
        "nvidia_efficientnet_b4": initialize_effnet,
        "nvidia_efficientnet_widese_b4": initialize_effnet,
        "maxvit_t": lambda mn, pt, tr, no: initialize_from_torchvision(mn, pt, tr, no, 'classifier.5'),
        "swin_v2_t": lambda mn, pt, tr, no: initialize_from_torchvision(mn, pt, tr, no, 'head'),
        "vit_b_16": lambda mn, pt, tr, no: initialize_from_torchvision(mn, pt, tr, no, 'heads.head'),
        "resnet18": initialize_resnet,
        "resnet50": initialize_resnet,
        "resnext50_32x4d": initialize_resnet,
        "resnext101_32x8d": initialize_resnet,
    }

    if model_name in models_dict:
        return models_dict[model_name](model_name, pretrained, training, num_output)
    else:
        raise ValueError("Model name not defined")


def batch_mixup(x: torch.Tensor, y: torch.Tensor, alpha: float = 0.3) -> (torch.Tensor, torch.Tensor):
    """
    Apply mixup augmentation to a given batch of data.
    
    :param x: Tensor, input data.
    :param y: Tensor, labels.
    :param alpha: Beta distribution alpha parameter for mixup.
    :return: Mixed inputs and corresponding mixed labels.
    """

    lam = torch.from_numpy(np.random.beta(
        alpha, alpha, x.size()[0])).float().to(x.device)
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)
    mixed_x = lam.unsqueeze(1).unsqueeze(2).unsqueeze(
        3) * x + (1 - lam.unsqueeze(1).unsqueeze(2).unsqueeze(3)) * x[index, :]
    mixed_y = lam * y + (1 - lam) * y[index]
    return mixed_x, mixed_y
