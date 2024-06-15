import torch
import torch.nn as nn

def count_conv2d_flops(layer, input, output):
    """
    """
    _, in_channels, in_h, in_w = input[0].size()
    out_channels, _, k_h, k_w = layer.weight.size()
    out_h, out_w = output[0].size()[2:]

    flops_per_instance = (2 * in_channels * k_h * k_w - 1) 
    num_instances = out_channels * out_h * out_w
    total_flops = flops_per_instance * num_instances

    return total_flops

def count_linear_flops(layer, input, output):
    """
    """
    in_features = input[0].size()[1]
    out_features = output[0].size()[1]

    total_flops = 2 * in_features * out_features

    return total_flops

def count_bn_flops(layer, input, output):
    """
    """
    total_flops = 2 * input[0].numel()
    return total_flops

def count_relu_flops(layer, input, output):
    """
    """
    total_flops = input[0].numel()
    return total_flops

def count_pool_flops(layer, input, output):
    """
    """
    _, in_channels, in_h, in_w = input[0].size()
    out_h, out_w = output[0].size()[2:]

    kernel_size = layer.kernel_size
    if isinstance(kernel_size, tuple):
        k_h, k_w = kernel_size
    else:
        k_h = k_w = kernel_size

    total_flops = in_channels * out_h * out_w * k_h * k_w 

    return total_flops

def count_model_flops(model, input_size):
    """
    """
    hooks = []
    total_flops = []

    def register_hooks(module):
        if isinstance(module, nn.Conv2d):
            hooks.append(module.register_forward_hook(lambda layer, inp, out: total_flops.append(count_conv2d_flops(layer, inp, out))))
        elif isinstance(module, nn.Linear):
            hooks.append(module.register_forward_hook(lambda layer, inp, out: total_flops.append(count_linear_flops(layer, inp, out))))
        elif isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.BatchNorm1d):
            hooks.append(module.register_forward_hook(lambda layer, inp, out: total_flops.append(count_bn_flops(layer, inp, out))))
        elif isinstance(module, nn.ReLU):
            hooks.append(module.register_forward_hook(lambda layer, inp, out: total_flops.append(count_relu_flops(layer, inp, out))))
        elif isinstance(module, nn.MaxPool2d) or isinstance(module, nn.AvgPool2d):
            hooks.append(module.register_forward_hook(lambda layer, inp, out: total_flops.append(count_pool_flops(layer, inp, out))))

    model.apply(register_hooks)

    input_data = torch.randn(1, *input_size)
    with torch.no_grad():
        model(input_data)

    for hook in hooks:
        hook.remove()

    total_flops_sum = sum(total_flops)
    gflops = total_flops_sum / 1e9

    return gflops