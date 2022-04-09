import torch.nn as nn

def get_activation_function(func_name: str):
    if func_name == 'relu':
        return nn.ReLU()
    elif func_name == 'gelu':
        return nn.GELU()
    elif func_name == 'leaky_relu':
        return nn.LeakyReLU()
    else:
        raise NotImplementedError(f"The Activation Function {func_name} Not Support!")

