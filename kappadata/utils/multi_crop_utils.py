import torch
from collections import defaultdict


class MultiCropJointForwardModule(torch.nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, *args, **kwargs):
        return multi_crop_joint_forward(self.module, *args, **kwargs)

class MultiCropSplitForwardModule(torch.nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, *args, **kwargs):
        return multi_crop_split_forward(self.module, *args, **kwargs)


def multi_crop_joint_forward(model, *args, **kwargs):
    # concat if input is list/tuple
    if isinstance(x, (list, tuple)):
        n_chunks = len(x)
        x = torch.concat(x)
    else:
        n_chunks = None

    # joint forward
    y = model(x)

    # chunk if input was list/tuple
    if n_chunks is not None:
        return y.chunk(n_chunks)
    return y

def concat_same_shapes(*args, **kwargs):
    # collect same shapes
    concat_args = []
    for i in range(len(args)):
        if isinstance(args[i], (list, tuple)) and torch.is_tensor(args[i][0]):
            items_by_shape = defaultdict(list)
            for item in args[i]:
                items_by_shape[item.shape].append(item)
            
            if len(items_by_shape) == 1:
                concat_args.append(torch.concat(list(items_by_shape.values())[0]))
            else:

        else:
            concat_args.append(args[i])




def multi_crop_split_forward(model, x, n_chunks=None):
    # chunk if input is tensor
    if torch.is_tensor(x):
        assert n_chunks is not None and len(x) % n_chunks == 0
        x = x.chunk(n_chunks)
    else:
        assert n_chunks is None

    # split forward
    results = []
    for chunk in x:
        results.append(model(chunk))

    # concat if input was tensor
    if n_chunks is not None:
        results = torch.concat(results)

    return results