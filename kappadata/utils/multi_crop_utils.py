import torch
from collections import defaultdict
from itertools import chain

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
    concat_args_idxs = defaultdict(lambda: defaultdict(list))
    for i in range(len(args)):
        if isinstance(args[i], (list, tuple)) and torch.is_tensor(args[i][0]):
            items_by_shape = defaultdict(list)
            for j, item in enumerate(args[i]):
                items_by_shape[item.shape].append(item)
                concat_args_idxs[i][item.shape].append(j)
            
            if len(items_by_shape) == 1:
                concat_args.append(torch.concat(list(items_by_shape.values())[0]))
            else:
                concat_args.append([torch.concat(values) for values in items_by_shape.values()])

        else:
            concat_args.append(args[i])

    # flatten indices
    concat_args_idxs_flat = {}
    for k, v in concat_args_idxs.items():
        concat_args_idxs_flat[k] = [vv for vv in v.values()]

    return concat_args, concat_args_idxs_flat, kwargs

def split_same_shapes(concat_args, concat_args_idxs, kwargs):
    # reverse the concat_same_shapes operation
    split_args = []
    for arg_idx in range(len(concat_args)):
        idxs = concat_args_idxs[arg_idx]
        results = []
        # chunk concated tensor
        for group_idx in range(len(concat_args[arg_idx])):
            results += concat_args[arg_idx][group_idx].chunk(len(idxs[group_idx]))
        # unshuffle
        flat_idxs = list(chain(*idxs))
        if len(flat_idxs) == 1:
            split_args.append(results[0])
        else:
            split_args.append([results[i] for i in flat_idxs])

    # TODO kwargs
    return split_args, kwargs


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