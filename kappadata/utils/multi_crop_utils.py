import torch
from collections import defaultdict


class MultiCropSplitForwardModule(torch.nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, *args, **kwargs):
        return multi_crop_split_forward(self.module, *args, **kwargs)


def multi_crop_split_forward(model, x, batch_size=None):
    # chunk if input is tensor
    if torch.is_tensor(x):
        assert batch_size is not None and len(x) % batch_size == 0
        x = x.chunk(len(x) // batch_size)
    else:
        assert isinstance(x, (list, tuple)) and batch_size is None

    # split forward
    results = []
    for chunk in x:
        results.append(model(chunk))

    # concat if input was tensor
    if batch_size is not None:
        results = torch.concat(results)

    return results

# TODO write unittests
def concat_same_shape_inputs(x):
    if torch.is_tensor(x):
        return [x], len(x)
    results = defaultdict(list)
    for xx in x:
        results[tuple(xx.shape[1:])].append(xx)
    return [torch.concat(v) for v in results.values()], len(x[0])