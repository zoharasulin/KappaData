import torch
import torch.nn as nn
import unittest
from kappadata.utils.multi_crop_utils import (
    multi_crop_split_forward,
    multi_crop_joint_forward,
    MultiCropSplitForwardModule,
    MultiCropJointForwardModule,
    concat_same_shapes,
    split_same_shapes,
)
from tests_util.modules.memorize_shape_module import MemorizeShapeModule

class TestMultiCropUtils(unittest.TestCase):
    class MultiCropModule(nn.Module):
        def __init__(self, n_views=None):
            super().__init__()
            self.n_views = n_views
            self.layer0 = MemorizeShapeModule(nn.Linear(4, 8, bias=False))
            self.layer1 = MemorizeShapeModule(nn.Linear(8, 6, bias=False))

        def forward(self, x):
            x = multi_crop_joint_forward(self.layer0, x)
            x = multi_crop_split_forward(self.layer1, x, n_chunks=self.n_views)
            return x

    def test_tensor_2views(self):
        model = self.MultiCropModule(n_views=2)
        model(torch.ones(10, 4))
        self.assertEqual((10, 4), model.layer0.shapes[0])
        self.assertEqual((5, 8), model.layer1.shapes[0])
        self.assertEqual((5, 8), model.layer1.shapes[1])

    def test_list_2views(self):
        model = self.MultiCropModule()
        model(torch.ones(10, 4).chunk(2))
        self.assertEqual((10, 4), model.layer0.shapes[0])
        self.assertEqual((5, 8), model.layer1.shapes[0])
        self.assertEqual((5, 8), model.layer1.shapes[1])

    def test_list_2views_sequential(self):
        model = nn.Sequential(
            MultiCropJointForwardModule(MemorizeShapeModule(nn.Linear(4, 8, bias=False))),
            MultiCropSplitForwardModule(MemorizeShapeModule(nn.Linear(8, 6, bias=False))),
        )
        model(torch.ones(10, 4).chunk(2))
        self.assertEqual((10, 4), model[0].module.shapes[0])
        self.assertEqual((5, 8), model[1].module.shapes[0])
        self.assertEqual((5, 8), model[1].module.shapes[1])

    def test_concat_same_shape_unshuffle(self):
        rng = torch.Generator().manual_seed(5)
        x0 = torch.randn(3, 2, generator=rng)
        x1 = torch.randn(3, 3, generator=rng)
        x2 = torch.randn(3, 2, generator=rng)
        args, args_idxs, kwargs = concat_same_shapes([x0, x1, x2])
        result_args, result_kwargs = split_same_shapes(args, args_idxs, kwargs)
        self.assertEquals(x0.tolist(), result_args[0][0].tolist())
        self.assertEquals(x1.tolist(), result_args[0][1].tolist())
        self.assertEquals(x2.tolist(), result_args[0][2].tolist())
        # TODO kwargs