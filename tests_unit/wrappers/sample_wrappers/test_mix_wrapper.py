import unittest

import torch

from kappadata.error_messages import KD_MIX_WRAPPER_REQUIRES_SEED_OR_CONTEXT
from kappadata.wrappers.sample_wrappers.kd_mix_wrapper import KDMixWrapper
from tests_util.datasets import create_image_classification_dataset


class TestOneHotWrapper(unittest.TestCase):
    def test_deterministic(self):
        ds = KDMixWrapper(
            dataset=create_image_classification_dataset(size=5, seed=3),
            mixup_alpha=0.8,
            cutmix_alpha=1.0,
            mixup_p=0.5,
            cutmix_p=0.5,
            seed=12,
        )
        x0 = []
        y0 = []
        for i in range(len(ds)):
            ctx = {}
            x0.append(ds.getitem_x(i, ctx=ctx))
            y0.append(ds.getitem_class(i, ctx=ctx))
        x1 = []
        y1 = []
        for i in range(len(ds)):
            ctx = {}
            x1.append(ds.getitem_x(i, ctx=ctx))
            y1.append(ds.getitem_class(i, ctx=ctx))
        self.assertTrue(torch.all(torch.stack(x0) == torch.stack(x1)))
        self.assertTrue(torch.all(torch.stack(y0) == torch.stack(y1)))

    def test_seed_noctx(self):
        ds = KDMixWrapper(
            dataset=create_image_classification_dataset(size=5, seed=3),
            mixup_alpha=0.8,
            cutmix_alpha=1.0,
            mixup_p=0.5,
            cutmix_p=0.5,
            seed=12,
        )
        x0 = []
        y0 = []
        for i in range(len(ds)):
            ctx = {}
            x0.append(ds.getitem_x(i, ctx=ctx))
            y0.append(ds.getitem_class(i, ctx=ctx))
        x1 = []
        y1 = []
        for i in range(len(ds)):
            x1.append(ds.getitem_x(i))
            y1.append(ds.getitem_class(i))
        self.assertTrue(torch.all(torch.stack(x0) == torch.stack(x1)))
        self.assertTrue(torch.all(torch.stack(y0) == torch.stack(y1)))

    def test_noseed_noctx(self):
        ds = KDMixWrapper(
            dataset=create_image_classification_dataset(size=5, seed=3),
            mixup_alpha=0.8,
            cutmix_alpha=1.0,
            mixup_p=0.5,
            cutmix_p=0.5,
        )
        with self.assertRaises(AssertionError) as ex:
            ds.getitem_x(0)
        self.assertEqual(KD_MIX_WRAPPER_REQUIRES_SEED_OR_CONTEXT, str(ex.exception))