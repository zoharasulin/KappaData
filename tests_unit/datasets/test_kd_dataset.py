import unittest

from kappadata.errors import UseModeWrapperException
from kappadata.error_messages import getshape_instead_of_getdim
from tests_util.datasets.index_dataset import IndexDataset
from tests_util.datasets.class_dataset import ClassDataset
from kappadata.datasets.kd_dataset import KDDataset


class TestKDDataset(unittest.TestCase):
    def test_dispose(self):
        ds = IndexDataset(size=3)
        ds.dispose()
        self.assertTrue(ds.disposed)

    def test_dispose_context_manager(self):
        with IndexDataset(size=3) as ds:
            pass
        self.assertTrue(ds.disposed)

    def test_root_dataset(self):
        ds = IndexDataset(size=3)
        self.assertEqual(ds, ds.root_dataset)

    def test_getitem(self):
        ds = IndexDataset(size=3)
        with self.assertRaises(UseModeWrapperException):
            _ = ds[0]

    def test_has_wrapper(self):
        root_ds = IndexDataset(size=3)
        self.assertFalse(root_ds.has_wrapper(None))

    def test_has_wrapper_type(self):
        root_ds = IndexDataset(size=3)
        self.assertFalse(root_ds.has_wrapper_type(None))

    def test_all_wrappers(self):
        root_ds = IndexDataset(size=3)
        self.assertEqual([], root_ds.all_wrappers)

    def test_all_wrapper_types(self):
        root_ds = IndexDataset(size=3)
        self.assertEqual([], root_ds.all_wrapper_types)

    def test_getshape(self):
        root_ds = ClassDataset(classes=[0, 1, 2])
        self.assertEqual((3,), root_ds.getshape("class"))

    def test_getdim(self):
        root_ds = ClassDataset(classes=[0, 1, 2])
        self.assertEqual(3, root_ds.getdim("class"))
        self.assertEqual(3, root_ds.getdim_class())

    def test_detect_getdim_implementation(self):
        class GetdimDataset(KDDataset):
            def __len__(self):
                raise RuntimeError

            @staticmethod
            def getdim_class():
                return 3

        with self.assertRaises(AssertionError) as ex:
            GetdimDataset()
        self.assertEqual(getshape_instead_of_getdim(["getdim_class"]), str(ex.exception))