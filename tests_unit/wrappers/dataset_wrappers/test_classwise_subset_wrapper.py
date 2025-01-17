import unittest
from itertools import chain

from kappadata.error_messages import too_little_samples_for_class
from kappadata.utils.class_counts import get_class_counts_from_dataset
from kappadata.wrappers.dataset_wrappers.classwise_subset_wrapper import ClasswiseSubsetWrapper
from kappadata.wrappers.dataset_wrappers.shuffle_wrapper import ShuffleWrapper
from tests_util.datasets.class_dataset import ClassDataset
from tests_util.datasets.classification_dataset import ClassificationDataset


class TestClasswiseSubsetWrapper(unittest.TestCase):
    def test_percent(self):
        dataset = ClassificationDataset(x=list(range(6)), classes=[0, 1, 0, 0, 1, 0])
        subset = ClasswiseSubsetWrapper(dataset=dataset, end_percent=0.5)
        self.assertEqual(3, len(subset))
        items = [subset.getitem_x(i) for i in range(len(subset))]
        self.assertEqual([0, 2, 1], items)

    def test_percent_shuffled(self):
        counts = [100, 150, 250]
        classes = list(chain(*[[i] * count for i, count in enumerate(counts)]))
        dataset = ShuffleWrapper(dataset=ClassDataset(classes=classes), seed=5)
        subset = ClasswiseSubsetWrapper(dataset=dataset, end_percent=0.1)
        self.assertEqual(50, len(subset))
        counts = get_class_counts_from_dataset(subset)
        self.assertEqual([10, 15, 25], counts.tolist())

    def test_index_complete_endidx(self):
        classes = [0, 1, 0, 0, 1, 0]
        dataset = ClassificationDataset(x=list(range(len(classes))), classes=classes)
        subset = ClasswiseSubsetWrapper(dataset=dataset, end_index=2)
        self.assertEqual(4, len(subset))
        items = [subset.getitem_x(i) for i in range(len(subset))]
        self.assertEqual([0, 2, 1, 4], items)

    def test_index_complete_startendidx(self):
        classes = [0, 1, 0, 0, 1, 0]
        dataset = ClassificationDataset(x=list(range(len(classes))), classes=classes)
        subset = ClasswiseSubsetWrapper(dataset=dataset, start_index=1, end_index=2)
        self.assertEqual(2, len(subset))
        items = [subset.getitem_x(i) for i in range(len(subset))]
        self.assertEqual([2, 4], items)

    def test_index_incomplete(self):
        classes = [0, 1, 0, 0, 1, 0]
        dataset = ClassificationDataset(x=list(range(len(classes))), classes=classes)
        with self.assertRaises(AssertionError) as ex:
            ClasswiseSubsetWrapper(dataset=dataset, end_index=3)
        self.assertEqual(too_little_samples_for_class(class_idx=1, actual=2, expected=3), str(ex.exception))
        subset = ClasswiseSubsetWrapper(dataset=dataset, end_index=3, check_enough_samples=False)
        self.assertEqual(5, len(subset))
        items = [subset.getitem_x(i) for i in range(len(subset))]
        self.assertEqual([0, 2, 3, 1, 4], items)
