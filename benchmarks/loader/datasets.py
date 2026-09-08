"""Importable torch adapters; imported only by pipeline/memory child processes."""
from torch.utils.data import IterableDataset
from run import SequenceSource


class SequenceDataset(SequenceSource, IterableDataset):
    pass
