"""Optional dependencies; required mode never silently skips missing torch."""
import importlib.util
import multiprocessing as mp
import os
from pathlib import Path
import random
import sys

import numpy as np
import pytest
from PIL import Image

_REQUIRED = os.environ.get("IMGREAD_REQUIRE_TORCH") == "1"
if importlib.util.find_spec("torch") is None or importlib.util.find_spec("torchvision") is None:
    if _REQUIRED:
        raise RuntimeError("IMGREAD_REQUIRE_TORCH=1 needs torch and torchvision")
    pytest.skip("optional torch integration environment", allow_module_level=True)

import torch
from torch.utils.data import DataLoader

# Import the example under a stable name so spawn can reconstruct its classes.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "examples"))
from loader_datasets import IndexedDataset, image_folder, numpy_to_tensor
import imgread


def thread_limits(_worker_id):
    torch.set_num_threads(1)


@pytest.mark.parametrize("method", [None, *mp.get_all_start_methods()])
@pytest.mark.parametrize("warm", [False, True])
def test_path_and_index_datasets(tmp_path, method, warm):
    torch.set_num_threads(1)
    for label in range(2):
        directory = tmp_path / f"class-{label}"
        directory.mkdir()
        for number, suffix in enumerate((".JPEG", ".png", ".tiff")):
            Image.new("RGB", (13, 7), (label * 50 + number, 20, 30)).save(directory / f"{number}{suffix}")
        (directory / "ignored.txt").write_text("not an image")
    path_dataset = image_folder(tmp_path)
    index_dataset = IndexedDataset(path_dataset.samples)
    assert len(path_dataset) == len(index_dataset) == 6
    assert [str(p) for p, _ in path_dataset.samples] == [p for p, _ in index_dataset.samples]
    if warm:
        path_dataset[0]
        index_dataset[0]
    sequential = list(range(6))
    shuffled = sequential.copy()
    random.Random(37).shuffle(shuffled)
    for order in (sequential, shuffled, [2, 0, 2, 5, 1, 1]):
        path_dataset = image_folder(tmp_path)
        index_dataset = IndexedDataset(path_dataset.samples)
        if warm:
            path_dataset[0]
            index_dataset[0]
        expected = [(numpy_to_tensor(imgread.load_numpy(path_dataset.samples[index][0])), path_dataset.samples[index][1]) for index in order]
        for dataset in (path_dataset, index_dataset):
            loader = DataLoader(dataset, sampler=order, batch_size=2,
                                num_workers=0 if method is None else 2,
                                prefetch_factor=None if method is None else 2,
                                persistent_workers=method is not None,
                                multiprocessing_context=method,
                                pin_memory=False, drop_last=False,
                                worker_init_fn=thread_limits, timeout=0 if method is None else 20)
            workers = []
            try:
                pids = None
                for _epoch in range(2):
                    actual = []
                    for images, labels in loader:
                        actual.extend(zip(images, labels.tolist()))
                    assert len(actual) == len(expected)
                    for (image, label), (want, target) in zip(actual, expected):
                        assert label == target
                        np.testing.assert_array_equal(image.numpy(), want.numpy())
                    if method is not None:
                        workers = list(loader._iterator._workers)
                        current = [worker.pid for worker in workers]
                        assert pids is None or current == pids
                        pids = current
                assert dataset[0][1] == path_dataset[0][1]
            finally:
                if loader._iterator is not None:
                    loader._iterator._shutdown_workers()
                for worker in workers:
                    worker.join(5)
                    if worker.is_alive():
                        worker.kill()
                        worker.join(5)
                assert not any(worker.is_alive() for worker in workers)
