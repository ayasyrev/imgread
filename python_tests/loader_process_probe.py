"""Importable, torch/pytest-free process and lifecycle probes."""
import argparse
import gc
import json
import multiprocessing as mp
import os
import pickle
from pathlib import Path
import sys
import tempfile
import threading
import warnings

import imgread


def state(loader):
    return loader._debug_state() if hasattr(loader, "_debug_state") else {"pid": os.getpid()}


def child_calls(loader, connection):
    try:
        initial = state(loader)
        phases = []
        for _ in range(2):
            assert connection.poll(5), "parent phase timeout"
            assert connection.recv() == "load"
            array = loader[0]
            phases.append(state(loader))
            connection.send({"shape": array.shape, "pixels": array.tobytes().hex(), "state": phases[-1]})
        connection.send({"initial": initial, "phases": phases})
    finally:
        connection.close()


def terminate(process):
    process.join(5)
    if process.is_alive():
        process.terminate()
        process.join(5)
    if process.is_alive():
        process.kill()
        process.join(5)


def process_probe(path, method, warm):
    context = mp.get_context(method)
    loader = imgread.Loader([str(path)])
    expected = imgread.load_numpy(path)
    if warm:
        loader[0]
    before = state(loader)
    parent, child = context.Pipe()
    process = context.Process(target=child_calls, args=(loader, child))
    process.start()
    child.close()
    try:
        results = []
        for _ in range(2):
            parent.send("load")
            assert parent.poll(5), "child phase timeout"
            result = parent.recv()
            assert tuple(result["shape"]) == expected.shape
            assert bytes.fromhex(result["pixels"]) == expected.tobytes()
            assert result["state"]["pid"] == process.pid != os.getpid()
            results.append(result)
        assert parent.poll(5)
        summary = parent.recv()
        assert summary["initial"].get("native_creations", 0) == 0
        assert summary["initial"].get("input_capacity", 0) == 0
        process.join(5)
        assert process.exitcode == 0
        assert state(loader) == before
        assert loader[0].tobytes() == expected.tobytes()
        return {"method": method, "warm": warm, "parent_before": before, "parent_after": state(loader), "child": summary}
    finally:
        parent.close()
        terminate(process)


def pickle_probe(path):
    results = []
    for paths in (None, [], [str(path), str(path)]):
        for warm in (False, True):
            loader = imgread.Loader(paths, color="BGR", dtype="UINT8", backend="IMAGE", limits="UNLIMITED", max_buffer_bytes=12345)
            if warm:
                loader(path)
            before = state(loader)
            for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
                payload = pickle.dumps(loader, protocol=protocol)
                restored = pickle.loads(payload)
                assert restored is not loader and bool(restored)
                assert restored.__reduce_ex__(protocol)[1] == loader.__reduce_ex__(protocol)[1]
                assert state(restored).get("input_capacity", 0) == 0
                assert state(restored).get("native_creations", 0) == 0
                assert restored(path).tobytes() == loader(path).tobytes()
                if paths is not None:
                    assert len(restored) == len(paths)
                results.append({"protocol": protocol, "warm": warm, "paths": paths, "payload_bytes": len(payload)})
            assert before["pid"] == os.getpid()
    return results


def path_outcomes(loader):
    results = []
    for index in range(len(loader)):
        try:
            array = loader[index]
            results.append({"shape": list(array.shape), "pixels": array.tobytes().hex()})
        except OSError as error:
            results.append({"error": type(error).__name__, "errno": error.errno})
    return results


def path_spelling_child(loader, paths, connection):
    try:
        assert loader.__reduce_ex__(4)[1][0] == tuple(paths)
        connection.send({"pid": os.getpid(), "outcomes": path_outcomes(loader)})
    finally:
        connection.close()


def path_spelling_probe(path, method):
    previous = Path.cwd()
    os.chdir(path.parent)
    try:
        paths = [str(path), str(path) + "/", "", "./" + path.name,
                 "././" + path.name, str(path.parent) + "//" + path.name]
        if os.name == "posix":
            raw_name = b"raw-\xff.jpg"
            with open(raw_name, "wb") as stream:
                stream.write(path.read_bytes())
            paths.append(os.fsdecode(raw_name))
        expected = path_outcomes(imgread.Loader(paths))
        assert expected[1]["error"] == "NotADirectoryError"
        assert expected[2]["error"] == "FileNotFoundError"
        for warm in (False, True):
            loader = imgread.Loader(paths)
            if warm:
                loader[0]
            for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
                payload_paths = loader.__reduce_ex__(protocol)[1][0]
                assert all(type(value) is str for value in payload_paths)
                assert payload_paths == tuple(paths)
                restored = pickle.loads(pickle.dumps(loader, protocol))
                assert restored.__reduce_ex__(protocol)[1][0] == tuple(paths)
                assert path_outcomes(restored) == expected
            context = mp.get_context(method)
            parent, child = context.Pipe()
            process = context.Process(target=path_spelling_child, args=(loader, paths, child))
            process.start()
            child.close()
            try:
                assert parent.poll(5), "path spelling child timeout"
                result = parent.recv()
                assert result["pid"] == process.pid != os.getpid()
                assert result["outcomes"] == expected
                process.join(5)
                assert process.exitcode == 0
            finally:
                parent.close()
                terminate(process)
        return {"method": method, "paths": paths, "outcomes": expected}
    finally:
        os.chdir(previous)


def busy(call):
    try:
        call()
    except RuntimeError as error:
        assert str(error) == "Loader is busy", str(error)
    else:
        raise AssertionError("overlap was accepted")


def overlap_probe(path):
    loader = imgread.Loader([str(path)])
    entered, release = threading.Event(), threading.Event()
    errors = []
    class ControlledPath:
        def __fspath__(self):
            entered.set()
            assert release.wait(5)
            return str(path)
    def call():
        try:
            loader(ControlledPath())
        except BaseException as error:
            errors.append(error)
    thread = threading.Thread(target=call)
    thread.start()
    try:
        assert entered.wait(5)
        busy(lambda: loader(path))
        busy(lambda: loader[0])
        if hasattr(loader, "_debug_state"):
            busy(loader._debug_state)
        assert bool(loader) and len(loader) == 1
        pickle.dumps(loader)
        imgread.Loader()(path)
    finally:
        release.set()
        thread.join(5)
    assert not thread.is_alive() and not errors, errors
    class ReentrantPath:
        def __fspath__(self):
            busy(lambda: loader(path))
            return str(path)
    class ReentrantIndex:
        def __index__(self):
            busy(lambda: loader[0])
            return 0
    loader(ReentrantPath())
    loader[ReentrantIndex()]
    warning_loader = imgread.Loader(backend="turbojpeg")
    png = Path(path).with_suffix(".png")
    from PIL import Image
    Image.new("RGB", (3, 2)).save(png)
    with warnings.catch_warnings():
        warnings.simplefilter("always")
        def showwarning(*args, **kwargs):
            busy(lambda: warning_loader(png))
        warnings.showwarning = showwarning
        warning_loader(png)
    loader(path)
    return {"overlap": "passed", "pid": os.getpid()}


def fifo_probe(path):
    fifo = Path(path).with_suffix(".fifo")
    os.mkfifo(fifo)
    loader = imgread.Loader()
    errors = []
    def decode():
        try:
            loader(fifo)
        except BaseException as error:
            errors.append(error)
    thread = threading.Thread(target=decode, daemon=True)
    thread.start()
    # Opening the writer proves the detached reader has opened the FIFO; retaining
    # the writer without bytes holds read() deterministically until the busy check.
    writer = os.open(fifo, os.O_WRONLY)
    try:
        busy(lambda: loader(path))
        data = Path(path).read_bytes()
        while data:
            count = os.write(writer, data)
            data = data[count:]
    finally:
        os.close(writer)
        thread.join(5)
        fifo.unlink()
    assert not thread.is_alive() and not errors, errors
    loader(path)
    return {"fifo": "passed"}


def constructor_no_io(root):
    class PathValue:
        def __init__(self, value):
            self.value = value
        def __fspath__(self):
            return self.value
    paths = [root + "/a.jpg", PathValue(root + "/b.png")]
    loader = imgread.Loader(paths)
    assert len(loader) == 2
    return {"entries": len(loader)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["process", "pickle", "path-spelling", "overlap", "fifo", "constructor-no-io", "import-no-torch"])
    parser.add_argument("--method", choices=mp.get_all_start_methods())
    parser.add_argument("--warm", action="store_true")
    parser.add_argument("--paths-root", default="/tmp/imgread-loader-noio-sentinel")
    args = parser.parse_args()
    if args.mode == "constructor-no-io":
        result = constructor_no_io(args.paths_root)
    else:
        if args.mode == "import-no-torch":
            class BlockTorch:
                def find_spec(self, fullname, *args):
                    if fullname.split(".")[0] in ("torch", "torchvision"):
                        raise AssertionError("torch import attempted")
            sys.meta_path.insert(0, BlockTorch())
            # Verify a fresh imgread import, including its native module.
            for key in list(sys.modules):
                if key == "imgread" or key.startswith("imgread."):
                    del sys.modules[key]
            __import__("imgread")
        from PIL import Image
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "unicode-изображение.jpg"
            Image.new("RGB", (13, 7), (10, 50, 100)).save(path)
            if args.mode == "process":
                result = process_probe(path, args.method, args.warm)
            elif args.mode == "pickle":
                result = pickle_probe(path)
            elif args.mode == "path-spelling":
                result = path_spelling_probe(path, args.method)
            elif args.mode == "overlap":
                result = overlap_probe(path)
            elif args.mode == "fifo":
                result = fifo_probe(path)
            else:
                assert imgread.Loader()(path).shape == (7, 13, 3)
                result = {"import_no_torch": "passed"}
    print(json.dumps(result))


if __name__ == "__main__":
    main()
