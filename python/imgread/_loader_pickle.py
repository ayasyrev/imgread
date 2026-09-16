"""Private, configuration-only Loader reconstruction for multiprocessing/pickle."""


def rebuild(paths, color, dtype, backend, limits, max_buffer_bytes):
    from . import Loader

    return Loader(paths, color=color, dtype=dtype, backend=backend,
                  limits=limits, max_buffer_bytes=max_buffer_bytes)
