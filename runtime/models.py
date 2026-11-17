"""Process-local model ownership for an explicitly enabled resident worker."""

from threading import RLock

_enabled = False
_models = {}
_loads = {}
_lock = RLock()


def enable():
    global _enabled
    with _lock:
        _enabled = True


def is_enabled():
    return _enabled


def get(key, loader):
    if not _enabled:
        return loader()
    with _lock:
        if key not in _models:
            _models[key] = loader()
            _loads[key] = _loads.get(key, 0) + 1
        return _models[key]


def info():
    with _lock:
        return {"enabled": _enabled, "model_count": len(_models), "load_count": sum(_loads.values()),
                "models": [{"key": list(key), "loads": _loads[key]} for key in _models]}
