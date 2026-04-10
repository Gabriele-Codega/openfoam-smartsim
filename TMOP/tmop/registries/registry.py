from typing import Any

class Registry:
    def __init__(self):
        self._data: dict[str, Any] = {}

    def register(self):
        def decorator(item):
            self[item.__name__] = item
            return item
        return decorator

    def __setitem__(self, key: str, value: Any):
        if key in self._data:
            raise ValueError(f"Key {key} already registered.")
        self._data[key] = value

    def __getitem__(self, key: str):
        try:
            return self._data[key]
        except KeyError as e:
            e.add_note(f"Invalid key '{key}'. Possible values are {list(self._data.keys())}.")
            raise e
