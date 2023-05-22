"""
This file defines miscellanous utilities.
"""
import functools
import importlib
from pathlib import Path
from typing import Any
import importlib.util


def t(duration: float) -> str:
    if duration is None:
        return "n/a"
    if duration < 1:
        return f"{(1000*duration):0.3f}ms"
    elif duration < 60:
        return f"{duration:0.3f}s"
    else:
        return f"{duration//60}min{int(duration%60)}s"


def make_object(object_ref: Any, *args: Any, **kwargs: Any) -> Any:
    modname, qualname_separator, qualname = object_ref.partition(":")
    obj = importlib.import_module(modname)
    if qualname_separator:
        for attr in qualname.split("."):
            obj = getattr(obj, attr)
    return functools.partial(obj, *args, **kwargs)

def make_object2(object_ref: str, *args: Any, **kwargs: Any) -> Any:
    modname, _, classname = object_ref.partition(":")
    # path from modname and parent directory and end with .py
    path = Path(__file__).parent.parent / modname.replace(".", "/")
    path = path.with_suffix(".py")
    spec = importlib.util.spec_from_file_location(modname, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    obj = getattr(module, classname)
    return functools.partial(obj, *args, **kwargs)
