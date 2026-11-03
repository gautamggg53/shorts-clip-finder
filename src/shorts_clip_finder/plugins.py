"""Optional pluggable scorers (for example, your own LLM).

A scorer is any callable ``scorer(candidate) -> float`` returning a value in ``[0, 1]``, or
an object with such a ``score`` method. Load one from the CLI with
``--scorer my_module:my_scorer`` or ``--scorer path/to/file.py:MyScorer`` (classes are
instantiated with no arguments). This package never calls a network API itself.
"""

from __future__ import annotations

import importlib
import importlib.util
import inspect
from collections.abc import Callable
from pathlib import Path
from typing import Protocol, runtime_checkable

from .models import Candidate


@runtime_checkable
class ClipScorer(Protocol):
    def score(self, candidate: Candidate) -> float: ...


ScoreFn = Callable[[Candidate], float]


def as_score_fn(obj: object) -> ScoreFn:
    if inspect.isclass(obj):
        obj = obj()
    if isinstance(obj, ClipScorer):
        return obj.score
    if callable(obj):
        return obj  # type: ignore[return-value]
    raise TypeError(f"{obj!r} is not callable and has no score() method")


def load_scorer(spec: str) -> ScoreFn:
    """Load ``module:attr`` or ``file.py:attr``."""
    if ":" not in spec:
        raise ValueError("scorer must look like 'module:attr' or 'file.py:attr'")
    target, attr = spec.rsplit(":", 1)
    if target.endswith(".py"):
        path = Path(target)
        if not path.is_file():
            raise FileNotFoundError(path)
        mod_spec = importlib.util.spec_from_file_location(f"_scf_scorer_{path.stem}", path)
        if mod_spec is None or mod_spec.loader is None:
            raise ImportError(f"cannot load {path}")
        module = importlib.util.module_from_spec(mod_spec)
        mod_spec.loader.exec_module(module)
    else:
        module = importlib.import_module(target)
    return as_score_fn(getattr(module, attr))
