from typing import Callable, Dict, List, Tuple

PracticeFn = Callable[[], None]
_REGISTRY: Dict[str, Tuple[str, PracticeFn]] = {}
# id -> (título legible, función sin args)

def register(pid: str, title: str):
    def deco(fn: PracticeFn):
        _REGISTRY[pid] = (title, fn)
        return fn
    return deco

def all_practices() -> List[Tuple[str, str, PracticeFn]]:
    # ordena por id (p01, p02, ...)
    return [(pid, title, fn) for pid, (title, fn) in sorted(_REGISTRY.items())]
