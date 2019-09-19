import os
import pathlib
from functools import update_wrapper
from typing import Any, Callable, Dict, Generic, Sequence, Type, TypeVar

import yaml
from dataclasses import dataclass

T = TypeVar("T")
R = TypeVar("R")


class cached_getter(Generic[T, R]):
    """
    The cached getter calls the wrapped function like a property but replaces
    itself in the owner instance after the first access.
    """

    def __init__(self, fget: Callable[[T], R]) -> None:
        self.fget = fget
        update_wrapper(self, fget)

    def __get__(self, instance: T, owner: Type[T]) -> R:
        if instance is None:
            return self

        value = self.fget(instance)
        setattr(instance, self.fget.__name__, value)
        return value


def format_duration(d: float) -> str:
    """
    Format a duration given in seconds.
    """
    m, s = divmod(d, 60)
    h, m = map(int, divmod(m, 60))
    if h:
        return f"{h}h{m}m{s:.2f}s"
    return f"{m}m{s:.2f}s"


def get_path_kwargs_from_env() -> Dict[str, pathlib.Path]:
    """
    Get path kwargs for the database from environment.
    """
    keys = ((k, f"STPLFCNN_{k.upper()}") for k in ["base_path", "params_path", "models_path"])
    return {k: pathlib.Path(os.environ[env_k]) for k, env_k in keys if env_k in os.environ}


@dataclass
class Reference:
    name: str


class ReferenceLoader(yaml.SafeLoader):
    """
    Load Reference objects from !ref tags.
    """


def ref_constructor(loader: yaml.BaseLoader, node: yaml.Node) -> Reference:
    return Reference(loader.construct_scalar(node))


ReferenceLoader.add_constructor("!ref", ref_constructor)


class ReferenceDumper(yaml.SafeDumper):
    """
    Dump Reference objects as !ref tags.
    """


def ref_representer(dumper: yaml.BaseDumper, ref: Reference) -> yaml.Node:
    return dumper.represent_scalar("!ref", ref.name)


ReferenceDumper.add_representer(Reference, ref_representer)


class HyperoptLoader(yaml.SafeLoader):
    """
    Loader with Hyperopt functions using tags with '!hp.' prefix.
    """


def hyperopt_constructor(loader: yaml.BaseLoader, suffix: str, node: yaml.Node):
    if suffix not in [
        "choice", "pchoice",
        "normal", "qnormal",
        "lognormal", "qlognormal",
        "uniform", "quniform", "shiftedquniform",
        "loguniform", "qloguniform",
        "randint",
    ]:
        raise ValueError(f"{suffix} is not a valid function")

    from hyperopt import hp
    from hyperopt.pyll import scope
    loader.hp_label_inc = getattr(loader, "hp_label_inc", -1) + 1
    label = f"label{loader.hp_label_inc}"
    func = getattr(hp, suffix, None)
    if suffix == "choice" or suffix == "pchoice":
        return func(label, loader.construct_sequence(node, deep=True))
    if suffix == "shiftedquniform":
        # shift hp.uniform so that low is 0 and shift back after rounding
        kwargs = _construct_hyperopt_params(loader, node, ["low", "high", "q"])
        low = kwargs["low"]
        kwargs["low"] = 0
        kwargs["high"] -= low
        apply = getattr(hp, "quniform")(label, **kwargs) + low
        if isinstance(kwargs["q"], int) and isinstance(low, int):
            # convert to int if low and q are ints
            apply = scope.int(apply)
        return apply
    if suffix[0] == "q":
        if suffix == "quniform" or suffix == "qloguniform":
            args_order = ["low", "high", "q"]
        else:
            args_order = ["mu", "sigma", "q"]
        kwargs = _construct_hyperopt_params(loader, node, args_order)
        if isinstance(kwargs["q"], int):
            # convert to int if q is an int
            return scope.int(func(label, **kwargs))
        return func(label, **kwargs)
    if isinstance(node, yaml.SequenceNode):
        return func(label, *loader.construct_sequence(node, deep=True))
    else:
        return func(label, **loader.construct_mapping(node, deep=True))


def _construct_hyperopt_params(
        loader: yaml.BaseLoader,
        node: yaml.Node,
        args_order: Sequence[str],
) -> Dict[str, Any]:
    if isinstance(node, yaml.SequenceNode):
        values = loader.construct_sequence(node, deep=True)
        return dict((k, v) for k, v in zip(args_order, values))
    else:
        return loader.construct_mapping(node, deep=True)


HyperoptLoader.add_multi_constructor("!hp.", hyperopt_constructor)
