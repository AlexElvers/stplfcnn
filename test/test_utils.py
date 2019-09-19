import pathlib

import yaml
from hyperopt import pyll

from stplfcnn.utils import HyperoptLoader, Reference, ReferenceDumper, ReferenceLoader, get_path_kwargs_from_env


def test_get_path_kwargs_from_env(monkeypatch):
    monkeypatch.setenv("STPLFCNN_BASE_PATH", "base")
    monkeypatch.setenv("STPLFCNN_PARAMS_PATH", "params")
    monkeypatch.setenv("STPLFCNN_MODELS_PATH", "models")
    assert get_path_kwargs_from_env() == dict(
        base_path=pathlib.Path("base"),
        params_path=pathlib.Path("params"),
        models_path=pathlib.Path("models"),
    )


class TestReferenceLoader:
    def test_load(self):
        result = yaml.load("!ref foo", ReferenceLoader)
        assert isinstance(result, Reference)
        assert result.name == "foo"

    def test_dump(self):
        result = yaml.dump(Reference("foo"), Dumper=ReferenceDumper)
        assert result == "!ref 'foo'\n"


class TestHyperoptLoader:
    def test_load(self):
        result = yaml.load("!hp.uniform [0, 1]", HyperoptLoader)
        assert isinstance(result, pyll.Apply)
        assert result.pos_args[0].pos_args[0]._obj == "label0"
        assert result.pos_args[0].pos_args[1].pos_args[0]._obj == 0
        assert result.pos_args[0].pos_args[1].pos_args[1]._obj == 1

        # choice and pchoice are special cases
        result = yaml.load("!hp.choice [a, b, c]", HyperoptLoader)
        assert isinstance(result, pyll.Apply)
        assert result.pos_args[0].pos_args[0]._obj == "label0"
        assert result.pos_args[1]._obj == "a"
        assert result.pos_args[2]._obj == "b"
        assert result.pos_args[3]._obj == "c"

        # check label increment
        result = yaml.load("[!hp.uniform [0, 1], !hp.normal [0, 1]]", HyperoptLoader)
        assert result[0].pos_args[0].pos_args[0]._obj == "label0"
        assert result[1].pos_args[0].pos_args[0]._obj == "label1"
