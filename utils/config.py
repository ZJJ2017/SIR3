import collections.abc as collections_abc
import os.path as osp

from addict import Dict
import yaml


class ConfigDict(Dict):
    def __missing__(self, name):
        raise KeyError(name)

    def __getattr__(self, name):
        try:
            value = super(ConfigDict, self).__getattr__(name)
        except KeyError:
            ex = AttributeError(
                "'{}' object has no attribute '{}'".format(
                    self.__class__.__name__, name
                )
            )
        except Exception as e:
            ex = e
        else:
            return value
        raise ex

    def __setitem__(self, name, value):
        if isinstance(value, dict):
            value = ConfigDict(value)

        super(ConfigDict, self).__setitem__(name, value)


class YamlConfig:
    """Manager of ConfigDict from yaml."""

    # def __init__(self, config_paths: dict):
    #     """Make ConfigDict from yaml path."""
    #     self.cfg = ConfigDict()
    #     for key, path in config_paths.items():
    #         self.cfg[key] = self._yaml_to_config_dict(path)
    def __init__(self, config_paths):
        self.cfg = self._yaml_to_config_dict(config_paths)

    @staticmethod
    def _yaml_to_config_dict(path: str) -> ConfigDict:
        """Return ConfigDict from yaml."""
        try:
            with open(path) as f:
                data = yaml.load(f, Loader=yaml.FullLoader)
        except FileNotFoundError:
            with open(osp.expanduser(path)) as f:
                data = yaml.load(f, Loader=yaml.FullLoader)
        return ConfigDict(data)

    def get_config_dict(self):
        return self.cfg
