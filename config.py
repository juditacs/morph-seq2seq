#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017 Judit Acs <judit@sch.bme.hu>
#
# Distributed under terms of the MIT license.

import os
import yaml


class Config(object):
    default_fn = os.path.join('config', 'default.yaml')

    @staticmethod
    def load_defaults():
        with open(Config.default_fn) as f:
            return yaml.load(f)

    @classmethod
    def from_yaml(cls, filename):
        with open(filename) as f:
            params = yaml.load(f)
        return cls(**params)

    def __init__(self, **kwargs):
        defaults = Config.load_defaults()
        for param, val in defaults.items():
            setattr(self, param, val)
        for param, val in kwargs.items():
            setattr(self, param, val)
