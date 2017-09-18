#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017 Judit Acs <judit@sch.bme.hu>
#
# Distributed under terms of the MIT license.

import os
import yaml


class ConfigError(Exception):
    pass


class Config(object):
    __slots__ = (
        'train_file', 'dev_file', 'test_file', 'share_vocab', 'src_maxlen',
        'tgt_maxlen', 'src_embedding_dim', 'tgt_embedding_dim', 'batch_size',
        'bi_encoder', 'layers', 'dropout_prob', 'cell_type', 'cell_size',
        'time_major', 'attention', 'optimizer', 'optimizer_kwargs', 'log_dir',
        'train_schedule', 'save_all_gradients', 'generate_log_dir',
        'test_size', 'src_vocab_file', 'tgt_vocab_file', 'infer_vocab',
    )
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
        self.derive_params()
        self.validate_params()

    def derive_params(self):
        if self.generate_log_dir:
            i = 0
            fmt = '{0:04d}'
            while os.path.exists(os.path.join(self.log_dir, fmt.format(i))):
                i += 1
            self.log_dir = os.path.join(self.log_dir, fmt.format(i))
        if self.test_size == 'derive':
            with open(self.test_file) as f:
                self.test_size = len(f.readlines())

    def validate_params(self):
        att_types = ('luong', 'scaled_luong', 'bahdanau')
        if self.attention not in att_types:
            raise ConfigError("Attention type must be one of {}".format(
                ', '.join(map(str, att_types))))

    def save(self, fn):
        d = {k: getattr(self, k) for k in self.__slots__}
        with open(fn, 'w') as f:
            yaml.dump(d, f)
