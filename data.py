#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017 Judit Acs <judit@sch.bme.hu>
#
# Distributed under terms of the MIT license.
import os
import tempfile
from sys import stdin

import tensorflow as tf
from tensorflow.python.ops import lookup_ops


class Dataset(object):
    PAD = 0
    SOS = 1
    EOS = 2
    UNK = 4
    constants = ['PAD', 'SOS', 'EOS', 'UNK']

    hu_alphabet = list("aábcdeéfghiíjklmnoóöőpqrstuúüűvwxyz-.")

    default_vocab = constants + hu_alphabet

    def __init__(self, config):
        self.config = config
        self.create_vocabs()
        self.create_tables()
        self.load_and_preprocess_data()

    def create_vocabs(self):
        """Create vocabularies via one of four ways
        1. read from file
        2. read from config directly
        3. use defaults - Hungarian alphabet
        4. infer from training data
        """
        if self.config.infer_vocab:
            self.infer_vocab_from_train()
            return
        self.__create_vocab('src_vocab')
        if self.config.share_vocab:
            self.tgt_vocab = self.src_vocab
            self.tgt_vocab_size = self.src_vocab_size
        else:
            self.__create_vocab('tgt_vocab')

    def infer_vocab_from_train(self):
        src_abc = set()
        if self.config.share_vocab:
            tgt_abc = src_abc
        else:
            tgt_abc = set()
        with open(self.config.train_file) as f:
            for line in f:
                enc, dec = line.rstrip('\n').split('\t')
                src_abc |= set(enc.split(' '))
                tgt_abc |= set(dec.split(' '))
        self.src_vocab = Dataset.constants + list(src_abc)
        self.src_vocab_size = len(self.src_vocab)
        if self.config.share_vocab:
            self.tgt_vocab = self.src_vocab
            self.tgt_vocab_size = self.src_vocab_size
        else:
            self.tgt_vocab = Dataset.constants + list(tgt_abc)
            self.tgt_vocab_size = len(self.tgt_vocab)

    def __create_vocab(self, attr_name):
        vocab_fn = attr_name + '_file'
        if hasattr(self.config, vocab_fn):
            with open(getattr(self.config, vocab_fn)) as f:
                vocab = Dataset.constants + [l.rstrip() for l in f]
                setattr(self, attr_name, vocab)
        elif hasattr(self.config, attr_name):
            vocab = Dataset.constants + getattr(self.config, attr_name)
            setattr(self, attr_name, getattr(self.config, attr_name))
        else:
            setattr(self, attr_name, Dataset.default_vocab)
        setattr(self, attr_name+'_size', len(getattr(self, attr_name)))

    def create_tables(self):
        self.src_table = lookup_ops.index_table_from_tensor(
            tf.constant(self.src_vocab), default_value=Dataset.UNK)
        if self.config.share_vocab:
            self.tgt_table = self.src_table
        else:
            self.tgt_table = lookup_ops.index_table_from_tensor(
                tf.constant(self.tgt_vocab), default_value=Dataset.UNK)

    def load_and_preprocess_data(self):
        self.train = self.__load_data(self.config.train_file)
        if hasattr(self.config, 'dev_file'):
            self.dev = self.__load_data(self.config.dev_file)
        if hasattr(self.config, 'test_file'):
            self.test = self.__load_data(self.config.test_file)

    def __load_data(self, fn):
        dataset = tf.contrib.data.TextLineDataset(fn)
        dataset = dataset.repeat()
        dataset = dataset.map(
            lambda s: tf.string_split([s], delimiter='\t').values)

        src = dataset.map(lambda s: s[0])
        tgt = dataset.map(lambda s: s[1])

        src = src.map(lambda s: tf.string_split([s], delimiter=' ').values)
        src = src.map(lambda s: s[:self.config.src_maxlen])
        tgt = tgt.map(lambda s: tf.string_split([s], delimiter=' ').values)
        tgt = tgt.map(lambda s: s[:self.config.tgt_maxlen])

        src = src.map(lambda words: self.src_table.lookup(words))
        tgt = tgt.map(lambda words: self.tgt_table.lookup(words))

        dataset = tf.contrib.data.Dataset.zip((src, tgt))
        dataset = dataset.map(
            lambda src, tgt: (
                src,
                tf.concat(([Dataset.SOS], tgt), 0),
                tf.concat((tgt, [Dataset.EOS]), 0),
            )
        )
        dataset = dataset.map(
            lambda src, tgt_in, tgt_out:
            (src, tgt_in, tgt_out, tf.size(src), tf.size(tgt_in))
        )
        batched = dataset.padded_batch(
            self.config.batch_size,
            padded_shapes=(
                tf.TensorShape([self.config.src_maxlen]),
                tf.TensorShape([self.config.tgt_maxlen+2]),
                tf.TensorShape([None]),
                tf.TensorShape([]),
                tf.TensorShape([]),
            ),
        )
        batched_iter = batched.make_initializable_iterator()
        s = batched_iter.get_next()
        return {
            'src_ids': s[0],
            'tgt_in_ids': s[1],
            'tgt_out_ids': s[2],
            'src_size': s[3],
            'tgt_size': s[4],
            'batched_iter': batched_iter,
        }

    def run_initializers(self, session):
        session.run(tf.tables_initializer())
        session.run(self.train['batched_iter'].initializer)
        if hasattr(self.config, 'dev_file'):
            session.run(self.dev['batched_iter'].initializer)
        if hasattr(self.config, 'test_file'):
            session.run(self.test['batched_iter'].initializer)

    def create_inverse_vocabs(self):
        self.src_inv_vocab = {i: c for i, c in enumerate(self.src_vocab)}
        self.tgt_inv_vocab = {i: c for i, c in enumerate(self.tgt_vocab)}

    def save_vocabs(self):
        with open(os.path.join(self.config.log_dir, 'src_vocab'), 'w') as f:
            f.write('\n'.join(self.src_vocab))
        with open(os.path.join(self.config.log_dir, 'tgt_vocab'), 'w') as f:
            f.write('\n'.join(self.tgt_vocab))


class InferenceDataset(Dataset):
    def __init__(self, config):
        self.config = config
        self.create_tables()
        self.test = self.load_and_preprocess_data()

    def create_tables(self):
        with open(os.path.join(self.config.config_dir, 'src_vocab')) as f:
            self.src_vocab = [l.rstrip('\n') for l in f]
            self.src_table = lookup_ops.index_table_from_tensor(
                tf.constant(self.src_vocab), default_value=Dataset.UNK)
            self.src_vocab_size = len(self.src_vocab)
        with open(os.path.join(self.config.config_dir, 'tgt_vocab')) as f:
            self.tgt_vocab = [l.rstrip('\n') for l in f]
            self.tgt_table = lookup_ops.index_table_from_tensor(
                tf.constant(self.tgt_vocab), default_value=Dataset.UNK)
            self.tgt_vocab_size = len(self.tgt_vocab)

    def set_test_fn(self):
        if self.config.test_fn is None:
            with tempfile.NamedTemporaryFile("wt", delete=False) as test_fn:
                for line in stdin:
                    test_fn.write(" ".join(line))
            return test_fn.name
        return self.config.test_fn

    def load_and_preprocess_data(self):
        test_fn = self.set_test_fn()
        with open(test_fn) as f:
            self.config.test_size = len(f.readlines())
        dataset = tf.contrib.data.TextLineDataset(test_fn)
        dataset = dataset.map(lambda s: tf.string_split(
            [s], delimiter='\t').values[0])
        dataset = dataset.map(lambda s: tf.string_split(
            [s], delimiter=' ').values)
        dataset = dataset.map(lambda words: self.src_table.lookup(words))
        dataset = dataset.map(
            lambda src: (
                src[:self.config.src_maxlen],
                tf.concat(([Dataset.SOS], [0]), 0),
                tf.concat(([0], [Dataset.EOS]), 0),
            )
        )
        dataset = dataset.map(
            lambda src, tgt_in, tgt_out:
                (src, tgt_in, tgt_out, tf.size(src), tf.size(tgt_in)))
        dataset = dataset.repeat()
        batched = dataset.padded_batch(
            self.config.batch_size,
            padded_shapes=(
                tf.TensorShape([self.config.src_maxlen]),
                tf.TensorShape([self.config.tgt_maxlen+2]),
                tf.TensorShape([None]),
                tf.TensorShape([]),
                tf.TensorShape([]),
            ),
        )
        batched_iter = batched.make_initializable_iterator()
        s = batched_iter.get_next()
        return {
            'src_ids': s[0],
            'tgt_in_ids': s[1],
            'tgt_out_ids': s[2],
            'src_size': s[3],
            'tgt_size': s[4],
            'batched_iter': batched_iter,
        }

    def run_initializers(self, session):
        session.run(tf.tables_initializer())
        session.run(self.test['batched_iter'].initializer)
