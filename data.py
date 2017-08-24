#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017 Judit Acs <judit@sch.bme.hu>
#
# Distributed under terms of the MIT license.
import tensorflow as tf
from tensorflow.python.ops import lookup_ops


class Dataset(object):
    PAD = 0
    SOS = 1
    EOS = 2
    UNK = -1
    constants = ['UNK', 'PAD', 'SOS', 'EOS']

    hu_alphabet = list("aábcdeéfghiíjklmnoóöőpqrstuúüűvwxyz-.")

    default_vocab = constants + hu_alphabet

    def __init__(self, config):
        self.config = config
        self.create_vocabs()

    def create_vocabs(self):
        """Create vocabularies via one of three ways
        1. read from file
        2. read from config directly
        3. use defaults - Hungarian alphabet
        """
        self.__create_vocab('src_vocab')
        if self.config.share_vocab:
            self.tgt_vocab = self.src_vocab
        else:
            self.__create_vocab('tgt_vocab')

    def __create_vocab(self, attr_name):
        vocab_fn = attr_name = '_file'
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
            tf.contant(self.src_vocab), default_value=Dataset.UNK)
        if self.config.share_vocab:
            self.tgt_table = self.src_table
        else:
            self.tgt_table = lookup_ops.index_table_from_tensor(
                tf.contant(self.tgt_vocab), default_value=Dataset.UNK)

    def load_and_preprocess_data(self):
        self.train = self.__load_data(self.config.train_file)
        if hasattr(self.config, 'dev_file'):
            self.dev = self.__load_data(self.config.dev_file)

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
            )
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
