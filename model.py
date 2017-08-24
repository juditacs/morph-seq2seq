#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017 Judit Acs <judit@sch.bme.hu>
#
# Distributed under terms of the MIT license.
import tensorflow as tf


class Seq2seqTrainModel(object):
    def __init__(self, config, dataset):
        self.config = config
        self.dataset = dataset
        self.create_model()

    def create_model(self):
        self.create_embedding()
        self.create_encoder()
        self.create_decoder()

    def create_cell(self, scope, reuse):
        with tf.variable_scope(scope, reuse=reuse):
            if self.config.cell_type == 'LSTM':
                cell = tf.nn.rnn_cell.BasicLSTMCell(self.config.cell_size)
                if self.config.dropout_prob > 0:
                    cell = tf.contrib.rnn.DropoutWrapper(
                        cell, input_keep_prob=1.0-self.config.dropout_prob)
            return cell

    def create_graph(self, reuse, data):
        with tf.variable_scope("embedding", reuse=reuse):
            src_embedding = tf.get_variable(
                "src_embedding",
                [self.dataset.src_vocab_size, self.config.src_embedding_dim],
                dtype=tf.float32)
            if self.config.share_vocab:
                tgt_embedding = src_embedding
            else:
                tgt_embedding = tf.get_variable(
                    "tgt_embedding",
                    [self.dataset.tgt_vocab_size, self.config.tgt_embedding_dim],
                    dtype=tf.float32)
            enc_emb_inp = tf.nn.embedding_lookup(src_embedding, data['src_ids'])
            dec_emb_inp = tf.nn.embedding_lookup(tgt_embedding, data['tgt_in_ids'])
            if is_time_major:
                enc_emb_inp = tf.transpose(enc_emb_inp, [1, 0, 2])
                dec_emb_inp = tf.transpose(dec_emb_inp, [1, 0, 2])

        with tf.variable_scope("encoder", reuse=reuse):
            if self.config.bi_encoder:
                fw_cells = []
                bw_cells = []
                for i in range(self.config.layers // 2):
                    fw_cells.append(self.create_cell("encoder", reuse))
                    bw_cells.append(self.create_cell("encoder", reuse))
                if len(fw_cells) > 1:
                    fw_cell = tf.contrib.rnn.MultiRNNCell(fw_cells)
                    bw_cell = tf.contrib.rnn.MultiRNNCell(bw_cells)
                else:
                    fw_cell = fw_cells[0]
                    bw_cell = bw_cells[0]
                o, e = tf.nn.bidirectional_dynamic_rnn(
                    fw_cell, bw_cell, enc_emb_inp, dtype='float32',
                    sequence_length=data['src_len'], time_major=self.config.time_major
                )
                encoder_outputs = tf.concat(o, -1)
                encoder_state = []
                for i in range(self.config.layers // 2):
                    encoder_state.append(e[0][i])
                    encoder_state.append(e[1][i])
                encoder_state = tuple(encoder_state)
            else:
                cell = self.create_cell("encoder", reuse)
                o, e = tf.nn.dynamic_rnn(
                    cell, enc_emb_inp, dtype='float32',
                    sequence_length=data['src_len'], time_major=self.config.time_major
                )
                encoder_outpus = o
                encoder_state = e

        with tf.variable_scope("decoder", reuse=reuse) as scope:
            pass
