#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017 Judit Acs <judit@sch.bme.hu>
#
# Distributed under terms of the MIT license.
import logging
import tensorflow as tf
from tensorflow.python.layers import core as layers_core


class Graph(object):
    def __init__(self, **kwargs):
        for param, value in kwargs.items():
            setattr(self, param, value)


class Seq2seqTrainModel(object):
    def __init__(self, config, dataset):
        self.config = config
        self.dataset = dataset
        self.create_model()

    def create_model(self):
        self.train_graph = self.create_graph(reuse=False, data=self.dataset.train)
        self.valid_graph = self.create_graph(reuse=True, data=self.dataset.dev)

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
            if self.config.time_major:
                enc_emb_inp = tf.transpose(enc_emb_inp, [1, 0, 2])
                dec_emb_inp = tf.transpose(dec_emb_inp, [1, 0, 2])

        with tf.variable_scope("encoder", reuse=reuse):
            if self.config.bi_encoder:
                fw_cells = []
                bw_cells = []
                for i in range(self.config.layers // 2):
                    fw_cells.append(self.create_cell("encoder", reuse))
                    bw_cells.append(self.create_cell("encoder", reuse))
                fw_cell = tf.contrib.rnn.MultiRNNCell(fw_cells)
                bw_cell = tf.contrib.rnn.MultiRNNCell(bw_cells)
                o, e = tf.nn.bidirectional_dynamic_rnn(
                    fw_cell, bw_cell, enc_emb_inp, dtype='float32',
                    sequence_length=data['src_size'], time_major=self.config.time_major
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
                    sequence_length=data['src_size'], time_major=self.config.time_major
                )
                encoder_outputs = o
                encoder_state = e

        with tf.variable_scope("decoder", reuse=reuse) as scope:
            if self.config.bi_encoder:
                decoder_cells = [self.create_cell("decoder", reuse) for _ in range(self.config.layers)]
                decoder_cell = tf.contrib.rnn.MultiRNNCell(decoder_cells)
            else:
                decoder_cell = self.create_cell("decoder", reuse)
            if self.config.time_major:
                attention_states = tf.transpose(encoder_outputs, [1, 0, 2])
            else:
                attention_states = encoder_outputs
            if self.config.attention == 'luong':
                mechanism = tf.contrib.seq2seq.LuongAttention(
                    self.config.cell_size, attention_states,
                    memory_sequence_length=data['src_size'],
                    scale=False
                )
            elif self.config.attention == 'scaled_luong':
                mechanism = tf.contrib.seq2seq.LuongAttention(
                    self.config.cell_size, attention_states,
                    memory_sequence_length=data['src_size'],
                    scale=True
                )
            elif self.config.attention == 'bahdanau':
                mechanism = tf.contrib.seq2seq.BahdanauAttention(
                    self.config.cell_size, attention_states,
                    memory_sequence_length=data['src_size'],
                    scale=True
                )
            decoder_cell = tf.contrib.seq2seq.AttentionWrapper(
                decoder_cell, mechanism,
                attention_layer_size=self.config.cell_size, name="attention"
            )
            size_dim = 1 if self.config.time_major else 0
            dec_init_state = decoder_cell.zero_state(
                tf.shape(dec_emb_inp)[size_dim],
                tf.float32).clone(cell_state=encoder_state)

            helper = tf.contrib.seq2seq.TrainingHelper(
                dec_emb_inp, data['tgt_size'], time_major=self.config.time_major)
            decoder = tf.contrib.seq2seq.BasicDecoder(
                decoder_cell, helper, dec_init_state
            )
            outputs, final, _ = tf.contrib.seq2seq.dynamic_decode(
                decoder, output_time_major=self.config.time_major,
                swap_memory=True, scope=scope
            )
            output_proj = layers_core.Dense(self.dataset.tgt_vocab_size, name="output_proj")
            logits = output_proj(outputs.rnn_output)

        with tf.variable_scope("train", reuse=reuse):
            if self.config.time_major:
                logits =  tf.transpose(logits, [1, 0, 2])
            xent = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=data['tgt_out_ids'], logits=logits)
            target_weights = tf.sequence_mask(
                data['tgt_size'], tf.shape(logits)[1], tf.float32)
            loss = tf.reduce_sum(xent * target_weights) / tf.to_float(self.config.batch_size)
            tf.summary.scalar("loss", loss)
            learning_rate =  tf.placeholder(dtype=tf.float32, name="learning_rate")
            # max_global_norm = tf.placeholder(dtype=tf.float32, name="max_global_norm")
            optimizer = getattr(tf.train, self.config.optimizer)(learning_rate, **self.config.optimizer_kwargs)
            params = tf.trainable_variables()
            gradients = tf.gradients(loss, params)
            for grad, var in zip(gradients, params):
                tf.summary.histogram(var.op.name+'/gradient', grad)
            gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
            for grad, var in zip(gradients, params):
                tf.summary.histogram(var.op.name+'/clipped_gradient', grad)
            update = optimizer.apply_gradients(zip(gradients, params))
        return Graph(
            learning_rate=learning_rate,
            loss=loss,
            update=update,
        )

    def init_session(self):
        self.sess = tf.Session()
        self.dataset.run_initializers(self.sess)
        self.sess.run(tf.global_variables_initializer())
        self.merged_summary = tf.summary.merge_all()
        self.writer = tf.summary.FileWriter(self.config.log_dir)
        self.writer.add_graph(self.sess.graph)

    def train(self, epochs, learning_rate, logstep=0):
        logstep = int(epochs // 10) if logstep == 0 else logstep
        for i in range(epochs):
            _, loss, s = self.sess.run(
                [self.train_graph.update, self.train_graph.loss, self.merged_summary],
                feed_dict={self.train_graph.learning_rate: learning_rate})
            val_loss, _ = self.sess.run([self.valid_graph.loss, self.merged_summary])
            self.writer.add_summary(s, i)
            if i % logstep == logstep - 1:
                logging.info('Iter {}, train loss: {}, val loss: {}'.format(i+1, loss, val_loss))
