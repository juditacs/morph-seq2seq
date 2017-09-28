#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017 Judit Acs <judit@sch.bme.hu>
#
# Distributed under terms of the MIT license.

from argparse import ArgumentParser
from model import Seq2seqInferenceModel
from data import InferenceDataset
from config import InferenceConfig


def parse_args():
    p = ArgumentParser()
    p.add_argument('-e', '--exp-dir', type=str)
    p.add_argument('-t', '--test-file', type=str)
    p.add_argument('-o', '--output-file', type=str)
    return p.parse_args()


def main():
    args = parse_args()
    config = InferenceConfig.from_config_dir(args.exp_dir)
    config.test_fn = args.test_file
    dataset = InferenceDataset(config)
    model = Seq2seqInferenceModel(config, dataset)
    model.run_inference(outfile=args.output_file)


if __name__ == '__main__':
    import logging
    log_fmt = '%(asctime)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()
