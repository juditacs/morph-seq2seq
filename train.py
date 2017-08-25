#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017 Judit Acs <judit@sch.bme.hu>
#
# Distributed under terms of the MIT license.

from argparse import ArgumentParser
from model import Seq2seqTrainModel
from data import Dataset
from config import Config


def parse_args():
    p = ArgumentParser()
    p.add_argument('-c', '--config', type=str)
    return p.parse_args()


def main():
    args = parse_args()
    config = Config.from_yaml(args.config)
    dataset = Dataset(config)
    model = Seq2seqTrainModel(config, dataset)
    model.run_experiment()

if __name__ == '__main__':
    import logging
    log_fmt = '%(asctime)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()
