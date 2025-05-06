#!/usr/bin/env python
# -*- coding:utf-8 -*-
__author__ = 'Author'
__email__ = 'Email'


# dependency
# built-in
import os, argparse


def init_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    # model
    # language models
    # Llama-4-Scout-17B-16E
    # Llama3.2-1B, Llama3.2-3B
    # meta-llama/Llama-3.1-8B, meta-llama/Llama-3.2-3B
    # Qwen/Qwen3-4B
    parser.add_argument('--llm', type=str, default='meta-llama/Llama-3.2-3B')
    # save as argparse space
    return parser.parse_known_args()[0]


class Config(object):
    """docstring for Config"""
    def __init__(self):
        super(Config, self).__init__()
        self.update_config(**vars(init_args()))

    def update_config(self, **kwargs):
        # load config from parser
        for k,v in kwargs.items():
            setattr(self, k, v)
        # I/O
        self.CURR_PATH = './'
        self.RESOURCE_PATH = os.path.join(self.CURR_PATH, 'res')
        self.DATA_PATH = os.path.join(self.RESOURCE_PATH, 'data')
        self.RESULTS_PATH = os.path.join(self.RESOURCE_PATH, 'results')
        self.LLMS_PATH = os.path.join(self.RESOURCE_PATH, 'llms')
        self.LLM_PATH = os.path.join(self.LLMS_PATH, self.llm)
