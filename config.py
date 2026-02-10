#!/usr/bin/env python
# -*- coding:utf-8 -*-
__author__ = 'Ning Shi'
__email__ = 'mrshininnnnn@gmail.com'


# dependency
# built-in
import os, argparse


def init_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    # Llama-3.2-3B
    # Llama-3.2-3B-Instruct
    # gemma-3-4b-pt
    # Qwen3-4B
    parser.add_argument('--llm', type=str, default='Qwen3-4B', help='llm model name')
    # beam search for word probability
    # the max beam depth 10 is based on the PWN
    parser.add_argument('--beam_depth', type=int, default=10, help='beam search depth')
    # method
    parser.add_argument('--temp', type=bool, default=True, help='use template context')
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
        # update other config
        self.llm_name = self.llm.split('/')[-1]
        # I/O
        self.CURR_PATH = './'
        self.RESOURCE_PATH = os.path.join(self.CURR_PATH, 'res')
        self.DATA_PATH = os.path.join(self.RESOURCE_PATH, 'data')
        self.RESULTS_PATH = os.path.join(self.RESOURCE_PATH, 'results')
        self.LLMS_PATH = os.path.join(self.RESOURCE_PATH, 'llms')
        self.LLM_PATH = os.path.join(self.LLMS_PATH, self.llm)
        self.LOG_PATH = os.path.join(self.RESOURCE_PATH, 'logs')
