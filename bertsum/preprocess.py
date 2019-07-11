#encoding=utf-8


import argparse
import time

from others.logging import init_logger
from prepro import data_builder


def do_format_to_lines(args):
    print(time.clock())
    data_builder.format_to_lines(args)
    print(time.clock())

def do_tokenize(args):
    print(time.clock())
    data_builder.tokenize(args)
    print(time.clock())

def do_format_to_bert(args):
    print(time.clock())
    data_builder.format_to_bert(args)
    print(time.clock())




def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-mode", default='', type=str, help='format_to_lines or format_to_bert')
    parser.add_argument("-oracle_mode", default='greedy', type=str, help='how to generate oracle summaries, greedy or combination, combination will generate more accurate oracles but take much longer time.')
    parser.add_argument("-map_path", default='../data/')
    parser.add_argument("-raw_path", default='../json_data/')
    parser.add_argument("-save_path", default='../bert_data/')

    parser.add_argument("-shard_size", default=2000, type=int)
    parser.add_argument('-min_nsents', default=3, type=int)
    parser.add_argument('-max_nsents', default=100, type=int)
    parser.add_argument('-min_src_ntokens', default=5, type=int)
    parser.add_argument('-max_src_ntokens', default=200, type=int)

    parser.add_argument("-lower", type=str2bool, nargs='?',const=True,default=True)

    parser.add_argument('-log_file', default='../../logs/cnndm.log')

    parser.add_argument('-dataset', default='', help='train, valid or test, defaul will process all datasets')

    parser.add_argument('-n_cpus', default=2, type=int)


    args = parser.parse_args()
    init_logger(args.log_file)
    eval('data_builder.'+args.mode + '(args)')
