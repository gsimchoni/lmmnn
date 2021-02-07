#!/usr/bin/env python3

import argparse
import yaml
import logging
from lmmnn.simulation import simulation

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--conf', help='Configuration YAML file', required=True)
    parser.add_argument('--out', help='Output CSV file name', required=True)
    args = parser.parse_args()
    return args

def load_yaml(conf_file):
    with open(conf_file) as f:
        params = yaml.load(f, Loader=yaml.FullLoader)
    return params

def get_logger():
    logger = logging.getLogger('LMMNN.logger')
    logger.setLevel(logging.DEBUG)
    c_handler = logging.StreamHandler()
    f_handler = logging.FileHandler('lmmnn_sim.log')
    c_handler.setLevel(logging.INFO)
    f_handler.setLevel(logging.DEBUG)
    c_format = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
    f_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    c_handler.setFormatter(c_format)
    f_handler.setFormatter(f_format)
    logger.addHandler(c_handler)
    logger.addHandler(f_handler)
    return logger

def main():
    logger = get_logger()
    logger.info('Started LMMMNN simulation.')
    args = parse()
    params = load_yaml(args.conf)
    simulation(args.out, params)

if __name__ == "__main__":
    main()