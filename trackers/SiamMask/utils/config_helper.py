# --------------------------------------------------------
# SiamMask
# Licensed under The MIT License
# Written by Qiang Wang (wangqiang2015 at ia.ac.cn)
# Revised for THOR by Axel Sauer (axel.sauer@tum.de)
# --------------------------------------------------------

import json
from os.path import exists


def load_config(config, arch):
    assert exists(config), '"{}" not exists'.format(config)
    config = json.load(open(config))

    # deal with network
    if 'network' not in config:
        print('Warning: network lost in config. This will be error in next version')

        config['network'] = {}

        if not arch:
            raise Exception('no arch provided')

    arch = config['network']['arch']

    return config

