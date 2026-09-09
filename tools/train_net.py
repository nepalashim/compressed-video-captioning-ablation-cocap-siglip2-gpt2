# -*- coding: utf-8 -*-
# @Time    : 6/17/25
# @Author  : Yaojie Shen
# @Project : CoCap
# @File    : train_net.py

import logging
from pathlib import Path

from hydra_zen import store, zen

from cocap.modeling.config_store import train, register_configs

logger = logging.getLogger(__name__)

if __name__ == '__main__':
    register_configs(store)
    store.add_to_hydra_store()

    zen(train).hydra_main(
        config_path=(Path(__file__).parent.parent / "configs").as_posix(),
        version_base=None,
    )
