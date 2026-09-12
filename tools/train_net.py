# -*- coding: utf-8 -*-
# @Time    : 6/17/25
# @Author  : Yaojie Shen
# @Project : CoCap
# @File    : train_net.py

import logging
from pathlib import Path

from hydra_zen import store, zen

from cocap.modeling.config_store import train, register_configs, seed_from_config

logger = logging.getLogger(__name__)

if __name__ == '__main__':
    register_configs(store)
    store.add_to_hydra_store()

    # pre_call runs before hydra instantiates the model, which is the only point at which
    # seeding also covers weight initialization
    zen(train, pre_call=seed_from_config).hydra_main(
        config_path=(Path(__file__).parent.parent / "configs").as_posix(),
        version_base=None,
    )
