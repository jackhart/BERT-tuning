#!/bin/bash

set -e
set -f
set -u

create_savedmodel.py --model-dir /home/jack/PycharmProjects/Comp_Ling_FinalProject/models/cased_L-12_H-768_A-12 \
--ckpt-name bert_model.ckpt \
--config-name bert_config.json \
--output-dir /home/jack/PycharmProjects/Comp_Ling_FinalProject/models
