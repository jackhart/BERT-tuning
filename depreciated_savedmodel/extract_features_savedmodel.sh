#!/bin/bash

set -e
set -f
set -u

create_embeddings.py --input-file /home/jack/PycharmProjects/Comp_Ling_FinalProject/data/corpus.txt \
--vocab_file /home/jack/PycharmProjects/Comp_Ling_FinalProject/models/cased_L-12_H-768_A-12/vocab.txt \
--savedmodel-dir /home/jack/PycharmProjects/Comp_Ling_FinalProject/models/base_savedmodel
