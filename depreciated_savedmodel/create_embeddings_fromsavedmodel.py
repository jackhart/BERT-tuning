#!/usr/bin/env python3
"""About this script"""

import os
import argparse
import collections
import numpy as np
import tensorflow as tf
import bert.tokenization as tokenization
from bert.extract_features import read_examples, convert_examples_to_features, InputFeatures

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--do-lower-case", type=bool, default=False)
    parser.add_argument("--max-seq-length", type=int, default=128)

    parser.add_argument("--vocab_file", required=True)
    parser.add_argument('--savedmodel-dir', type=str, required=True)
    parser.add_argument('--input-file', type=str, required=True)
    #parser.add_argument('--output-file', type=str, required=True)

    return parser.parse_args()

def create_dataset_from_features(features, seq_length):
  """creates Dataset for BERT

    Parameters
    ----------
    features : InputFeatures
    seq_length : Int

    Returns
    -------
    tf.Dataset
    """

  all_unique_ids = []
  all_input_ids = []
  all_input_mask = []

  for feature in features:
    all_unique_ids.append(feature.unique_id)
    all_input_ids.append(feature.input_ids)
    all_input_mask.append(feature.input_mask)

  num_examples = len(features)
  d = tf.data.Dataset.from_tensor_slices({
    "unique_ids":
        tf.constant(all_unique_ids, shape=[num_examples], dtype=tf.int32),
    "input_ids":
        tf.constant(
            all_input_ids, shape=[num_examples, seq_length],
            dtype=tf.int32),
    "input_mask":
        tf.constant(
            all_input_mask,
            shape=[num_examples, seq_length],
            dtype=tf.int32)
  })
  return d



def main(args):
    """running the script"""

    tf.logging.set_verbosity(tf.logging.INFO)

    tf.logging.info("convert sequences in features...")
    tokenizer = tokenization.FullTokenizer(
        vocab_file=args.vocab_file, do_lower_case=args.do_lower_case)

    examples = read_examples(args.input_file)

    features = convert_examples_to_features(
        examples=examples, seq_length=args.max_seq_length, tokenizer=tokenizer)

    unique_id_to_feature = {}
    for feature in features:
        unique_id_to_feature[feature.unique_id] = feature

    tf.logging.info("importing model as SavedModelPredictor...")
    infer = tf.contrib.predictor.from_saved_model(args.savedmodel_dir, signature_def_key="model")

    tf.logging.info("creating Dataset...")
    d = create_dataset_from_features(features, args.max_seq_length)
    d = d.batch(batch_size=args.batch_size, drop_remainder=False)
    iterator = d.make_one_shot_iterator()
    data_batch = iterator.get_next()

    with tf.Session() as sess:
        while True:
            try:
                created_input = sess.run(data_batch)
                pred_input = dict((key, created_input[key]) for key in ('input_ids', 'input_mask'))
                result = infer(pred_input)

                text_features = []
                for i in range(0, created_input["input_ids"].shape[0]):
                    layer_output = result["pooled_output"][i]
                    text_features.append(layer_output)

            except tf.errors.OutOfRangeError:
                break

    print("Completed!!")
    print(text_features[0])



if __name__ == "__main__":
    main(get_arguments())