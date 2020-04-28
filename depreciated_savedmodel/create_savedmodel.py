#!/usr/bin/env python3
"""This script is used initially to create SavedModel object.  This will make extracting embeddings easier"""

import os
import argparse
import json
import tensorflow as tf
import bert.modeling as modeling

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-dir', type=str, required=True)
    parser.add_argument('--config-name', type=str, required=True)
    parser.add_argument('--ckpt-name', type=str, required=True)
    parser.add_argument('--output-dir', type=str)

    return parser.parse_args()


def create_bert_graph(config_name, ckpt_name, model_dir):
    """Returns a BERT graph containing placeholders for input and session

    Parameters
    ----------
    config_name : str, The name of config file for BERT.  Asummed to be in model_dir.
    ckpt_name : str, The name of the BERT checkpoint file.  Location depents on if tuned_model_dir is passed.
    model_dir : str, The directory containing the originonal BERT model.

    Returns
    -------
    tf.Graph, The default graph being used in a session.
    tf.Session, The session that contains the graph.
    tf.tensor, The pooled output layer
    """

    #create config and checkpoint paths
    config_fp = os.path.join(model_dir, config_name)
    init_checkpoint = os.path.join(model_dir, ckpt_name)

    #load config file
    with tf.gfile.GFile(config_fp, 'r') as g_file:
        bert_config = modeling.BertConfig.from_dict(json.load(g_file))

    tf.logging.info('build graph...')
    input_ids = tf.placeholder(tf.int32, (None, None), 'input_ids')
    input_mask = tf.placeholder(tf.int32, (None, None), 'input_mask')
    input_type_ids = tf.zeros_like(input_mask, name='input_type_ids')

    model = modeling.BertModel(
        config=bert_config,
        is_training=False,
        input_ids=input_ids,
        input_mask=input_mask,
        token_type_ids=input_type_ids,
        use_one_hot_embeddings=False)

    tvars = tf.trainable_variables()

    tf.logging.info('load parameters from checkpoint...')
    (assignment_map, _
    ) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)

    tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

    tf.logging.info('loading graph from session...')
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    # name the output layer
    output_layer = model.get_pooled_output()

    return tf.get_default_graph(), sess, output_layer


def savedmodel_bert(output_tensor, output_dir, graph, sess):
    """Creates a SavedModel object for BERT in a given directory.

    Parameters
    ----------
    output_tensor : tf.tensor
        tf.tensor of the pooled output
    output_dir : str
        The directory this file should be saved in.
    graph : tf.Graph
        The BERT graph.
    sess : tf.Session
        The session containing the BERT graph.
    """

    tf.logging.info('exporting model...')
    inputs = {"input_ids": graph.get_tensor_by_name(name='input_ids:0'),
              "input_mask":  graph.get_tensor_by_name(name='input_mask:0')}
    outputs = {"pooled_output": output_tensor}

    def_map = {"model": tf.saved_model.signature_def_utils.predict_signature_def(inputs=inputs, outputs=outputs)}
    builder = tf.saved_model.builder.SavedModelBuilder(output_dir)
    builder.add_meta_graph_and_variables(sess,
                                         [tf.saved_model.tag_constants.SERVING],
                                         signature_def_map=def_map,
                                         assets_collection=graph.get_collection(tf.GraphKeys.ASSET_FILEPATHS))
    builder.save()


def main(args):
    """Exports BERT model checkpoint into a SavedModel Object"""

    #initalize graph and session
    graph, sess, pooled_output_layer = create_bert_graph(args.config_name, args.ckpt_name, args.model_dir)

    #create SavedModel
    savedmodel_bert(pooled_output_layer, args.output_dir, graph, sess)


if __name__ == "__main__":
    main(get_arguments())

