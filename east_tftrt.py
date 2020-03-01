from __future__ import print_function
#import tensorflow.compat.v1 as tf
#tf.disable_v2_behavior()

import os
import time
import sys
import tensorflow as tf
from tensorflow.python.compiler.tensorrt import trt_convert as trt
import numpy as np
from tensorflow.python.client import timeline
import logging
import east

size = 1024
width = 1024
height = 1024
inputs = np.random.randint(1, 255, size=(1,width, height, 3))

def tftrt():
  options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
  run_metadata = tf.RunMetadata()

  input_images = tf.placeholder(tf.float32, shape=[1, width, height, 3], name='input_images')
  with tf.Session() as sess:
    converter = trt.TrtGraphConverter(
            #nput_graph_def=frozen_graph,
            "/tmp/SavedModel-1024-1024/1",
            is_dynamic_op=True,
            #minimum_segment_size=10,
            precision_mode=trt.TrtPrecisionMode.FP32,
            nodes_blacklist=['feature_fusion/concat_3', 'feature_fusion/Conv_7/Sigmoid']) #output nodes
    trt_graph = converter.convert()

    # For INT8
    if False:
      trt_graph = converter.calibrate(
        fetch_names=['feature_fusion/concat_3:0', 'feature_fusion/Conv_7/Sigmoid:0'],
        num_runs=10,
        feed_dict_fn=lambda: {'input_images:0': inputs})

    # Import the TensorRT graph into a new graph and run:
    output_node = tf.import_graph_def(
        trt_graph,
        input_map={"input_images": input_images},
        return_elements=['feature_fusion/concat_3', 'feature_fusion/Conv_7/Sigmoid'])
    for i in range(0, 1000):
        start_time = time.time()
        sess.run(output_node, feed_dict={input_images: inputs})
        if i % 100 == 1:
            sess.run([output_node], feed_dict={input_images: inputs}, options=options, run_metadata=run_metadata)
            fetched_timeline = timeline.Timeline(run_metadata.step_stats)
            chrome_trace = fetched_timeline.generate_chrome_trace_format()
            with tf.gfile.Open('tf-trt_timeline_01.json' + str(i), 'w') as f:
                f.write("")
            with tf.gfile.Open('tf-trt_timeline_01.json' + str(i), 'w') as f:
                f.write(chrome_trace)
            logging.info("write timeline")
        print("tf-trt cost %s seconds " % (time.time() - start_time))
        sys.stdout.flush()

  print('------------------------------------')

if __name__ == '__main__':
  tftrt()

