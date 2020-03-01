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

def tfsm():
  options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
  run_metadata = tf.RunMetadata()
  
  with tf.Session() as sess:
    tf.saved_model.loader.load(
        sess, [tf.saved_model.tag_constants.SERVING],
        "/tmp/SavedModel-1024-1024/1"
    )
    input_images = sess.graph.get_tensor_by_name('input_images:0')
    output_node1 = sess.graph.get_operation_by_name('feature_fusion/concat_3')
    output_node2 = sess.graph.get_operation_by_name('feature_fusion/Conv_7/Sigmoid')

    for i in range(0, 1000):
        start_time = time.time()
        sess.run([output_node1,output_node2], feed_dict={input_images: inputs})
        if i % 100 == 1:
            sess.run([output_node1,output_node2], feed_dict={input_images: inputs}, options=options, run_metadata=run_metadata)
            fetched_timeline = timeline.Timeline(run_metadata.step_stats)
            chrome_trace = fetched_timeline.generate_chrome_trace_format()
            with tf.gfile.Open('tf-sm_timeline_01.json' + str(i), 'w') as f:
                f.write("")
            with tf.gfile.Open('tf-sm_timeline_01.json' + str(i), 'w') as f:
                f.write(chrome_trace)
            logging.info("write timeline")
        print("saved_model cost %s seconds " % (time.time() - start_time))
        sys.stdout.flush()
  print('------------------------------------')

if __name__ == '__main__':
  tfsm()

