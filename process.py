import time
import cv2
import tensorflow as tf
import numpy as np
import json
import base64

from os import listdir
from os.path import isfile, join


in_dir = 'in_frames/'
output_dir = 'out_frames/'
model_dir = 'hallway_exported/'

# list of all paths to the frames to process
frames_path = [in_dir + f for f in listdir(in_dir) if isfile(join(in_dir, f))]

# restoring the exported model
sess = tf.Session()
saver = tf.train.import_meta_graph(model_dir + "export.meta")
saver.restore(sess, model_dir + "export")
input_vars = json.loads(tf.get_collection("inputs")[0])
output_vars = json.loads(tf.get_collection("outputs")[0])
input = tf.get_default_graph().get_tensor_by_name(input_vars["input"])
output = tf.get_default_graph().get_tensor_by_name(output_vars["output"])

# number of processed frames
frame_num = 0

window_title = 'preview'
cv2.startWindowThread()
cv2.namedWindow(window_title)

# starting processing
for frame_path in frames_path:

    with open(frame_path, "rb") as f:
        input_data = f.read()

    input_instance = dict(input=base64.urlsafe_b64encode(input_data).decode("ascii"), key="0")
    input_instance = json.loads(json.dumps(input_instance))
    input_value = np.array(input_instance["input"])

    t1 = time.time()
    output_value = sess.run(output, feed_dict={input: np.expand_dims(input_value, axis=0)})[0]
    t2 = time.time()
    print('run time: ' + str(t2 - t1))

    output_instance = dict(output=output_value.decode("ascii"), key="0")
    b64data = output_instance["output"]
    b64data += "=" * (-len(b64data) % 4)
    output_data = base64.urlsafe_b64decode(b64data.encode("ascii"))

    output_path = output_dir + str(frame_num) + '.png'
    with open(output_path, "wb") as f:
        f.write(output_data)

    frame_num += 1

    img = cv2.imread(output_path, cv2.IMREAD_UNCHANGED)
    cv2.imshow(window_title, img)

cv2.destroyAllWindows()

print (frames_path)
