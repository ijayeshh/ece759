import os
import argparse
import numpy as np
import cv2
import tensorflow as tf
from glob import glob
import time
from lanenet import EN, EN_as
from clster import get_masks

s = tf.contrib.slim

def r(sess, c_dir, i_img, b_s, f_d):
    cp = tf.train.latest_checkpoint(c_dir)
    n_b, s_c, s_r = 1, False, 2
    
    with s.arg_scope(EN_as()):
        _, _ = EN(i_img, 12, b_s, True, None, n_b, s_r, s_c)

    g = tf.get_default_graph()
    l_p = g.get_tensor_by_name('EN/bottleneck5_1_last_prelu:0')
    l_g = s.conv2d_transpose(l_p, f_d, [2,2], 2, scope='Instance/transfer_layer/conv2d_transpose')

    v_r = s.get_variables_to_restore()
    s_v = tf.train.Saver(v_r)
    s_v.restore(sess, cp)
    return l_g

def s_i(pred, o_dir, idx):
    mn, mx = pred.min(), pred.max()
    norm = ((pred - mn)*255/(mx-mn)).astype(np.uint8)
    cv2.imwrite(os.path.join(o_dir, f'color_{idx:04d}.png'), np.squeeze(norm))

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('-m','--model_dir', default='trained_model')
    ap.add_argument('-i','--input_dir', default=os.path.join('inference_test','images'))
    ap.add_argument('-o','--output_dir', default=os.path.join('inference_test','results'))
    args = ap.parse_args()

    if not os.path.isdir(args.output_dir):
        os.mkdir(args.output_dir)

    img_paths = sorted(glob(os.path.join(args.input_dir, '*.jpg')))
    img_shape = (512, 512)
    b_size, f_dim = 1, 3

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.5

    with tf.Session(config=config) as sess:
        input_img = tf.placeholder(tf.float32, (None, img_shape[1], img_shape[0], 3))
        logits = r(sess, args.model_dir, input_img, b_size, f_dim)

        inf_time = cl_time = 0
        for i, path in enumerate(img_paths):
            img = cv2.cvtColor(cv2.resize(cv2.imread(path), img_shape), cv2.COLOR_BGR2RGB)
            img_exp = np.expand_dims(img, 0)

            t = time.time()
            pred = sess.run(logits, {input_img: img_exp})
            inf_time += time.time() - t

            pred_c = np.squeeze(pred.copy())
            s_i(pred_c, args.output_dir, i)

            t = time.time()
            inst_mask = get_masks(pred.copy(), 1.)[0]
            colors, counts = np.unique(inst_mask.reshape(-1,3), return_counts=True, axis=0)
            bg = colors[np.argmax(counts)]
            inst_mask[np.where(inst_mask==bg)] = 0.
            blended = cv2.addWeighted(img, 1, cv2.resize(inst_mask, (1280,720)), 0.3, 0)
            cl_time += time.time() - t
            cv2.imwrite(os.path.join(args.output_dir, f'cluster_{i:04d}.png'), 
                       cv2.cvtColor(blended, cv2.COLOR_RGB2BGR))

        n = len(img_paths)
        print(f'Avg inference: {inf_time/n:.4f}s ({n/inf_time:.2f}fps)')
        print(f'Avg cluster: {cl_time/n:.4f}s ({n/cl_time:.2f}fps)')
        print(f'Total avg: {(inf_time+cl_time)/n:.4f}s ({n/(inf_time+cl_time):.2f}fps)')