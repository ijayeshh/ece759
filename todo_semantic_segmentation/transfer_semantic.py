import os.path
import tensorflow as tf
import helper
import warnings
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from glob import glob
from lanenet import EN, EN_as
from tensorflow.contrib.layers.python.layers import initializers
import numpy as np
slim = tf.contrib.slim

if not tf.test.gpu_device_name():
    warnings.warn('No GPU found.')
else:
    print('GPU: {}'.format(tf.test.gpu_device_name()))

def ld_enet(sess, ckpt_dir, img_in, bs, nc):
    ckpt = tf.train.latest_checkpoint(ckpt_dir)
    ni = 1
    sc = False
    st = 2

    with slim.arg_scope(EN_as()):
        lg, _ = EN(img_in, num_classes=12, batch_size=bs, 
                    is_training=True, reuse=None, num_initial_blocks=ni,
                    stage_two_repeat=st, skip_connections=sc)

    vr = slim.get_variables_to_restore()
    sv = tf.train.Saver(vr)
    sv.restore(sess, ckpt)
    gr = tf.get_default_graph()

    lp = gr.get_tensor_by_name('ENet/bottleneck5_1_last_prelu:0')
    ot = slim.conv2d_transpose(lp, nc, [2,2], stride=2, 
                            weights_initializer=initializers.xavier_initializer(), 
                            scope='Semantic/transfer_layer/conv2d_transpose')

    pb = tf.nn.softmax(ot, name='Semantic/transfer_layer/logits_to_softmax')

    with tf.variable_scope('', reuse=True):
        wt = tf.get_variable('Semantic/transfer_layer/conv2d_transpose/weights')
        bs = tf.get_variable('Semantic/transfer_layer/conv2d_transpose/biases')
        sess.run([wt.initializer, bs.initializer])

    return ot, pb

def opt(sess, lg, cl, lr, nc, tr, gs):
    wts = cl * np.array([1., 40.])
    wts = tf.reduce_sum(wts, axis=3)
    ls = tf.reduce_mean(tf.losses.softmax_cross_entropy(onehot_labels=cl, logits=lg, weights=wts))

    with tf.name_scope('Semantic/Adam'):
        to = tf.train.AdamOptimizer(learning_rate=lr).minimize(ls, var_list=tr, global_step=gs)
    ai = [var.initializer for var in tf.global_variables() if 'Adam' in var.name]
    sess.run(ai)
    return lg, to, ls

def rn():
    ish = (512, 512)
    mdir = '../checkpoint'
    ddir = '../../tusimple_api/clean_data'
    ldir = './log'
    odir = './saved_model'

    nc = 2
    ep = 20
    bs = 1
    slr = 1e-4
    lrdi = 500
    lrdr = 0.96

    ip = glob(os.path.join(ddir, 'images', '*.png'))
    lp = glob(os.path.join(ddir, 'labels', '*.png'))

    xt, xv, yt, yv = train_test_split(ip, lp, test_size=0.20, random_state=42)

    cfg = tf.ConfigProto()
    cfg.gpu_options.allow_growth = True
    cfg.gpu_options.per_process_gpu_memory_fraction = 0.7

    with tf.Session(config=cfg) as sess:
        ii = tf.placeholder(tf.float32, shape=[bs, ish[1], ish[0], 3])
        cl = tf.placeholder(dtype=tf.float32, shape=(None, ish[1], ish[0], 2), name='Semantic/input_image')

        lg, pb = ld_enet(sess, mdir, ii, bs, nc)
        pv = tf.argmax(pb, axis=-1)
        pv = tf.cast(pv, dtype=tf.float32)
        pv = tf.reshape(pv, shape=[bs, ish[1], ish[0], 1])

        gs = tf.Variable(0, trainable=False)
        sess.run(gs.initializer)
        lr = tf.train.exponential_decay(slr, gs, lrdi, lrdr, staircase=True)

        tr = [var for var in tf.trainable_variables() if 'bias' not in var.name and 'ENet/fullconv' not in var.name]

        lg, to, ls = opt(sess, lg, cl, lr, nc, tr, gs)
        st = tf.summary.scalar('training_loss', ls)
        si = tf.summary.image('Images/Validation_original_image', ii, max_outputs=1)
        so = tf.summary.image('Images/Validation_segmentation_output', pv, max_outputs=1)
        sm = tf.summary.merge_all()
        sv = tf.summary.scalar('validation_loss', ls)

        tw = tf.summary.FileWriter(ldir)
        sa = tf.train.Saver()

        st = 0
        sv = 0
        sc = 10
        for e in range(ep):
            print ('epoch', e)
            print ('training ...')
            tl = 0
            for im, lb in helper.get_batches_fn(bs, ish, xt, yt):
                cr = sess.run(lr)
                if st%sc==0:
                    _, su, lo = sess.run([to, sm, ls], feed_dict={ii: im, cl: lb})
                    tw.add_summary(su, st)
                    print ('epoch', e, '\t step', st, '\t loss', lo, '\t lr', cr)
                else:
                    _, lo = sess.run([to, ls], feed_dict={ii: im, cl: lb})
                st+=1
                tl += lo

                if (st%5000==4999):
                    sa.save(sess, os.path.join(odir, 'model.ckpt'), global_step=gs)

            print ('train loss', tl)

            print ('validating ...')
            vl = 0
            for im, lb in helper.get_batches_fn(bs, ish, xv, yv):
                if sv%sc==0:
                    su, lo = sess.run([sv, ls], feed_dict={ii: im, cl: lb})
                    tw.add_summary(su, sv)
                    print ('batch loss', lo)
                else:
                    lo = sess.run(ls, feed_dict={ii: im, cl: lb})

                vl += lo
                sv+=1

            print ('valid loss', vl)

if __name__ == '__main__':
    rn()