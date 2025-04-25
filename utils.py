import tensorflow as tf
s = tf.contrib.slim
import sys
sys.path.append('../base')
from lanenet import EN, EN_as
def ld_enet(sess, ckpt_dir, img, bs):
    ckpt = tf.train.latest_checkpoint(ckpt_dir)
    ni = 1
    sc = False
    sr = 2
    
    with s.arg_scope(EN_as()):
        _, _ = EN(img, num_classes=12, batch_size=bs, train=True,
                   reuse=None, ni=ni, sr=sr, sc=sc)

    vr = s.get_variables_to_restore()
    sv = tf.train.Saver(vr)
    sv.restore(sess, ckpt)

    g = tf.get_default_graph()
    lp = g.get_tensor_by_name('EN/b5_1_lp:0')
    return lp

def add_tl(sess, lp, fd):
    lg = s.conv2d_transpose(lp, fd, [2,2], stride=2, 
                          biases_init=tf.constant_initializer(10.0), 
                          weights_init=tf.contrib.layers.xavier_initializer(), 
                          scope='Inst/tl/ct')

    with tf.variable_scope('', reuse=True):
        w = tf.get_variable('Inst/tl/ct/weights')
        b = tf.get_variable('Inst/tl/ct/biases')
        sess.run([w.initializer, b.initializer])

    return lg

def get_tvars(sess, dbg=False):
    tv = [v for v in tf.trainable_variables() if 'bias' not in v.name]
    
    if dbg:
        print ('Trainable vars')
        for i, v in enumerate(tf.trainable_variables()):
            print (i, v)
        print ('Actual trained vars')
        for v in tv:
            print (v)

    sess.run(tf.variables_initializer(
        [v for v in tf.trainable_variables() 
         if 'bias' in v.name and 'ENet/ib1' not in v.name 
         and 'ENet/b1' not in v.name and 'ENet/b2' not in v.name]))
    return tv

def get_sums(dl, lv, ld, lr, img, pred, lbl):
    sm = []
    for v in tf.trainable_variables():
        sm.append(tf.summary.histogram(v.name, v))
    sm.append(tf.summary.scalar('Train/dl', dl))
    sm.append(tf.summary.scalar('Train/lv', lv))
    sm.append(tf.summary.scalar('Train/ld', ld))
    sm.append(tf.summary.scalar('Train/lr', lr))
    sm.append(tf.summary.image('Train/Img/In', img, max_out=1))
    sm.append(tf.summary.image('Train/Img/Pred', tf.expand_dims(pred[:,:,:,0], axis=3), max_out=1))
    sm.append(tf.summary.image('Train/Img/Lbl', tf.expand_dims(lbl, axis=3), max_out=1))

    for s in sm:
        tf.add_to_collection('CS', s)

    sm_tr = tf.summary.merge_all('CS')

    sm_v = []
    sm_v.append(tf.summary.image('Val/Img/In', img, max_out=1))
    sm_v.append(tf.summary.image('Val/Img/Pred', tf.expand_dims(pred[:,:,:,0], axis=3), max_out=1))
    sm_v.append(tf.summary.image('Val/Img/Lbl', tf.expand_dims(lbl, axis=3), max_out=1))    
    sm_v.append(tf.summary.scalar('Val/dl', dl))
    sm_val = tf.summary.merge(sm_v)
    return sm_tr, sm_val

def cnt_params():
    tp = 0
    for v in tf.trainable_variables():
        sh = v.get_shape()
        vp = 1
        for d in sh:
            vp *= d.value
        tp += vp
    return tp