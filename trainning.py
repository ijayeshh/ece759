import os
import sys
import numpy as np
import cv2
import tensorflow as tf
from sklearn.model_selection import train_test_split
import argparse
import glob
import utils
import datagenerator
import visualize
import clster
def main():
    p = argparse.ArgumentParser()
    p.add_argument('-s','--src', default='data')
    p.add_argument('-m', '--model', default='pretrained_model')
    p.add_argument('-o', '--out', default='saved_model')
    p.add_argument('-l', '--log', default='log')
    p.add_argument('--epochs', type=int, default=4)
    p.add_argument('--var', type=float, default=1.)
    p.add_argument('--dist', type=float, default=1.)
    p.add_argument('--reg', type=float, default=0.001)
    p.add_argument('--dvar', type=float, default=0.5)
    p.add_argument('--ddist', type=float, default=1.5)

    a = p.parse_args()

    if not os.path.isdir(a.src):
        raise IOError('No directory')
    if not os.path.isdir(a.model):
        raise IOError('No directory')
    if not os.path.isdir(a.log):
        os.mkdir(a.log)

    img_size = (512, 512)
    img_paths = sorted(glob(os.path.join(a.src, 'images', '*.png')))
    lbl_paths = sorted(glob(os.path.join(a.src, 'labels', '*.png')))

    Xt, Xv, yt, yv = train_test_split(img_paths, lbl_paths, test_size=0.10, random_state=42)

    print ('Train samples', len(yt))
    print ('Valid samples', len(yv))

    dbg = True
    bw = 0.7
    cc = 5000
    ec = 1000
    sc = 15000

    ep = a.epochs
    bs = 1
    lr = 1e-4
    lrd = 0.96
    lrdi = 5000

    fd = 3
    pv = a.var
    pd = a.dist
    pr = a.reg
    dv = a.dvar
    dd = a.ddist

    ps = f'fd{fd}_v{pv}_d{pd}_r{pr}_dv{dv}_dd{dd}_lr{lr}_b{bs}'

    if not os.path.exists(os.path.join(a.log, ps)):
        os.makedirs(os.path.join(a.log, ps))            

    cfg = tf.ConfigProto()

    with tf.Session(config=cfg) as s:

        inp = tf.placeholder(tf.float32, shape=(None, img_size[1], img_size[0], 3))
        lbl = tf.placeholder(dtype=tf.float32, shape=(None, img_size[1], img_size[0]))

        lp = utils.load_enet(s, a.model, inp, bs)
        pred = utils.add_transfer_layers(s, lp, fd)

        print ('Params', utils.count_params())
        gs = tf.Variable(0, trainable=False)
        s.run(gs.initializer)
        lr = tf.train.exponential_decay(lr, gs, lrdi, lrd, staircase=True)
        
        tv = utils.get_train_vars(s)

        dl, lv, ld, lr = dl(pred, lbl, fd, img_size, dv, dd, pv, pd, pr)
        with tf.name_scope('Inst/Adam'):
            to = tf.train.AdamOptimizer(learning_rate=lr).minimize(dl, var_list=tv, global_step=gs)
        ai = [v.initializer for v in tf.global_variables() if 'Adam' in v.name]
        s.run(ai)

        st, sv = utils.collect_summaries(dl, lv, ld, lr, inp, pred, lbl)

        tw = tf.summary.FileWriter(a.log)

        vi, vl = datagenerator.get_val_batch(a.src, img_size)

        saver = tf.train.Saver()
        st_tr = 0
        st_v = 0
        for e in range(ep):
            print ('epoch', e)
            
            tl = 0
            for im, lb in datagenerator.get_batches(bs, img_size, Xt, yt):

                clr = s.run(lr)
                
                if (st_tr%ec!=0):
                    _, sp, sl, sv, sd, sr = s.run([to, pred, dl, lv, ld, lr], 
                                            feed_dict={inp: im, lbl: lb})
                else:
                    print ('Evaluating...')
                    _, sm, sp, sl, sv, sd, sr = s.run([to, st, pred, dl, lv, ld, lr], 
                                        feed_dict={inp: im, lbl: lb})                 
                    tw.add_summary(sm, st_tr)

                    vp = s.run(pred, feed_dict={inp: np.expand_dims(vi[0], axis=0), 
                                             lbl: np.expand_dims(vl[0], axis=0)})
                    visualize.eval_plot(a.log, vp, vl, fd, ps, st_tr)
                    
                    if (st_tr%cc==0):
                        if dbg:
                            ims = clster.get_masks(vp, bw)
                            for i, m in enumerate(ims):
                                cv2.imwrite(os.path.join(a.log, ps, f'c_{str(st_tr).zfill(6)}_{i}.png'), m)

                st_tr += 1
                
                if (st_tr%sc==(sc-1)):
                    try:
                        print ('Saving...')
                        saver.save(s, os.path.join(a.out, 'model.ckpt'), global_step=st_tr)
                    except:
                        print ('Save failed')
                print ('step', st_tr, 'loss', sl, 'var', sv, 'dist', sd, 'reg', sr, 'lr', clr)

            print ('Validating...')
            for im, lb in datagenerator.get_batches(bs, img_size, Xv, yv):
                if st_v%100==0:
                    sm, sp, sl, sv, sd, sr = s.run([sv, pred, dl, lv, ld, lr], 
                                        feed_dict={inp: im, lbl: lb})
                    tw.add_summary(sm, st_v)
                else:
                    sp, sl, sv, sd, sr = s.run([pred, dl, lv, ld, lr], 
                                        feed_dict={inp: im, lbl: lb})
                st_v += 1

                print ('valid', st_v, 'loss', sl, 'var', sv, 'dist', sd, 'reg', sr)

        saver = tf.train.Saver()
        saver.save(s, os.path.join(a.out, 'model.ckpt'), global_step=st_tr)

if __name__ == '__main__':
    main()