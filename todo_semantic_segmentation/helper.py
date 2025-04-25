import os
from glob import glob
import numpy as np
import random
from sklearn.utils import shuffle
import shutil
import time
import tensorflow as tf
import cv2

m = 0.
s = 1.

def gbf(bs, ish, ips, lps):
    ips.sort()
    lps.sort()
    bc = np.array([0, 0, 0])
    ips, lps = shuffle(ips, lps)

    for bi in range(0, len(ips), bs):
        ims = []
        gis = []
        for ifl, gfl in zip(ips[bi:bi+bs], lps[bi:bi+bs]):
            im = cv2.resize(cv2.imread(ifl), (ish[1], ish[0]), cv2.INTER_LINEAR)
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            
            gi = cv2.resize(cv2.imread(gfl, cv2.IMREAD_COLOR), (ish[1], ish[0]), cv2.INTER_NEAREST)
            gb = np.all(gi == bc, axis=2)
            gb = gb.reshape(gb.shape[0], gb.shape[1], 1)
            gi = np.concatenate((gb, np.invert(gb)), axis=2)

            ims.append(im)
            gis.append(gi)

        yield np.array(ims), np.array(gis)

def ag(im, gm=1.0):
    ig = 1.0 / gm
    tbl = np.array([((i / 255.0) ** ig) * 255
        for i in np.arange(0, 256)]).astype("float32")
    return cv2.LUT(im, tbl)

def gto(sess, lg, kp, ipl, df, ish):
    for ifl in glob(os.path.join(df, 'test_images', '*.png'))[:40]:
        im = cv2.resize(cv2.imread(ifl), (ish[1], ish[0]), cv2.INTER_LINEAR)
        io = im.copy()
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

        ism = sess.run(tf.nn.softmax(lg), {kp: 1.0, ipl: [im]})
        ism = ism[:, 1].reshape(ish[0], ish[1])
        mi = np.where(ism > 0.3)

        bl = np.zeros_like(io)
        bl[mi] = np.array([0,255,0])
        bld = cv2.addWeighted(io, 1, bl, 0.7, 0)
        bld = cv2.cvtColor(bld, cv2.COLOR_BGR2RGB)

        yield os.path.basename(ifl), np.array(bld)

def sis(rd, dd, sess, ish, lg, kp, ii):
    od = os.path.join(rd, str(time.time()))
    if os.path.exists(od):
        shutil.rmtree(od)
    os.makedirs(od)

    print('Saving to: {}'.format(od))
    ios = gto(sess, lg, kp, ii, dd, ish)
    for n, im in ios:
        cv2.imwrite(os.path.join(od, n), im)