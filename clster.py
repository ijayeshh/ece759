import os
import numpy as np
from sklearn.cluster import MeanShift
import time
import cv2

CLR = [np.array([255,0,0]), 
       np.array([0,255,0]),
       np.array([0,0,255]),
       np.array([125,125,0]),
       np.array([0,125,125]),
       np.array([125,0,125]),
       np.array([50,100,50]),
       np.array([100,50,100])]

def clst(pred, bw):
    ms = MeanShift(bw, bin_seeding=True)
    t = time.time()
    ms.fit(pred)
    lbl = ms.labels_
    ctr = ms.cluster_centers_
    n = ctr.shape[0]
    return n, lbl, ctr

def get_masks(p, bw):
    bs, h, w, fd = p.shape
    masks = []
    for i in range(bs):
        n, lbl, ctr = clst(p[i].reshape([h*w, fd]), bw)
        lbl = np.array(lbl, dtype=np.uint8).reshape([h,w])
        m = np.zeros([h,w,3], dtype=np.uint8)
        n = min(n,8)
        for mid in range(n):
            idx = np.where(lbl==mid)
            m[idx] = CLR[mid]
        masks.append(m)
    return masks

def save_masks(p, out_dir, bw, cnt):
    bs, h, w, fd = p.shape
    for i in range(bs):
        n, lbl, _ = clst(p[i].reshape([h*w, fd]), bw)
        lbl = np.array(lbl, dtype=np.uint8).reshape([h,w])
        n = min(n,8)
        for mid in range(n):
            m = np.zeros([h,w,3], dtype=np.uint8)
            idx = np.where(lbl==mid)
            m[idx] = np.array([255,255,255])
            fname = os.path.join(out_dir, f'c_{str(cnt).zfill(4)}_{mid}.png')
            cv2.imwrite(fname, m)