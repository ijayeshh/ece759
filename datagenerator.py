import os
import numpy as np
import cv2
from sklearn.utils import shuffle

_m = np.array([92.14, 103.2, 103.47])
_s = np.array([49.16, 54.91, 59.41])

_clrs = [np.zeros(3),*[np.full(3,x) for x in [20,70,120,170,220]]]

def _gb(bs,shp,ips,lps):
    assert len(ips)==len(lps)
    ips,lps=shuffle(*map(sorted,[ips,lps]))
    for i in range(0,len(ips),bs):
        x,y=[],[]
        for p,q in zip(ips[i:i+bs],lps[i:i+bs]):
            img=cv2.cvtColor(cv2.resize(cv2.imread(p),shp,cv2.INTER_LINEAR),cv2.COLOR_BGR2RGB)
            msk=cv2.resize(cv2.imread(q,cv2.IMREAD_COLOR)[:,:,0],shp,cv2.INTER_NEAREST)
            x.append(img)
            y.append(msk)
        yield np.array(x),np.array(y)

def _gvb(d,shp):
    vp=os.path.join(d,'images','0000.png')
    lp=os.path.join(d,'labels','0000.png')
    x=cv2.cvtColor(cv2.resize(cv2.imread(vp),shp,cv2.INTER_LINEAR),cv2.COLOR_BGR2RGB)
    y=cv2.resize(cv2.imread(lp,cv2.IMREAD_COLOR)[:,:,0],shp,cv2.INTER_NEAREST)
    return np.array([x]),np.array([y])

if __name__=="__main__":
    pass