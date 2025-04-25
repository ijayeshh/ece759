import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

def save_img(vi, vl):
    assert len(vi.shape)==3 and len(vl.shape)==2
    nu = np.unique(vl)
    b = vi
    for cid, u in enumerate(list(nu[1:])):
        ii = np.where(vl==u)
        a = np.zeros_like(vi)
        a[ii] = np.array([cid*70, cid*70, 255-cid*50])
        b = cv2.addWeighted(b, 1, a, 1, 0)
    b = cv2.cvtColor(b, cv2.COLOR_RGB2BGR)
    cv2.imwrite('overlay.png', b)   

def eval_plot(ld, vp, vl, fd, ps, st):
    assert len(vp.shape)==4 and len(vl.shape)==3
    assert vp.shape[3]==fd

    fig = plt.figure()
    if fd==2:
        nu = np.unique(vl[0])
        for u in list(nu):
            ii = np.where(vl[0]==u)
            x = vp[0,:,:,0][ii]
            y = vp[0,:,:,1][ii]
            plt.plot(x, y, 'o')

    elif fd==3:
        ax = fig.add_subplot(1,1,1, projection='3d')
        nu = np.unique(vl[0])
        cs = [(0., 0., 1., 0.05), 'g', 'r', 'c', 'm', 'y']
        for cid, u in enumerate(list(nu)):
            ii = np.where(vl[0]==u)
            x = vp[0,:,:,0][ii]
            y = vp[0,:,:,1][ii]
            z = vp[0,:,:,2][ii]
            ax.scatter(x, y, z, c=cs[cid])
    elif fd > 3:
        plt.close(fig)
        return None

    plt.savefig(os.path.join(ld, ps, 'c_{}.png'.format(str(st).zfill(6))), bbox_inches='tight')
    plt.close(fig)