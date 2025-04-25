import tensorflow as tf
s = tf.contrib.slim

@tf.contrib.slim.add_arg_scope
def pl(x, sc, dec=False):
    if dec:
        return tf.nn.relu(x, name=sc)
    a = tf.get_variable(sc + 'a', x.get_shape()[-1],
                       initializer=tf.constant_initializer(0.0),
                        dtype=tf.float32)
    p = tf.nn.relu(x)
    n = a * (x - abs(x)) * 0.5
    return p + n

def sd(x, p, seed, sc, tr=True):
    if tr:
        kp = 1.0 - p
        sh = x.get_shape().as_list()
        ns = tf.constant(value=[sh[0], 1, 1, sh[3]])
        o = tf.nn.dropout(x, kp, ns, seed=seed, name=sc)
        return o
    return x

def up(u, m, ks=[1, 2, 2, 1], os=None, sc=''):
    with tf.variable_scope(sc):
        m = tf.cast(m, tf.int32)
        ish = tf.shape(u, out_type=tf.int32)
        if os is None:
            os = (ish[0], ish[1] * ks[1], ish[2] * ks[2], ish[3])
        lm = tf.ones_like(m, dtype=tf.int32)
        bs = tf.concat([[ish[0]], [1], [1], [1]], 0)
        br = tf.reshape(tf.range(os[0], dtype=tf.int32), shape=bs)
        b = lm * br
        y = m // (os[2] * os[3])
        x = (m // os[3]) % os[2]
        fr = tf.range(os[3], dtype=tf.int32)
        f = lm * fr
        us = tf.size(u)
        idx = tf.transpose(tf.reshape(tf.stack([b, y, x, f]), [4, us]))
        v = tf.reshape(u, [us])
        r = tf.scatter_nd(idx, v, os)
        return r

@tf.contrib.slim.add_arg_scope
def ib(inp, tr=True, sc='ib'):
    nc = s.conv2d(inp, 13, [3,3], stride=2, activation_fn=None, scope=sc+'_c')
    nc = s.batch_norm(nc, is_training=tr, fused=True, scope=sc+'_bn')
    nc = pl(nc, sc+'_p')
    np = s.max_pool2d(inp, [2,2], stride=2, scope=sc+'_mp')
    ncat = tf.concat([nc, np], axis=3, name=sc+'_cat')
    return ncat

@tf.contrib.slim.add_arg_scope
def bn(inp, od, fs, rp, pr=4, seed=0, tr=True, ds=False, us=False, pi=None, os=None, 
       dl=False, dr=None, asym=False, dec=False, sc='bn'):
    rd = int(inp.get_shape().as_list()[3] / pr)
    
    with tf.contrib.slim.arg_scope([pl], decoder=dec):
        if ds:
            nm, pi = tf.nn.max_pool_with_argmax(inp, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME', name=sc+'_mmp')
            ish = inp.get_shape().as_list()
            dp = abs(ish[3] - od)
            pad = tf.convert_to_tensor([[0,0], [0,0], [0,0], [0, dp]])
            nm = tf.pad(nm, paddings=pad, name=sc+'_mpad')
            
            n = s.conv2d(inp, rd, [2,2], stride=2, scope=sc+'_c1')
            n = s.batch_norm(n, is_training=tr, scope=sc+'_bn1')
            n = pl(n, sc+'_p1')
            
            n = s.conv2d(n, rd, [fs, fs], scope=sc+'_c2')
            n = s.batch_norm(n, is_training=tr, scope=sc+'_bn2')
            n = pl(n, sc+'_p2')
            
            n = s.conv2d(n, od, [1,1], scope=sc+'_c3')
            n = s.batch_norm(n, is_training=tr, scope=sc+'_bn3')
            n = pl(n, sc+'_p3')
            
            n = sd(n, p=rp, seed=seed, scope=sc+'_sd')
            n = tf.add(n, nm, name=sc+'_add')
            n = pl(n, sc+'_lp')
            return n, pi, ish
            
        elif dl:
            if not dr:
                raise ValueError('No dilation rate')
            nm = inp
            n = s.conv2d(inp, rd, [1,1], scope=sc+'_c1')
            n = s.batch_norm(n, is_training=tr, scope=sc+'_bn1')
            n = pl(n, sc+'_p1')
            
            n = s.conv2d(n, rd, [fs, fs], rate=dr, scope=sc+'_dc2')
            n = s.batch_norm(n, is_training=tr, scope=sc+'_bn2')
            n = pl(n, sc+'_p2')
            
            n = s.conv2d(n, od, [1,1], scope=sc+'_c3')
            n = s.batch_norm(n, is_training=tr, scope=sc+'_bn3')
            n = pl(n, sc+'_p3')
            
            n = sd(n, p=rp, seed=seed, scope=sc+'_sd')
            n = pl(n, sc+'_p4')
            n = tf.add(nm, n, name=sc+'_add')
            n = pl(n, sc+'_lp')
            return n
            
        elif asym:
            nm = inp
            n = s.conv2d(inp, rd, [1,1], scope=sc+'_c1')
            n = s.batch_norm(n, is_training=tr, scope=sc+'_bn1')
            n = pl(n, sc+'_p1')
            
            n = s.conv2d(n, rd, [fs, 1], scope=sc+'_ac2a')
            n = s.conv2d(n, rd, [1, fs], scope=sc+'_ac2b')
            n = s.batch_norm(n, is_training=tr, scope=sc+'_bn2')
            n = pl(n, sc+'_p2')
            
            n = s.conv2d(n, od, [1,1], scope=sc+'_c3')
            n = s.batch_norm(n, is_training=tr, scope=sc+'_bn3')
            n = pl(n, sc+'_p3')
            
            n = sd(n, p=rp, seed=seed, scope=sc+'_sd')
            n = pl(n, sc+'_p4')
            n = tf.add(nm, n, name=sc+'_add')
            n = pl(n, sc+'_lp')
            return n
            
        elif us:
            if pi == None:
                raise ValueError('No pooling indices')
            if os == None:
                raise ValueError('No output shape')
                
            nu = s.conv2d(inp, od, [1,1], scope=sc+'_mc1')
            nu = s.batch_norm(nu, is_training=tr, scope=sc+'bn1')
            nu = up(nu, pi, output_shape=os, scope='up')
            
            n = s.conv2d(inp, rd, [1,1], scope=sc+'_c1')
            n = s.batch_norm(n, is_training=tr, scope=sc+'_bn2')
            n = pl(n, sc+'_p1')
            
            nush = nu.get_shape().as_list()
            os = [nush[0], nush[1], nush[2], rd]
            os = tf.convert_to_tensor(os)
            fs = [fs, fs, rd, rd]
            fl = tf.get_variable(shape=fs, initializer=tf.glorot_uniform_initializer(seed=None, dtype=tf.dtypes.float32), dtype=tf.float32, name=sc+'_tcf')
            
            n = tf.nn.conv2d_transpose(n, filter=fl, strides=[1,2,2,1], output_shape=os, name=sc+'_tc2')
            n = s.batch_norm(n, is_training=tr, scope=sc+'_bn3')
            n = pl(n, sc+'_p2')
            
            n = s.conv2d(n, od, [1,1], scope=sc+'_c3')
            n = s.batch_norm(n, is_training=tr, scope=sc+'_bn4')
            n = pl(n, sc+'_p3')
            
            n = sd(n, p=rp, seed=seed, scope=sc+'_sd')
            n = pl(n, sc+'_p4')
            n = tf.add(n, nu, name=sc+'_add')
            n = pl(n, sc+'_lp')
            return n
            
        else:
            nm = inp
            n = s.conv2d(inp, rd, [1,1], scope=sc+'_c1')
            n = s.batch_norm(n, is_training=tr, scope=sc+'_bn1')
            n = pl(n, sc+'_p1')
            
            n = s.conv2d(n, rd, [fs, fs], scope=sc+'_c2')
            n = s.batch_norm(n, is_training=tr, scope=sc+'_bn2')
            n = pl(n, sc+'_p2')
            
            n = s.conv2d(n, od, [1,1], scope=sc+'_c3')
            n = s.batch_norm(n, is_training=tr, scope=sc+'_bn3')
            n = pl(n, sc+'_p3')
            
            n = sd(n, p=rp, seed=seed, scope=sc+'_sd')
            n = pl(n, sc+'_p4')
            n = tf.add(nm, n, name=sc+'_add')
            n = pl(n, sc+'_lp')
            return n

def EN(inp, nc, bs, nib=1, str=2, sc=True, re=None, tr=True, scope='EN'):
    ish = inp.get_shape().as_list()
    inp.set_shape(shape=(bs, ish[1], ish[2], ish[3]))

    with tf.variable_scope(scope, reuse=re):
        with tf.contrib.slim.arg_scope([ib, bn], is_training=tr),\
             tf.contrib.slim.arg_scope([s.batch_norm], fused=True), \
             tf.contrib.slim.arg_scope([s.conv2d, s.conv2d_transpose], activation_fn=None):
            n = ib(inp, scope='ib1')
            for i in range(2, max(nib, 1) + 1):
                n = ib(n, scope='ib' + str(i))

            if sc:
                n1 = n

            n, pi1, ish1 = bn(n, od=64, fs=3, rp=0.01, ds=True, scope='b1_0')
            n = bn(n, od=64, fs=3, rp=0.01, scope='b1_1')
            n = bn(n, od=64, fs=3, rp=0.01, scope='b1_2')
            n = bn(n, od=64, fs=3, rp=0.01, scope='b1_3')
            n = bn(n, od=64, fs=3, rp=0.01, scope='b1_4')

            if sc:
                n2 = n

            with tf.contrib.slim.arg_scope([bn], rp=0.1):
                n, pi2, ish2 = bn(n, od=128, fs=3, ds=True, scope='b2_0')
                
                for i in range(2, max(str, 2) + 2):
                    n = bn(n, od=128, fs=3, scope='b'+str(i)+'_1')
                    n = bn(n, od=128, fs=3, dl=True, dr=2, scope='b'+str(i)+'_2')
                    n = bn(n, od=128, fs=5, asym=True, scope='b'+str(i)+'_3')
                    n = bn(n, od=128, fs=3, dl=True, dr=4, scope='b'+str(i)+'_4')
                    n = bn(n, od=128, fs=3, scope='b'+str(i)+'_5')
                    n = bn(n, od=128, fs=3, dl=True, dr=8, scope='b'+str(i)+'_6')
                    n = bn(n, od=128, fs=5, asym=True, scope='b'+str(i)+'_7')
                    n = bn(n, od=128, fs=3, dl=True, dr=16, scope='b'+str(i)+'_8')

            with tf.contrib.slim.arg_scope([bn], rp=0.1, dec=True):
                bs = "b" + str(i + 1)
                n = bn(n, od=64, fs=3, us=True, pi=pi2, os=ish2, scope=bs+'_0')
                if sc:
                    n = tf.add(n, n2, name=bs+'_sc')
                n = bn(n, od=64, fs=3, scope=bs+'_1')
                n = bn(n, od=64, fs=3, scope=bs+'_2')
                
                bs = "b" + str(i + 2)
                n = bn(n, od=16, fs=3, us=True, pi=pi1, os=ish1, scope=bs+'_0')
                if sc:
                    n = tf.add(n, n1, name=bs+'_sc')
                n = bn(n, od=16, fs=3, scope=bs+'_1')

            lg = s.conv2d_transpose(n, nc, [2,2], stride=2, scope='fc')
            pb = tf.nn.softmax(lg, name='lg2sm')

        return lg, pb


def EN_as(wd=2e-4,
          bnd=0.1,
          bne=0.001):
  with tf.contrib.slim.arg_scope([s.conv2d],
                      weights_regularizer=s.l2_regularizer(wd),
                      biases_regularizer=s.l2_regularizer(wd)):
    with tf.contrib.slim.arg_scope([s.batch_norm],
                        decay=bnd,
                        epsilon=bne) as sc:
      return sc