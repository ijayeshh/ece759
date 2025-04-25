import tensorflow as tf

def dl_single(pred, lbl, fd, ls, dv, dd, pv, pd, pr):
    lbl = tf.reshape(lbl, [ls[1]*ls[0]])
    rp = tf.reshape(pred, [ls[1]*ls[0], fd])

    ul, ui, cnt = tf.unique_with_counts(lbl)
    cnt = tf.cast(cnt, tf.float32)
    ni = tf.size(ul)

    ss = tf.unsorted_segment_sum(rp, ui, ni)
    mu = tf.div(ss, tf.reshape(cnt, (-1, 1)))
    me = tf.gather(mu, ui)

    d = tf.norm(tf.subtract(me, rp), axis=1)
    d = tf.subtract(d, dv)
    d = tf.clip_by_value(d, 0., d)
    d = tf.square(d)

    lv = tf.unsorted_segment_sum(d, ui, ni)
    lv = tf.div(lv, cnt)
    lv = tf.reduce_sum(lv)
    lv = tf.divide(lv, tf.cast(ni, tf.float32))
    
    mir = tf.tile(mu, [ni, 1])
    mbr = tf.tile(mu, [1, ni])
    mbr = tf.reshape(mbr, (ni*ni, fd))

    md = tf.subtract(mbr, mir)
    
    it = tf.reduce_sum(tf.abs(md),axis=1)
    zv = tf.zeros(1, dtype=tf.float32)
    bm = tf.not_equal(it, zv)
    mdb = tf.boolean_mask(md, bm)

    mn = tf.norm(mdb, axis=1)
    mn = tf.subtract(2.*dd, mn)
    mn = tf.clip_by_value(mn, 0., mn)
    mn = tf.square(mn)

    ld = tf.reduce_mean(mn)

    lr = tf.reduce_mean(tf.norm(mu, axis=1))

    ps = 1.
    lv = pv * lv
    ld = pd * ld
    lr = pr * lr

    l = ps*(lv + ld + lr)
    
    return l, lv, ld, lr


def dl(pred, lbl, fd, ish, dv, dd, pv, pd, pr):
    def c(l, b, ol, ov, od, or_, i):
        return tf.less(i, tf.shape(b)[0])

    def b(l, bt, ol, ov, od, or_, i):
        dl, lv, ld, lr = dl_single(bt[i], l[i], fd, ish, dv, dd, pv, pd, pr)

        ol = ol.write(i, dl)
        ov = ov.write(i, lv)
        od = od.write(i, ld)
        or_ = or_.write(i, lr)

        return l, bt, ol, ov, od, or_, i + 1

    otl = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
    otv = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
    otd = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
    otr = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)

    _, _, olo, ovo, odo, oro, _ = tf.while_loop(c, b, [lbl, pred, otl, otv, otd, otr, 0])
    
    olo = olo.stack()
    ovo = ovo.stack()
    odo = odo.stack()
    oro = oro.stack()
    
    dl = tf.reduce_mean(olo)
    lv = tf.reduce_mean(ovo)
    ld = tf.reduce_mean(odo)
    lr = tf.reduce_mean(oro)

    return dl, lv, ld, lr