import tensorflow as tf


# the probability of having/not having an object at (m, n)
def logloss(Ptrue, Pred, szs, eps=10e-10):
    b, h, w, ch = szs
    Pred = tf.clip_by_value(Pred, eps, 1.)
    Pred = -tf.log(Pred)
    Pred = Pred*Ptrue
    Pred = tf.reshape(Pred, (b, h*w*ch))
    Pred = tf.reduce_sum(Pred, 1)
    return Pred


# the error between a warped version of the canonical square
def l1(true, pred, szs):
    b, h, w, ch = szs
    res = tf.reshape(true-pred, (b, h*w*ch))
    res = tf.abs(res)
    res = tf.reduce_sum(res, 1)
    return res


# Ytrue shape -> [b, h, w, 9], in '9' -> [prob_obj, x1, y1, x2, y2, x3, y3, x4, y4]
# Ypred shape -> [b, h, w, 8], in '8' -> [prob_obj, prob_non_obj, v3, v4, v7, v5, v6, v8]
def loss(Ytrue, Ypred):

    # b -> batch size
    b = tf.shape(Ytrue)[0]
    h = tf.shape(Ytrue)[1]
    w = tf.shape(Ytrue)[2]

    obj_probs_true = Ytrue[..., 0]  # [b, h, w], the dimension has been shrank
    obj_probs_pred = Ypred[..., 0]  # [b, h, w]

    non_obj_probs_true = 1. - Ytrue[..., 0]  # [b, h, w]
    non_obj_probs_pred = Ypred[..., 1]  # [b, h, w]

    affine_pred = Ypred[..., 2:]  # [b, h, w , 6]
    pts_true = Ytrue[..., 1:]  # [b, h, w, 8]

    # tf.stack -> make a new axis and put the argument arrays into it
    affinex = tf.stack([tf.maximum(affine_pred[..., 0], 0.), affine_pred[..., 1], affine_pred[..., 2]], 3)  # [b, h, w, 3]
    affiney = tf.stack([affine_pred[..., 3], tf.maximum(affine_pred[..., 4], 0.), affine_pred[..., 5]], 3)  # [b, h, w, 3]

    v = 0.5
    base = tf.stack([[[[v, v, 1, -v, v, 1, -v, -v, 1, v, -v, 1]]]])  # shape=[1, 1, 1, 12], here, 1 is used for affine bias
    base = tf.tile(base, tf.stack([b, h, w, 1]))  # shape = [b, h, w, 12], this is to make the base rectangles to all dimensions

    pts = tf.zeros((b, h, w, 0))  # zeros with shape = [b, h, w ,0], as a container to store the pts information

    for i in range(0, 12, 3):  # 0, 3, 6, 9, for the four base points
        row = base[..., i:i + 3]  # take a single base point -> shape = [b, h, w ,3]
        ptsx = tf.reduce_sum(affinex * row, 3)  # shape = [b, h, w], last channel -> v3*v + v4*v + v7
        ptsy = tf.reduce_sum(affiney * row, 3)  # shape = [b, h, w], last channel -> v5*v + v6*v + v8

        pts_xy = tf.stack([ptsx, ptsy], 3)  # shape = [b, h, w, 2]
        pts = tf.concat([pts, pts_xy], 3)  # final shape = [b, h, w, 8] after 4 times of concat (+2 for each loop)

    flags = tf.reshape(obj_probs_true, (b, h, w, 1))
    res = 1.*l1(pts_true * flags, pts * flags, (b, h, w, 4 * 2))
    res += 1.*logloss(obj_probs_true, obj_probs_pred, (b, h, w, 1))
    res += 1.*logloss(non_obj_probs_true, non_obj_probs_pred, (b, h, w, 1))

    return res