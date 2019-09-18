import tensorflow as tf


# focal loss for localization
def focal_loss(ptrue, ppred, szs, eps=10e-10, alpha=0.25, gama=2.0):
    b, h, w, ch = szs
    ppred = ppred * ptrue + (1 - ptrue) - (1 - ptrue) * ppred
    ppred = tf.clip_by_value(ppred, eps, 1.)
    ppred = -alpha * tf.pow(1 - ppred, gama) * tf.log(ppred)
    ppred = tf.reshape(ppred, (b, h * w * ch))
    ppred = tf.reduce_sum(ppred, axis=1)
    return ppred


def entropy_loss(ptrue, ppred, szs, eps=10e-10):
    b, h, w, ch = szs
    ppred = ppred * ptrue + (1 - ptrue) - (1 - ptrue) * ppred
    ppred = tf.clip_by_value(ppred, eps, 1.)
    ppred = -tf.log(ppred)
    ppred = tf.reshape(ppred, (b, h * w * ch))
    ppred = tf.reduce_sum(ppred, axis=1)
    return ppred


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


# Ytrue shape -> [b, h, w, 18], in '18' -> [prob_lp, prob_fr, (rxi, ryi) for i in range(4)
#                                                             (rxj, ryj) for j in range(4)]
# Ypred shape -> [b, h, w, 17], in '17' -> [prob_lp, prob_fr, (rxi, ryi) for i in range(4)
#                                                             (rxj, ryj) for j in range(4)]
def loss_Vernex_lpfr(Ytrue, Ypred):
    # b -> batch size
    b = tf.shape(Ytrue)[0]
    h = tf.shape(Ytrue)[1]
    w = tf.shape(Ytrue)[2]

    prob_lp_true = Ytrue[..., 0]  # [b, h, w], the dimension has been shrank
    prob_fr_true = Ytrue[..., 1]  # [b, h, w]
    prob_lp_pred = Ypred[..., 0]  # [b, h, w]
    prob_fr_pred = Ypred[..., 1]  # [b, h, w]

    ratio_lp_pred = Ypred[..., 2:10]  # [b, h, w, 8], ratio for lp vertex expanding
    ratio_fr_pred = Ypred[..., 10:18]  # [b, h, w, 8], ratio for fr vertex expanding

    pts_lp_true = Ytrue[..., 2:10]  # [b, h, w, 8]
    pts_fr_true = Ytrue[..., 10:18]

    base = tf.stack([[[[1., 1., -1., 1., -1., -1., 1., -1.]]]])  # shape=[1, 1, 1, 8]
    base = tf.tile(base,
                   tf.stack([b, h, w, 1]))  # shape = [b, h, w, 8], this is to make the base vectors to all dimensions

    pts_lp_pred = tf.zeros((b, h, w, 0))  # zeros with shape = [b, h, w ,0], as a container to store the pts information
    pts_fr_pred = tf.zeros((b, h, w, 0))  # zeros with shape = [b, h, w ,0], as a container to store the pts information

    for i in range(0, 8, 2):  # 0, 2, 4, 6, for the four directions, from br and clockwise
        row = base[..., i:i + 2]  # take a single base point -> shape = [b, h, w ,2]
        ratio_lp = ratio_lp_pred[..., i:i + 2]  # shape = [b, h, w ,2]
        ratio_fr = ratio_fr_pred[..., i:i + 2]  # shape = [b, h, w ,2]
        pts_lp_xy = row * ratio_lp  # shape = [b, h, w, 2]
        pts_fr_xy = row * ratio_fr

        pts_lp_pred = tf.concat([pts_lp_pred, pts_lp_xy],
                                3)  # final shape = [b, h, w, 8] after 4 times of concat (+2 for each loop)
        pts_fr_pred = tf.concat([pts_fr_pred, pts_fr_xy],
                                3)  # final shape = [b, h, w, 8] after 4 times of concat (+2 for each loop)

    flags_lp = tf.reshape(prob_lp_true, (b, h, w, 1))
    flags_fr = tf.reshape(prob_fr_true, (b, h, w, 1))

    # focal loss
    focal_lp = 1. * focal_loss(prob_lp_true, prob_lp_pred, (b, h, w, 1))
    focal_fr = 1. * focal_loss(prob_fr_true, prob_fr_pred, (b, h, w, 1))

    # localization loss
    loc_lp = 1. * l1(pts_lp_true * flags_lp / 64., pts_lp_pred * flags_lp / 64., (b, h, w, 4 * 2))
    loc_fr = 1. * l1(pts_fr_true * flags_fr / 64., pts_fr_pred * flags_fr / 64., (b, h, w, 4 * 2))

    res = focal_lp + focal_fr + loc_lp + loc_fr

    return res


# Ytrue shape -> [b, h, w, 9], in '9' -> [prob_obj, x1, y1, x2, y2, x3, y3, x4, y4]
# Ypred shape -> [b, h, w, 8], in '8' -> [prob_obj, prob_non_obj, v3, v4, v7, v5, v6, v8]
def loss_WPOD(Ytrue, Ypred):

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


# Ytrue shape -> [b, h, w, 9], in '9' -> [prob_obj, x1, y1, x2, y2, x3, y3, x4, y4]
# Ypred shape -> [b, h, w, 8], in '8' -> [prob_obj, prob_non_obj, rx1, ry1, rx2, ry2, rx3, ry3, rx4, ry4]
def loss_Vernex_lp(Ytrue, Ypred):

    # b -> batch size
    b = tf.shape(Ytrue)[0]
    h = tf.shape(Ytrue)[1]
    w = tf.shape(Ytrue)[2]

    obj_probs_true = Ytrue[..., 0]  # [b, h, w], the dimension has been shrank
    obj_probs_pred = Ypred[..., 0]  # [b, h, w]

    non_obj_probs_true = 1. - Ytrue[..., 0]  # [b, h, w]
    non_obj_probs_pred = Ypred[..., 1]  # [b, h, w]

    ratio_pred = Ypred[..., 2:]  # [b, h, w, 8], ratio for vertex expanding
    pts_true = Ytrue[..., 1:]  # [b, h, w, 8]

    base = tf.stack([[[[1., 1., -1., 1., -1., -1., 1., -1.]]]])  # shape=[1, 1, 1, 8]
    base = tf.tile(base, tf.stack([b, h, w, 1]))  # shape = [b, h, w, 8], this is to make the base vectors to all dimensions

    pts = tf.zeros((b, h, w, 0))  # zeros with shape = [b, h, w ,0], as a container to store the pts information

    for i in range(0, 8, 2):  # 0, 2, 4, 6, for the four directions, from br and clockwise
        row = base[..., i:i + 2]  # take a single base point -> shape = [b, h, w ,2]
        ratio = ratio_pred[..., i:i + 2]  # shape = [b, h, w ,2]
        pts_xy = row * ratio  # shape = [b, h, w, 2]
        pts = tf.concat([pts, pts_xy], 3)  # final shape = [b, h, w, 8] after 4 times of concat (+2 for each loop)

    flags = tf.reshape(obj_probs_true, (b, h, w, 1))
    res = 1.*l1(pts_true * flags / 64., pts * flags / 64., (b, h, w, 4 * 2))
    res += 1.*focal_loss(obj_probs_true, obj_probs_pred, (b, h, w, 1))
    # res += 1.*logloss(non_obj_probs_true, non_obj_probs_pred, (b, h, w, 1))

    return res