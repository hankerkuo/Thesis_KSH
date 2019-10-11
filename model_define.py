from keras.models import load_model
from keras.optimizers import Adam
from os.path import join

from models.Vernex_lp import create_hourglass_network_vernex_lp, bottleneck_block
from models.Vernex_lpfr import create_hourglass_network_vernex_lpfr
from models.hourglass_WPOD import create_hourglass_network_WPOD
from models.WPOD_WPOD import model_WPOD
from models.WPOD_lpfr import WPOD_lpfr
from loss import loss_WPOD, loss_Vernex_lp, loss_Vernex_lpfr
from config import Configs


def model_and_loss(training=False):
    c = Configs()
    '''model choosing'''
    assert c.model_code in ['WPOD+WPOD', 'Hourglass+WPOD', 'Hourglass+Vernex_lp', 'Hourglass+Vernex_lpfr',
                            'WPOD+vernex_lpfr'], 'model code not defined!'

    # if directly load from h5, then it's a compiled model with training states (loss, optimizer state)
    if training and c.load_model_byh5:
        print 'loading whole model:', 'Dim%dIt%dBsize%dLr%.5f.h5' % \
                                                 (c.training_dim, c.iteration_to_load, c.batch_size, c.lr)
        model = load_model(join(c.saving_folder, 'Dim%dIt%dBsize%dLr%.5f.h5' %
                                                 (c.training_dim, c.iteration_to_load, c.batch_size, c.lr)),
                           custom_objects={'loss_Vernex_lpfr': loss_Vernex_lpfr,
                                           'loss_Vernex_lp': loss_Vernex_lp,
                                           'loss_WPOD': loss_WPOD})
        return model

    # model choosing
    if c.model_code == 'WPOD+WPOD':
        model = model_WPOD()
        loss = loss_WPOD
    elif c.model_code == 'Hourglass+WPOD':
        model = create_hourglass_network_WPOD(None, num_stacks=2, num_channels=256,
                                              inres=(c.training_dim, c.training_dim),
                                              outres=(c.out_dim, c.out_dim),
                                              bottleneck=bottleneck_block)
        loss = loss_WPOD
    elif c.model_code == 'Hourglass+Vernex_lp':
        model = create_hourglass_network_vernex_lp(None, num_stacks=2, num_channels=256,
                                                   inres=(c.training_dim, c.training_dim),
                                                   outres=(c.out_dim, c.out_dim),
                                                   bottleneck=bottleneck_block)
        loss = loss_Vernex_lp
    elif c.model_code == 'Hourglass+Vernex_lpfr':
        model = create_hourglass_network_vernex_lpfr(None, num_stacks=1, num_channels=256,
                                                     inres=(c.training_dim, c.training_dim),
                                                     outres=(c.out_dim, c.out_dim),
                                                     bottleneck=bottleneck_block)
        loss = loss_Vernex_lpfr
    elif c.model_code == 'WPOD+vernex_lpfr':
        model = WPOD_lpfr()
        loss = loss_Vernex_lpfr

    # load weight or not, useful when want to apply different optimizer
    if training:
        if c.train_from_stratch:
            pass
        elif c.load_weight:
            model.load_weights(join(c.saving_folder, 'Dim%dIt%dBsize%dLr%.5f.h5' %
                                                     (c.training_dim, c.iteration_to_load, c.batch_size, c.lr)))
        elif c.transfer_learning:
            model.load_weights(c.transfer_weight, by_name=True)
    elif not training:
        model.load_weights(c.weight)

    model.compile(loss=loss, optimizer=Adam(lr=c.lr))

    return model
