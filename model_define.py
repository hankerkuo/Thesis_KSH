from models.Vernex_lp import create_hourglass_network_vernex_lp, bottleneck_block
from models.Vernex_lpfr import create_hourglass_network_vernex_lpfr
from models.hourglass_WPOD import create_hourglass_network_WPOD
from models.WPOD_WPOD import model_WPOD
from loss import loss_WPOD, loss_Vernex_lp, loss_Vernex_lpfr
from config import Configs


def model_and_loss():
    c = Configs()
    '''model choosing'''
    assert c.model_code in ['WPOD+WPOD', 'Hourglass+WPOD', 'Hourglass+Vernex_lp', 'Hourglass+Vernex_lpfr'], \
        'model code not defined!'

    if c.model_code == 'WPOD+WPOD':
        model = model_WPOD()
        loss = loss_WPOD
    elif c.model_code == 'Hourglass+WPOD':
        model = create_hourglass_network_WPOD(None, num_stacks=2, num_channels=256,
                                              inres=(c.training_dim_1, c.training_dim_1),
                                              outres=(c.out_dim, c.out_dim),
                                              bottleneck=bottleneck_block)
        loss = loss_WPOD
    elif c.model_code == 'Hourglass+Vernex_lp':
        model = create_hourglass_network_vernex_lp(None, num_stacks=2, num_channels=256,
                                                   inres=(c.training_dim_1, c.training_dim_1),
                                                   outres=(c.out_dim, c.out_dim),
                                                   bottleneck=bottleneck_block)
        loss = loss_Vernex_lp
    elif c.model_code == 'Hourglass+Vernex_lpfr':
        model = create_hourglass_network_vernex_lpfr(None, num_stacks=2, num_channels=256,
                                                     inres=(c.training_dim_1, c.training_dim_1),
                                                     outres=(c.out_dim, c.out_dim),
                                                     bottleneck=bottleneck_block)
        loss = loss_Vernex_lpfr

    return model, loss