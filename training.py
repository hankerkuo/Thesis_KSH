"""
main training script
remember to set 'iteration_to_load' variable to 0 when train from scratch
"""
import lpdetect_model
from loss import loss
from keras.optimizers import Adam
from label_processing import DataProvider
from src_others.hourglass import create_hourglass_network, bottleneck_block
from os.path import join
from time import time

if __name__ == '__main__':

    '''Config'''
    training_dim = 256
    stride = 4
    out_dim = training_dim / stride
    iterations = 300000
    batch_size = 16
    lr = 0.00025
    optimizer = Adam(lr=lr)
    record_interval = 1000
    splice_train = False
    CCPD_origin = False
    side = 16.

    '''WPOD
    model = lpdetect_model.model_WPOD()
    '''

    '''Hourglass'''
    model = create_hourglass_network(None, num_stacks=2, num_channels=256, inres=(training_dim, training_dim),
                                     outres=(out_dim, out_dim), bottleneck=bottleneck_block)

    if splice_train:
        real_training_dim = training_dim * 2
    else:
        real_training_dim = training_dim

    training_data_folder = '/home/shaoheng/Documents/Thesis_KSH/training_data/CCPD_FR_total2333'
    saving_folder = '/home/shaoheng/Documents/Thesis_KSH/Link to training_result/CCPD_FR_2333_AUG_hourglass'

    iteration_to_load = 0
    # comment this line if train from scratch
    # model.load_weights(join(saving_folder, 'Dim%dIt%dBsize%dLr%.5f.h5' %
    #                                        (real_training_dim, iteration_to_load, batch_size, lr)))

    # if use transfer learning
    model.load_weights('/home/shaoheng/Documents/Thesis_KSH/Link to training_result/CCPD_FR_2333_AUG_hourglass/ini_Dim256It123000Bsize16Lr0.00025.h5')

    model.compile(loss=loss, optimizer=optimizer)

    data_provider = DataProvider(training_data_folder, batch_size, training_dim, stride=stride,
                                 CCPD_origin=CCPD_origin, splice_train=splice_train, side=side)
    data_provider.start_loading()

    start_time = time()

    for iteration in range(iterations):
        train_x, train_y = data_provider.get_batch()

        print 'start training on mini batch'
        training_loss = model.train_on_batch(train_x, train_y)

        total_iteration = iteration_to_load + (iteration + 1)
        print 'iteration: %d, training loss:' % total_iteration, training_loss

        # save model
        if (iteration + 1) % record_interval == 0:

            time_used = time() - start_time
            start_time = time()

            print 'save weight to', saving_folder
            model.save_weights(join(saving_folder, 'Dim%dIt%dBsize%dLr%.5f.h5' %
                                    (real_training_dim, total_iteration, batch_size, lr)))

            with open(join(saving_folder, 'record.txt'), 'a+') as f:
                f.write('iteration %d to %d, time spent: %.2f sec, loss: %.2f\n' %
                        (total_iteration - record_interval, total_iteration, time_used, training_loss))

    data_provider.stop_loading()


