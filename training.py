"""
main training script
remember to set 'iteration_to_load' variable to 0 when train from scratch
"""
from data_provider import DataProvider
from model_define import model_and_loss
from os.path import join
from time import time
from config import Configs
from keras.optimizers import Adam

if __name__ == '__main__':

    c = Configs()

    model, loss = model_and_loss()

    if c.splice_train:
        real_training_dim = c.training_dim_1 * 2
    else:
        real_training_dim = c.training_dim_1

    if c.load_weight:
        model.load_weights(join(c.saving_folder, 'Dim%dIt%dBsize%dLr%.5f.h5' %
                                                 (real_training_dim, c.iteration_to_load, c.batch_size, c.lr)))

    elif c.transfer_learning:
        model.load_weights(c.transfer_weight, by_name=True)

    model.compile(loss=loss, optimizer=Adam(lr=c.lr))

    data_provider_1 = DataProvider(c.training_data_folder, c.batch_size, c.training_dim_1, stride=c.stride,
                                   splice_train=c.splice_train, side=c.side,
                                   mixing_train=c.mixing_train, model_code=c.model_code)
    data_provider_1.start_loading()

    # multi-scale training
    '''
    data_provider_2 = DataProvider(c.training_data_folder, c.batch_size / 4, c.training_dim_2, stride=c.stride,
                                   splice_train=c.splice_train, side=c.side,
                                   mixing_train=c.mixing_train, model_code=c.model_code)
    data_provider_2.start_loading()
    '''

    start_time = time()

    for iteration in range(c.iterations):

        train_x, train_y = data_provider_1.get_batch()

        print 'start training on mini batch'
        training_loss = model.train_on_batch(train_x, train_y)

        total_iteration = c.iteration_to_load + (iteration + 1)
        print 'iteration: %d, training loss:' % total_iteration, training_loss

        # save model
        if (iteration + 1) % c.record_interval == 0:

            time_used = time() - start_time
            start_time = time()

            print 'save weight to', c.saving_folder
            model.save(join(c.saving_folder, 'Dim%dIt%dBsize%dLr%.5f.h5' %
                            (real_training_dim, total_iteration, c.batch_size, c.lr)))

            with open(join(c.saving_folder, 'record.txt'), 'a+') as f:
                f.write('iteration %d to %d, time spent: %.2f sec, loss: %.2f\n' %
                        (total_iteration - c.record_interval, total_iteration, time_used, training_loss))

    data_provider_1.stop_loading()
    # data_provider_2.stop_loading()


