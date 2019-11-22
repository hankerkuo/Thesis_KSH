"""
main training script
remember to set 'iteration_to_load' variable to 0 when train from scratch
"""
from os.path import join
from time import time

from src.data_provider import DataProvider
from src.model_define import model_and_loss
from config import Configs
from model_evaluation.weight_evaluation import test_on_benchmark


if __name__ == '__main__':

    c = Configs()

    model = model_and_loss(training=True)

    data_provider = DataProvider(c.training_data_folder, c.batch_size, c.training_dim, stride=c.stride,
                                 side=c.side, mixing_train=c.mixing_train, model_code=c.model_code)
    data_provider.start_loading()

    start_time = time()

    for iteration in range(c.iterations):

        train_x, train_y = data_provider.get_batch()

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
                            (c.training_dim, total_iteration, c.batch_size, c.lr)))

            with open(join(c.saving_folder, 'record.txt'), 'a+') as f:
                f.write('iteration %d to %d, time spent: %.2f sec, loss: %.2f\n' %
                        (total_iteration - c.record_interval, total_iteration, time_used, training_loss))

            # test on benchmark and save to .txt
            test_on_benchmark(model=model,
                              weight_name='Dim%dIt%dBsize%dLr%.5f.h5' %
                              (c.training_dim, total_iteration, c.batch_size, c.lr),
                              weight_folder=None,
                              load_weight=False)

    data_provider.stop_loading()


