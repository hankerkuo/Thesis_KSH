import lpdetect_model
from loss import loss
from keras.optimizers import Adam
from label_processing import batch_data_generator
from os.path import join

if __name__ == '__main__':

    model = lpdetect_model.model_WPOD()

    training_dim = 208
    iterations = 20000
    batch_size = 64
    optimizer = Adam(lr=0.01)

    training_data_folder = '/home/shaoheng/Documents/Thesis_KSH/training_data/CCPD_FR_total746/'
    saving_folder = '/home/shaoheng/Documents/Thesis_KSH/training_result'

    model.compile(loss=loss, optimizer=optimizer)

    for iteration in range(iterations):
        train_x, train_y = batch_data_generator(training_data_folder, batch_size, training_dim, 16)
        training_loss = model.train_on_batch(train_x, train_y)
        print 'iteration: %d, training loss:' % iteration, training_loss

        # save model
        if (iteration + 1) % 500 == 0:
            print 'save model to', saving_folder
            model.save_weights(join(saving_folder, 'Dim%dIt%dBsize%d.h5' % (training_dim, iteration + 1, batch_size)))
