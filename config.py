class Configs:
    def __init__(self):
        """
        All manual settings are here
        """

        """MODEL, in ['Hourglass+Vernex_lpfr', 'Hourglass+Vernex_lp', 'Hourglass+WPOD', 'WPOD+WPOD']"""
        self.model_code = 'Hourglass+Vernex_lpfr'
        """DATA SET, in ['CCPD_FR', 'vernex']"""
        self.dataset_code = 'vernex'

        """Training configs"""
        self.training_dim_1 = 256
        self.training_dim_2 = 512
        self.training_dim_3 = 1024
        self.stride = 4
        self.out_dim = self.training_dim_1 / self.stride
        self.iterations = 300000
        self.batch_size = 16
        self.lr = 0.00025
        self.record_interval = 1000
        self.splice_train = False
        self.side = 1.
        self.mixing_train = True
        self.training_data_folder = '/home/shaoheng/Documents/Thesis_KSH/training_data/vernex'
        self.saving_folder = '/home/shaoheng/Documents/Thesis_KSH/Link to training_result/vernex_lpfr_fr_aux_class'
        """Weight loading"""
        self.load_weight = True
        self.iteration_to_load = 517000
        """Transfer learning"""
        self.transfer_learning = False
        self.transfer_weight = '/home/shaoheng/Documents/Thesis_KSH/Link to training_result/maximun_please/ini-Dim256It170000Bsize16Lr0.00025.h5'

        """Testing configs"""
        self.prob_threshold = 0.3
        self.use_nms = True
        self.input_norm = True
        self.test_input_dim = (512, 512)
        self.LPs_to_find = 10
        self.weight = '/home/shaoheng/Documents/Thesis_KSH/Link to training_result/vernex_lpfr_fr_aux_class/Dim256It517000Bsize16Lr0.00025.h5'
        self.input_dir = '/home/shaoheng/Documents/Thesis_KSH/samples/cn'
        self.output_dir = '/home/shaoheng/Documents/Thesis_KSH/output_results/cn'

