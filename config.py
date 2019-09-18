class Configs:
    def __init__(self):
        """MODEL, in ['Hourglass+Vernex_lpfr', 'Hourglass+Vernex_lp', 'Hourglass+WPOD', 'WPOD+WPOD']"""
        self.model_code = 'Hourglass+Vernex_lp'

        """Training configs"""
        self.training_dim = 256
        self.stride = 4
        self.out_dim = self.training_dim / self.stride
        self.iterations = 300000
        self.batch_size = 16
        self.lr = 0.00025
        self.record_interval = 1000
        self.splice_train = False
        self.CCPD_origin = False
        self.side = 1.
        self.mixing_train = True
        self.training_data_folder = '/home/shaoheng/Documents/Thesis_KSH/training_data/CCPD_FR'
        self.saving_folder = '/home/shaoheng/Documents/Thesis_KSH/Link to training_result/vernex_lp_focal'
        """Weight loading"""
        self.load_weight = True
        self.iteration_to_load = 1000
        """Transfer learning"""
        self.transfer_learning = False
        self.transfer_weight = '/home/shaoheng/Documents/Thesis_KSH/Link to training_result/CCPD_FR&openALPR/ini_Dim256It300000Bsize16Lr0.00025.h5'

        """Testing configs"""
        self.prob_threshold = 0.2
        self.use_nms = True
        self.input_norm = True
        self.test_input_dim = (256, 256)
        self.LPs_to_find = 5
        self.weight = '/home/shaoheng/Documents/Thesis_KSH/Link to training_result/vernex_lp_focal/Dim256It1000Bsize16Lr0.00025.h5'
        self.input_dir = '/home/shaoheng/Documents/Thesis_KSH/samples/tw'
        self.output_dir = '/home/shaoheng/Documents/Thesis_KSH/output_results/tw'
