class Configs:
    def __init__(self):
        """
        All manual settings are here
        """

        """MODEL, in ['Hourglass+Vernex_lpfr', 'Hourglass+Vernex_lp', 'Hourglass+WPOD', 'WPOD+WPOD', 'WPOD+vernex_lpfr']"""
        self.model_code = 'Hourglass+Vernex_lpfr'
        """DATA SET, in ['CCPD_FR', 'vernex']"""
        self.dataset_code = 'vernex'

        """Training configs"""
        self.training_dim = 256
        self.stride = 4  # 4 for hourglass, 16 for WPOD
        self.out_dim = self.training_dim / self.stride
        self.iterations = 300000
        self.batch_size = 16
        self.lr = 0.00025
        self.record_interval = 1000
        self.side = 1.
        self.mixing_train = True
        self.training_data_folder = '/home/shaoheng/Documents/Thesis_KSH/training_data/vernex'
        self.saving_folder = '/home/shaoheng/Documents/Thesis_KSH/Link to training_result/vernex_stack1_lpfr'
        """Weight loading"""
        self.load_model_byh5 = True
        self.load_weight = False
        self.train_from_stratch = False
        self.iteration_to_load = 139000
        """Transfer learning"""
        self.transfer_learning = False
        self.transfer_weight = '/home/shaoheng/Documents/Thesis_KSH/Link to training_result/final_train/ini_Dim256It598000Bsize16Lr0.00025.h5'
        """Online Benchmark Validation"""
        self.online_val_scale = ((512, 512),)
        self.val_prob_threshold = 0.3
        self.val_LPs_to_find = 10
        self.val_use_nms = True
        self.val_input_norm = True

        """Testing configs"""
        # if single scale testing, then just put one tuple
        self.multi_scales = ((512, 512), (256, 256))

        self.prob_threshold = 0.3
        self.LPs_to_find = 10
        self.use_nms = True
        self.input_norm = True

        self.weight = '/home/shaoheng/Documents/Thesis_KSH/Link to training_result/Dim256It598000Bsize16Lr0.00025.h5'
        self.input_dir = '/home/shaoheng/Documents/Thesis_KSH/benchmark/cd_hard_vernex'
        self.output_dir = '/home/shaoheng/Documents/Thesis_KSH/output_results/vernex_niceresults/lpfr_class/cd_hard_598000_DIM256+512'

        """Benchmark configs"""
        self.info_saving_folder = '/home/shaoheng/Documents/Thesis_KSH/benchmark/mAP_weights/vernex_stack1_lpfr_dim512_thres0.3'  # MOST IMPORTANT
        self.weight_folder_to_eval = '/home/shaoheng/Documents/Thesis_KSH/Link to training_result/WPOD_vernex_lpfr'  # SINGLE
        self.valid_data_folder = '/home/shaoheng/Documents/Thesis_KSH/benchmark/cd_hard_vernex'
        self.temp_outout_folder = '/home/shaoheng/Documents/Thesis_KSH/benchmark/output_results/cd_hard_vernex'



