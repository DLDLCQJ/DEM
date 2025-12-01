import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(
        description='DEM implementation in sMRI using PyTorch.')
    parser.add_argument('--path', type=str, default='/../..', help='Path of data files')
    parser.add_argument('--src_img_file', type=str, default='XXX', help='Path of data files')
    parser.add_argument('--src_label_file', type=str, default="XXX", help='Path of label files')
    parser.add_argument('--tgt_img_file', type=str, default="XXX", help='Path of data files')
    parser.add_argument('--tgt_label_file', type=str, default="XXX", help='Path of label files')
    parser.add_argument('--test_img_file1', type=str, default="XXX", help='Path of data files')
    parser.add_argument('--test_label_file1', type=str, default="XXX", help='Path of label files')
    parser.add_argument('--test_img_file2', type=str, default="XXX", help='Path of data files')
    parser.add_argument('--test_label_file2', type=str, default="XXX", help='Path of label files')
    parser.add_argument('--desired_shape', type=list, default=[128,128], help='target shape of input image')
    #parser.add_argument('--crop_shape', type=list, default=[128,128], help='target shape of input image')
    # parser.add_argument('--num_epochs_cluster', type=int, default=100,
    #                     help='Number of epochs for clutering')
    # parser.add_argument('--num_epochs_inittrain', type=int, default=100,
    #                     help='Number of epochs for clutering')
    parser.add_argument('--num_epochs_train', type=int, default=100,  
                        help='Number of epochs for training')
    parser.add_argument('--num_epochs_pretrain', type=int, default=100,
                        help='Number of epochs for pretraining')
    parser.add_argument('--num_epochs_rl', type=int, default=100,
                        help='Number of epochs for pretraining')
    parser.add_argument('--num_epochs', type=int, default=100,
                        help='Number of epochs for training')
    # parser.add_argument('--n_iterations', type=int, default=10,
    #                     help='Number of clustering')
    parser.add_argument('--batch_size', type=int, default=64, 
                        help='Batch size for training')
    parser.add_argument('--mini_sample_size', type=int, default=10,
                        help='Batch size for training')
    # parser.add_argument('--hidden_size', type=int, default=128, metavar='N', 
    #                     help='What would you like to classify?')
    parser.add_argument('--clinical_features', type=int, default=19, metavar='N',
                        help='What would you like to classify?')
    parser.add_argument('--num_classes', type=int, default=1, metavar='N',
                        help='What would you like to classify?')
    parser.add_argument('--epsilon_labels', type=float, default=0.1, 
                        help='Epsilon parameter for label smoothing')
    # parser.add_argument('--clustering', type=str, default='True', choices=['True', 'False'],
    #                     help='clustering for pseudo labels')
    # parser.add_argument('--cluster_method', type=str, default='agg', choices=['kmeans', 'spectral','agg'],
    #                     help='clustering methods')
    # parser.add_argument('--clustering_step', type=int, default=5, metavar='N',
    #                     help='What would you like to classify?')
    # parser.add_argument('--confidence_scores', type=int, default=0.80,
    #                     help='confidence_scores for models filtering')
    # parser.add_argument('--embedding_dim', type=int, default=768,
    #                     help='Embedding dimension')   
    parser.add_argument('--continual_type', type=str, default='CRL', choices=['CRL','RL'],
                        help='continual learning or continual reinforcement learning')
    parser.add_argument('--pretrained', type=str, default=True, metavar='str',
                        help='Loading pretrained model (default: False)')
    parser.add_argument('--adaptive', type=str, default=False,
                        help='Enable CRL with domain adaptation modules (default: True).')
    parser.add_argument('--reinitial', type=str, default='cbp',choices=["cbp", "ewc","scratch"],
                        help='Enable CRL with triggered reinitialization methods (default: True).')
    parser.add_argument('--frozen', type=str, default=True,
                        help='Enable CRL with frozen (default: True).')
    parser.add_argument('--probs_mapping', type=str, default=True,
                        help='Enable CRL with probability mapping (default: False).')
    parser.add_argument('--c', type=float, default=0.1,
                        help='Regularization parameter') 
    parser.add_argument('--fold', type=float, default=5,
                        help='the number of nfold in Cross_validation')
    parser.add_argument('--num_thre_iterates', type=int, default=50,
                        help='Number of interates to reatrain data')
    parser.add_argument('--num_evol_iterates', type=int, default=50,
                        help='Number of interates to reatrain data')
    parser.add_argument('--buffer_state_width', type=int, default=25,
                        help='width of beam searching to reatrain data')
    parser.add_argument('--initial_percentile', type=int, default=40,
                        help='initial_percentile of beam searching to reatrain data')
    parser.add_argument('--MUT_RATE', type=float, default=0.3, 
                        help='the rate of genetic mutate')
    parser.add_argument('--optim', type=str, default='adam',
                        help='Optimizer parameters')  
    parser.add_argument('--betas', type=float, default=(0.9,0.999),
                        help='Optimizer parameters') 
    parser.add_argument('--epsilon', type=float, default=1e-08,
                        help='Optimizer parameters')
    parser.add_argument('--weight_decay', type=float, default=0.0,
                        help='Optimizer parameters')
    parser.add_argument('--init', type=str, default='kaiming',
                        help='CBP parameters')       
    parser.add_argument('--util_type', type=str, default='contribution',
                        help='CBP parameters')    
    parser.add_argument('--decay_rate', type=float, default=0.99,
                        help='CBP parameters') 
    parser.add_argument('--replacement_rate', type=float, default=0.01,
                        help='CBP parameters')  
    parser.add_argument('--maturity_threshold', type=int, default=100,
                        help='CBP parameters')        
    parser.add_argument('--norm_type', type=str, default='demean_std', choices=["demean_std", "minmax","None"],
                        help='help=How to preprocess data: demean_std, minmax or None')
    parser.add_argument('--rg_type', type=str, default='KM', choices=['L1', 'L2', 'L1L2','HybridReg'],
                        help='Regularization type to use: L1 (LASSO), L2 (Ridge), Elastic net (beta*L2 + L1) or HybridReg')
    parser.add_argument('--kernel', type=int, default=16, metavar='N',
                        help='the kernel size (default: 16')
    parser.add_argument('--network', type=str, default='Mobilenet', metavar='str',
                        help='the network name (default: [DeiT, Googlenet, Alexnet, Mobilenet,VGG, Resnet, Densenet, Inception])')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of workers to use in data loading')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help='Number of workers to use in data loading')
    parser.add_argument('--lr_pre', type=float, default=1e-4,
                        help='Pretrain learning rate for training')
    parser.add_argument('--lr_init', type=float, default=1e-4,
                        help='Initial learning rate for training')
    parser.add_argument('--lr_re', type=float, default=1e-4,
                        help='Reatrain learning rate for training')
    parser.add_argument('--lr_rl', type=float, default=1e-2,
                        help='Reatrain learning rate for training') 
    parser.add_argument('--pre_patience', type=int, default=20,
                        help='Early stopping')
    parser.add_argument('--re_patience', type=int, default=20,
                        help='Early stopping')
    parser.add_argument('--patience', type=int, default=10,
                        help='Early stopping')
    parser.add_argument('--lr_scheduler_warmup_ratio', type=int, default=0.1,
                        help='Number of workers to use in data loading')
    parser.add_argument('--lr_scheduler_type', type=str, default='linear',
                        help='help=Type of learning rate (default):[linear, cosine_annealing, cosine_annealing_warm_restarts, reduce_on_plateau]')
    parser.add_argument('--save_dir', type=str, default='/../..', help='Location of checkpoints')
    parser.add_argument('--plotting_dir', type=str, default='/../..', help='Location of checkpoints')
     ## DDP
    parser.add_argument('--gpu', default=None, type=int, help='GPU id to use')
    parser.add_argument('--world-size', default=-1, type=int, help='number of nodes for distributed training')
    parser.add_argument('--rank', default=-1, type=int, help='node rank for distributed training')
    # parser.add_argument('--dis-url', type=str, default='env://', help='url used to set up distributed training')
    parser.add_argument('--dist_backend', default='nccl', type=str, help='distributed backend')
    parser.add_argument('--local-rank', default=-1, type=int, help='local rank for distributed training')
    return parser.parse_args()  
    return parser.parse_args() 
