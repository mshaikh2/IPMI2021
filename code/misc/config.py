#### used in main project, do not change ####
class Config(object):
    def __init__(self):
        self.DATASET_NAME = 'OpenI'
        self.CUDA = True
        self.snapshot_interval = 10
        self.text_encoder_path = ''
        self.CONFIG_NAME = 'test'
        self.DATA_DIR = '../'
        self.TRAIN = True
        self.GPU_ID = 1
        self.GAMMA1 = 4.0
        self.GAMMA2 = 5.0
        self.GAMMA3 = 10.0
        self.clip_max_norm = 1.0
        # Learning Rates
        self.lr_backbone = 0
        self.lr = 5e-5

        # Epochs
        self.epochs = 100
        self.lr_drop = 30
        self.lr_gamma = 0.2
        self.start_epoch = 0
        self.weight_decay = 1e-4

        # Backbone
        self.backbone = 'v3'
        self.position_embedding = 'sine'
#         self.dilation = True
        
        # Basic
        self.device = 'cuda:3'
        self.seed = 42
        self.batch_size = 32
        self.val_batch_size = 50
        self.num_workers = 4
#         self.checkpoint = 'catr/checkpoints/catr_damsm256_proj_coco2014_ep02.pth'
        self.checkpoint = './checkpoint.pth'
        
        
        self.prefix = 'catr_damsm256_proj'

        # Transformer
        self.hidden_dim = 512
        self.pad_token_id = 0
        self.max_position_embeddings = 512
        self.max_length = 160
        self.layer_norm_eps = 1e-12
        self.dropout = 0.1
        self.vocab_size = 30522

        self.enc_layers = 6
        self.dec_layers = 6
        self.dim_feedforward = 2048
        self.nheads = 8
        self.pre_norm = True

        # Dataset
        self.dir = '/media/MyDataStor1/mmrl/MMRL/data/coco'
#         self.dir = '/media/MyDataStor2/zhanghex/coco2017'
        self.data_ver = '2014' 
        self.limit = -1
        
        self.vocab = 'bert-base-uncased'
#         self.vocab = 'catr/damsm_vocab.txt'
