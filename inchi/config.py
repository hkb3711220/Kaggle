from torch.optim import lr_scheduler
# train parameters
seed = 42
num_steps = 8000000
n_epochs = 20
n_fold = 10
batch_size = 32
num_workers = 4
use_mix_precision = True
accumulate_steps = 1
encoder_lr = 1e-4
decoder_lr = 1e-4
weight_decay = 1e-6
lr = 1e-4
# SchedulerClass = lr_scheduler.CosineAnnealingLR
# scheduler_params = dict(T_max=n_epochs)
SchedulerClass = lr_scheduler.StepLR
scheduler_params = dict(step_size=350000,
                        gamma=0.5)
folder = 'effnet_transformers_v3'
step_scheduler = True
validation_scheduler = False
verbose = True
verbose_step = 1
clip_grad = True

# model parameters
encoder_model_name = 'tf_efficientnet_b4_ns' #!TODO: TNT
encoder_model_pretrained = True
max_len = 300
decoder_dim = 512
dim_feedforward = 1024
num_head = 8
num_layer = 6

# transform parameters
size = 456

# others
recreate_tokenizer = False
train_csv_directory = '../input'
STOI = {
            '<sos>': 190,
            '<eos>': 191,
            '<pad>': 192,
        }