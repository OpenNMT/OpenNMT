
**train.lua**


config
:   Read options from this file []


**Data options**


data
:   Path to the training *-train.t7 file from preprocess.lua []

save_file
:   Savefile name (model will be saved assavefile_epochX_PPL.t7 where X is the X-th epoch and PPL isthe validation perplexity []

train_from
:   If training from a checkpoint then this is the path to the pretrained model. []

continue
:   If training from a checkpoint, whether to continue the training in the same configuration or not. [false]


**Model options**


num_layers
:   Number of layers in the LSTM encoder/decoder [2]

rnn_size
:   Size of LSTM hidden states [500]

word_vec_size
:   Word embedding sizes [500]

feat_vec_exponent
:   If the feature takes N values, then theembedding dimension will be set to N^exponent [0.7]

input_feed
:   Feed the context vector at each time step as additional input (via concatenation with the word embeddings) to the decoder. [true]

brnn
:   Use a bidirectional encoder [false]

brnn_merge
:   Merge action for the bidirectional hidden states: concat or sum [sum]


**Optimization options**


max_batch_size
:   Maximum batch size [64]

epochs
:   Number of training epochs [13]

start_epoch
:   If loading from a checkpoint, the epoch from which to start [1]

start_iteration
:   If loading from a checkpoint, the iteration from which to start [1]

param_init
:   Parameters are initialized over uniform distribution with support (-param_init, param_init) [0.1]

optim
:   Optimization method. Possible options are: sgd, adagrad, adadelta, adam [sgd]

learning_rate
:   Starting learning rate. If adagrad/adadelta/adam is used,then this is the global learning rate. Recommended settings: sgd =1,adagrad = 0.1, adadelta = 1, adam = 0.1 [1]

max_grad_norm
:   If the norm of the gradient vector exceeds this renormalize it to have the norm equal to max_grad_norm [5]

dropout
:   Dropout probability. Dropout is applied between vertical LSTM stacks. [0.3]

lr_decay
:   Decay learning rate by this much if (i) perplexity does not decreaseon the validation set or (ii) epoch has gone past the start_decay_at_limit [0.5]

start_decay_at
:   Start decay after this epoch [9]

curriculum
:   For this many epochs, order the minibatches based on sourcesequence length. Sometimes setting this to 1 will increase convergence speed. [0]

pre_word_vecs_enc
:   If a valid path is specified, then this will loadpretrained word embeddings on the encoder side.See README for specific formatting instructions. []

pre_word_vecs_dec
:   If a valid path is specified, then this will loadpretrained word embeddings on the decoder side.See README for specific formatting instructions. []

fix_word_vecs_enc
:   Fix word embeddings on the encoder side [false]

fix_word_vecs_dec
:   Fix word embeddings on the decoder side [false]


**Other options**


gpuid
:   Which gpu to use (1-indexed). < 1 = use CPU [-1]

nparallel
:   How many parallel process [1]

disable_mem_optimization
:   Disable sharing internal of internal buffers between clones - which is in general safe,except if you want to look inside clones for visualization purpose for instance. [false]

cudnn
:   Whether to use cudnn or not [false]

save_every
:   Save intermediate models every this many iterations within an epoch.If = 0, will not save models within an epoch.  [0]

print_every
:   Print stats every this many iterations within an epoch. [50]

