def build_preprocess_parser(parser):
    preprocess_opts(parser)
    return parser

def build_train_parser(parser):
    model_opts(parser)
    general_opts(parser)
    train_opts(parser)
    translate_opts(parser)
    return parser

def build_test_parser(parser):
    general_opts(parser)
    translate_opts(parser)
    return parser

def model_opts(parser):
    # Embedding
    group = parser.add_argument_group('Model - Embeddings')
    group.add('--embedding_size', type=int, default=256,
              help='Token embedding size for target')
    group.add('--share_dec_weights', action='store_true',
              help="Use a shared weight matrix for the input and "
                   "output word embeddings in the decoder.")

    # Embedding features
    group = parser.add_argument_group('Model - Embedding Features')
    group.add('--feat_merge', '-feat_merge', type=str, default='concat',
              choices=['concat', 'sum', 'mlp'],
              help="Merge action for incorporating features embeddings. "
                   "Options [concat|sum|mlp].")
    group.add('--feat_vec_size', '-feat_vec_size', type=int, default=-1,
              help="If specified, feature embedding sizes "
                   "will be set to this. Otherwise, feat_vec_exponent "
                   "will be used.")
    group.add('--feat_vec_exponent', '-feat_vec_exponent',
              type=float, default=0.7,
              help="If -feat_merge_size is not set, feature "
                   "embedding sizes will be set to N^feat_vec_exponent "
                   "where N is the number of values the feature takes.")

    # Encoder
    group = parser.add_argument_group('Model - Encoder')
    group.add('--enc_rnn_type', type=str, default='LSTM', choices=['LSTM', 'GRU'],
              help="Type of encoder RNN layer to use.")
    group.add('--enc_layers', type=int, default=3,
              help='Number of layers in the encoder')
    group.add('--enc_rnn_size', type=int, default=512,
              help="Size of encoder rnn hidden states.")
    group.add('--brnn', action='store_true',
              help="Whether to use bidirectional encoder.")
    group.add('--enc_pooling', type=str, default='2',
              help="The amount of pooling of audio encoder, "
                   "either the same amount of pooling across all layers "
                   "indicated by a single number, or different amounts of "
                   "pooling per layer separated by comma.")
    group.add('--enc_dropout', type=float, default=0.0,
              help="Dropout probability for encoder.")

    # Decoder
    group = parser.add_argument_group('Model - Decoder')
    group.add('--dec_rnn_type', type=str, default='LSTM', choices=['LSTM', 'GRU'],
              help="Type of decoder RNN layer to use.")
    group.add('--dec_layers', type=int, default=2,
              help='Number of layers in the decoder')
    group.add('--dec_rnn_size', type=int, default=256,
              help="Size of decoder rnn hidden states.")
    group.add('--dec_dropout', type=float, default=0.0,
              help="Dropout probability for decoder.")
    group.add('--init_sched_sampling_rate', type=float, default=0.0,
              help="Initial rate for scheduled sampling")

    # Attention
    group = parser.add_argument_group('Model - Attention')
    group.add('--attention_type', type=str, default='general',
              choices=['dot', 'general', 'mlp'],
              help="The attention type to use: "
                   "dotprod or general (Luong) or MLP (Bahdanau)")

    # Bridge
    group = parser.add_argument_group('Model - Bridge')
    group.add('--bridge_type', type=str, default='zero',
              choices=['copy', 'mlp', 'zero'],
              help="The bridge type to use between encoder and decoder.")

def preprocess_opts(parser):
    # Data
    group = parser.add_argument_group('Data')
    group.add('--src_train', required=True, nargs='+',
              help="Path(s) to the training source data")
    group.add('--tgt_train', required=True, nargs='+',
              help="Path(s) to the training target data")
    group.add('--src_valid', required=True, nargs='+',
              help="Path(s) to the validation source data")
    group.add('--tgt_valid', required=True, nargs='+',
              help="Path(s) to the validation target data")
    group.add('--src_dir', default="",
              help="Source directory for audio files.")
    group.add('--save_dir', required=True,
              help="Directory for saving the prepared data")
    group.add('--shard_size', type=int, default=6000,
              help="Divide src_corpus and tgt_corpus into "
                   "smaller multiple src_copus and tgt corpus files, then "
                   "build shards, each shard will have "
                   "opt.shard_size samples except last shard. "
                   "shard_size=0 means no segmentation "
                   "shard_size>0 means segment dataset into multiple shards, "
                   "each shard has shard_size samples")

    # Vocab
    group = parser.add_argument_group('Vocab')
    group.add('--vocab', type=str, required=True,
              help="File to be used for building vocabulary.")
    group.add('--max_vocab_size', type=int, default=50000,
              help="Maximum size of the vocabulary")

    # Audio processing
    group = parser.add_argument_group('Audio')
    group.add('--sample_rate', type=int, default=16000,
              help="Sample rate.")
    group.add('--window_size', type=float, default=.02,
              help="Window size for spectrogram in seconds.")
    group.add('--window_stride', type=float, default=.01,
              help="Window stride for spectrogram in seconds.")
    group.add('--window', default='hamming',
              help="Window type for spectrogram generation. "
                   "Passed to librosa as argument.")
    group.add('--feat_type', default='mfcc', choices=['fbank', 'stft', 'mfcc'],
              help="Type of audio features to be extracted")
    group.add('--normalize_audio', action='store_true',
              help="Whether to perform mean-variance normalization on features.")

def general_opts(parser):
    group = parser.add_argument_group('General')
    group.add('--data', type=str, required=True,
              help='Path prefix to .pt files generated by preprocess.py')
    group.add('--checkpoint', type=str, default='',
              help='Path to checkpoint of pretrained model')
    group.add('--seed', type=int, default=1234,
          help="Random seed used for the experiments reproducibility.")

def train_opts(parser):
    # Initialization
    group = parser.add_argument_group('Initialization')
    group.add('--param_init', type=float, default=0.1,
              help="Init parameters with uniform distribution "
                   "with support (-param_init, param_init). "
                   "Use 0 to not use initialization")
    group.add('--param_init_glorot', action='store_true',
              help="Init parameters with xavier_uniform.")

    # Optimization
    group = parser.add_argument_group('Optimization')
    group.add('--train_batch_size', type=int, default=32,
              help='Batch size for training')
    group.add('--bucket_size', type=int, default=256,
              help="Shuffle this many examples to reduce padding.")
    group.add('--bptt', type=int, default=0,
              help="Number of timesteps for truncated BPTT. Set to 0 to disable.")

    group.add('--train_steps', type=int, default=100000,
              help='Number of training steps')
    group.add('--eval_steps', type=int, default=10000,
              help='Perfom validation every X steps')
    group.add('--shard_size', type=int, default=0,
              help="Maximum batches of words in a sequence to run "
                   "the generator on in parallel. Higher is faster, but "
                   "uses more memory. Set to 0 to disable.")
    group.add('--single_pass', action='store_true',
              help="Make a single pass over the training dataset.")

    group.add('--optim', default='sgd',
              choices=['sgd', 'adagrad', 'adadelta', 'adam'],
              help="Optimization method.")
    group.add('--adagrad_accumulator_init', type=float, default=0,
              help="Initializes the accumulator values in adagrad. "
                   "Mirrors the initial_accumulator_value option "
                   "in the tensorflow adagrad (use 0.1 for their default).")
    group.add('--max_grad_norm', type=float, default=10,
              help="If the norm of the gradient vector exceeds this, "
                   "renormalize it to have the norm equal to "
                   "max_grad_norm")
    group.add('--adam_beta1', type=float, default=0.9,
              help="The beta1 parameter used by Adam. "
                   "Almost without exception a value of 0.9 is used in "
                   "the literature, seemingly giving good results, "
                   "so we would discourage changing this value from "
                   "the default without due consideration.")
    group.add('--adam_beta2', type=float, default=0.999,
              help='The beta2 parameter used by Adam. '
                   'Typically a value of 0.999 is recommended, as this is '
                   'the value suggested by the original paper describing '
                   'Adam, and is also the value adopted in other frameworks '
                   'such as Tensorflow and Kerras, i.e. see: '
                   'https://www.tensorflow.org/api_docs/python/tf/train/Adam'
                   'Optimizer or '
                   'https://keras.io/optimizers/ . '
                   'Whereas recently the paper "Attention is All You Need" '
                   'suggested a value of 0.98 for beta2, this parameter may '
                   'not work well for normal models / default '
                   'baselines.')

    group.add('--learning_rate', type=float, default=1.0,
              help="Starting learning rate. "
                   "Recommended settings: sgd = 1, adagrad = 0.1, "
                   "adadelta = 1, adam = 0.001")
    group.add('--learning_rate_decay', type=float, default=0.5,
              help="If update_learning_rate, decay learning rate by "
                   "this much if steps have gone past "
                   "start_decay_steps")
    group.add('--start_decay_steps', type=int, default=50000,
              help="Start decaying every decay_steps after start_decay_steps")
    group.add('--decay_steps', type=int, default=10000,
              help="Decay every decay_steps")
    group.add('--decay_method', type=str, default="none",
              choices=['noam', 'noamwd', 'rsqrt', 'none'],
              help="Use a custom decay rate.")
    group.add('--warmup_steps', type=int, default=4000,
              help="Number of warmup steps for custom decay.")

    group = parser.add_argument_group('Logging')
    group.add('--log_every', type=int, default=50,
              help="Print stats at this interval.")
    group.add("--tensorboard_dir", type=str, default="",
              help="Log directory for Tensorboard. "
                   "This is also the name of the run.")
    group.add("--save_dir", type=str, default="saved",
              help="Directory for saving checkpoints.")

def translate_opts(parser):
    group = parser.add_argument_group('Translation')
    group.add('--eval_batch_size', type=int, default=32,
              help='Batch size for evaluation')
    group.add('--eval_split', choices=['train', 'valid', 'test'], default='test',
              help='Split to be used for evaluation')
    group.add('--n_best', type=int, default=1,
              help='Number of hypotheses to return for each sample')
    group.add('--min_length', type=int, default=0,
              help='Minimum length of generated transcription')
    group.add('--max_length', type=int, default=100,
              help='Maximum length of generated transcription')
    group.add('--ratio', type=float, default=0.,
              help='If greater than 0, used for estimating transcription '
                    'length from length of encoded sequence')
    group.add('--beam_size', type=int, default=8,
              help='Size of beam during search')
    group.add('--block_ngram_repeat', type=int, default=0,
              help='Block hypotheses containing as many consecutive '
                   'repetitions of ngrams/tokens')
    group.add('--excluded_toks', type=str, default='',
              help='Comma-separated list of tokens not to be '
                    'blocked during decoding')
    group.add('--out', type=str, default='',
              help='File for writing generated hypotheses')
    group.add('--verbose', action='store_true',
              help='Print the best transcription as it is generated')
    group.add('--attn_debug', action='store_true',
              help='Print the attention heatmap for each sample')
