CUDA_VISIBLE_DEVICES=0 python3 preprocess.py --src_train data/cv/us/src-train.txt.filtered data/cv/england/src-train.txt.filtered data/cv/indian/src-train.txt.filtered data/cv/australia/src-train.txt.filtered --tgt_train data/cv/us/tgt-train.txt.filtered.converted data/cv/england/tgt-train.txt.filtered.converted data/cv/indian/tgt-train.txt.filtered.converted data/cv/australia/tgt-train.txt.filtered.converted --src_valid data/cv/us/src-val.txt.filtered data/cv/england/src-val.txt.filtered data/cv/indian/src-val.txt.filtered data/cv/australia/src-val.txt.filtered --tgt_valid data/cv/us/tgt-val.txt.filtered.converted data/cv/england/tgt-val.txt.filtered.converted data/cv/indian/tgt-val.txt.filtered.converted data/cv/australia/tgt-val.txt.filtered.converted --src_test data/cv/us/src-test.txt.filtered data/cv/england/src-test.txt.filtered data/cv/indian/src-test.txt.filtered data/cv/australia/src-test.txt.filtered --tgt_test data/cv/us/tgt-test.txt.filtered.converted data/cv/england/tgt-test.txt.filtered.converted data/cv/indian/tgt-test.txt.filtered.converted data/cv/australia/tgt-test.txt.filtered.converted --src_dir /home/data/MCV-v3/clips/ --save_dir shards/cv3_stft --shard_size 6000 --vocab data/cv/us/tgt-train.txt.filtered.converted --max_vocab_size 0 --sample_rate 22050 --window_size 0.02 --window_stride 0.01 --window hamming --feat_type stft --normalize_audio

# CUDA_VISIBLE_DEVICES=0 python3 train.py --share_dec_weights --brnn --data shards/cv3 --param_init_glorot --train_steps 50000 --valid_steps 3000 --optim adam --learning_rate 0.001 --save_dir saved/test --bridge_type mlp

# CUDA_VISIBLE_DEVICES=0 python3 train.py --share_dec_weights --brnn --data shards/cv3/us --param_init_glorot --train_steps 40000 --valid_steps 3500 --optim adam --learning_rate 0.001 --save_dir saved/us_mlp_adam --bridge_type mlp --batch_size 32 --train_from saved/us_mlp_adam/checkpoint.38501.pt
