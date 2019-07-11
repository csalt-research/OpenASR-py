#!/usr/bin/env bash 

model='saved/default_char_drop_5_pool_2/model_step_8000.pt'
out='default_char_drop_5_pool_2'
gpu=0
beam_size=8

# PREPARE PREDICTIONS FILE
# CUDA_VISIBLE_DEVICES=0 python translate.py -data_type audio -model $model -src_dir /hdddata/2scratch/datasets/speech/cv/ -src data/common-voice/indian/src-train.txt -output out/$out.indian.pred_train.txt -gpu $gpu --beam_size $beam_size
# CUDA_VISIBLE_DEVICES=0 python translate.py -data_type audio -model $model -src_dir /hdddata/2scratch/datasets/speech/cv/ -src data/common-voice/indian/src-val.txt -output out/$out.indian.pred_val.txt -gpu $gpu --beam_size $beam_size
# CUDA_VISIBLE_DEVICES=0 python translate.py -data_type audio -model $model -src_dir /hdddata/2scratch/datasets/speech/cv/ -src data/common-voice/indian/src-test.txt -output out/$out.indian.pred_test.txt -gpu $gpu --beam_size $beam_size

# COMPUTE ERROR RATES
echo "Train"
python error_rate.py --pred out/$out.indian.pred_train.txt --tgt data/common-voice/indian/tgt-train.txt.converted --token char
# echo "Val"
# python error_rate.py --pred out/$out.indian.pred_val.txt --tgt data/common-voice/indian/tgt-val.txt.converted --token char
# echo "Test"
# python error_rate.py --pred out/$out.indian.pred_test.txt --tgt data/common-voice/indian/tgt-test.txt.converted --token char