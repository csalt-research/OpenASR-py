#!/usr/bin/env bash 

model='saved/v3_sp_drop_5_pool_2_bi_enc_512_dec_2/model_step_40000.pt'
out='v3_sp_drop_5_pool_2_bi_enc_512_dec_2'
gpu=0
beam_size=8

# PREPARE PREDICTIONS FILE
# CUDA_VISIBLE_DEVICES=0 python translate.py -data_type audio -model $model -src_dir /hdddata/2scratch/datasets/speech/cv/ -src data/common-voice/indian/src-train.txt -output out/$out.indian.pred_train.txt -gpu $gpu --beam_size $beam_size
CUDA_VISIBLE_DEVICES=1 python translate.py -data_type audio -model $model -src_dir /home/data/MCV-v3/clips -src data/common-voice/us-v3/src-val.txt.filtered -output out/$out.us.pred_val.txt -gpu $gpu --beam_size $beam_size -sample_rate 48000
CUDA_VISIBLE_DEVICES=1 python translate.py -data_type audio -model $model -src_dir /home/data/MCV-v3/clips -src data/common-voice/us-v3/src-test.txt.filtered -output out/$out.us.pred_test.txt -gpu $gpu --beam_size $beam_size -sample_rate 48000

# COMPUTE ERROR RATES
# echo "Train"
# python error_rate.py --pred out/$out.indian.pred_train.txt --tgt data/common-voice/indian/tgt-train.txt.converted --token char
echo "Val"
python error_rate.py --pred out/$out.us.pred_val.txt --tgt data/common-voice/us-v3/tgt-val.txt.filtered.converted --token sp
echo "Test"
python error_rate.py --pred out/$out.us.pred_test.txt --tgt data/common-voice/us-v3/tgt-test.txt.filtered.converted --token sp
