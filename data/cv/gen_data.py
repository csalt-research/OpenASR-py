import pandas as pd

root_dir = '.'
train_tsv = 'non_us_train.tsv'
test_tsv = 'non_us_test.tsv'
accent = 'australian'

tr_tsv_file = pd.read_csv(root_dir + '/' + train_tsv, sep='\t', index_col=0)
tr_tsv_file = tr_tsv_file[tr_tsv_file['accent'] == accent]
te_tsv_file = pd.read_csv(root_dir + '/' + test_tsv, sep='\t', index_col=0)
te_tsv_file = te_tsv_file[te_tsv_file['accent'] == accent]

tr_trans = list(tr_tsv_file['Transcripts'])
tr_files = list(tr_tsv_file['file1'])

te_trans = list(te_tsv_file['Transcripts'])
te_files = list(te_tsv_file['file1'])

va_trans = tr_trans[-len(te_trans):]
va_files = tr_files[-len(te_trans):]
tr_trans = tr_trans[:-len(te_trans)]
tr_files = tr_files[:-len(te_trans)]

open(root_dir + '/' + accent + '/src-train.txt', 'w+').writelines([t+'\n' for t in tr_files])
open(root_dir + '/' + accent + '/src-val.txt', 'w+').writelines([t+'\n' for t in va_files])
open(root_dir + '/' + accent + '/src-test.txt', 'w+').writelines([t+'\n' for t in te_files])

open(root_dir + '/' + accent + '/tgt-train.txt', 'w+').writelines([t+'\n' for t in tr_trans])
open(root_dir + '/' + accent + '/tgt-val.txt', 'w+').writelines([t+'\n' for t in va_trans])
open(root_dir + '/' + accent + '/tgt-test.txt', 'w+').writelines([t+'\n' for t in te_trans])
