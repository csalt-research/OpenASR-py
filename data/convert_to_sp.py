import sentencepiece as spm

# parameters
root_dir = 'indian'
suffix = '.filtered'
vocab_size = 512
model_file = None

# train model if model file doesn't exist
if model_file is None:
	spm.SentencePieceTrainer.train('\
		--input={0} \
		--model_prefix={1} \
		--vocab_size={2} \
		--character_coverage=1'
		.format(root_dir + '/tgt-train.txt'+suffix, 
			    root_dir, 
			    str(vocab_size)))

# decode sentences
sp = spm.SentencePieceProcessor()

# load model if exists, else use last trained model	
if model_file is None:
	sp.Load(root_dir + '.model')
else:
	sp.Load(model_file)

# iterate over target files and convert
for split in ['train', 'val', 'test']:
    inp = open(root_dir + '/tgt-' + split + '.txt'+suffix, 'r')
    out = open(root_dir + '/tgt-' + split + '.txt'+suffix+'.converted', 'w')
    lines = [t.strip() for t in inp.readlines()]
    conv_lines = [' '.join(sp.EncodeAsPieces(t)) for t in lines]
    out.writelines([t+'\n' for t in conv_lines])
    inp.close()
    out.close()
