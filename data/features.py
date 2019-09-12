import librosa
import math

class AudioFeatureExtractor(object):
	def __init__(self,
				 sample_rate=16e3, 
				 window_size=0.02, 
				 window_stride=0.01,
				 window='hamming',
				 feat_type='mfcc', 
				 normalize_audio=True
				 ):
		self.sample_rate = sample_rate
		self.window_size = window_size
		self.window_stride = window_stride
		self.window = window
		self.normalize_audio = normalize_audio
		self.feat_type = feat_type

		self.win_length = math.ceil(self.sample_rate * self.window_size)
		self.hop_length = math.ceil(self.sample_rate * self.window_stride)

		if feat_type == 'mfcc':
			self.feat_dim = 40
		elif feat_type == 'stft':
			self.feat_dim = int(math.floor((sample_rate * window_size) / 2) + 1)
		else:
			pass

	def __call__(self, audio_path):
		sample, sr = librosa.load(audio_path)
		assert sr == self.sample_rate, \
			"sample rate mismatch (%d vs %d)" % (self.sample_rate, sr)

		if self.feat_type == 'mfcc':
			spect = librosa.feature.mfcc(
				y=sample,
				sr=sr,
				n_mfcc=40,
				n_fft=self.win_length,
				hop_length=self.hop_length,
				window=self.window)
		elif self.feat_type == 'stft':
			d = librosa.stft(
				y=sample,
				n_fft=self.win_length, 
				hop_length=self.hop_length,
				window=self.window)
			spect, _ = librosa.magphase(d)
			spect = np.log1p(spect)
		elif self.feat_type == 'fbank':
			pass
			
		if self.normalize_audio:
			mean = spect.mean()
			std = spect.std()
			spect -= mean
			spect /= std

		return spect.T