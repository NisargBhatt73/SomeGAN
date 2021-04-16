import os
import subprocess
import librosa
import numpy as np
import time


"""
Audio data preprocessing for SEGAN training.

It provides:
	1. 16k downsampling (sox required)
	2. slicing and serializing
	3. verifying serialized data
"""


# specify the paths - modify the paths at your will
DATA_ROOT_DIR = ''  # the base folder for dataset
CLEAN_TRAIN_DIR = 'CleanSpeech_Training'  # where original clean train data exist
NOISY_TRAIN_DIR = 'NoisySpeech_Training'  # where original noisy train data exist
DST_CLEAN_TRAIN_DIR = 'clean_trainset_wav_16k'  # clean preprocessed data folder
DST_NOISY_TRAIN_DIR = 'noisy_trainset_wav_16k'  # noisy preprocessed data folder
SER_DATA_DIR = 'ser_data'  # serialized data folder
SER_DST_PATH = os.path.join(DATA_ROOT_DIR, SER_DATA_DIR)


def verify_data(filename):
	"""
	Verifies the length of each data after preprocessing.
	"""

	data_pair = np.load(os.path.join(SER_DST_PATH, filename))
	if data_pair.shape[1] != 16384:
		print('Snippet length not 16384 : {} instead'.format(data_pair.shape[1]))
		return -1



def resample_16(input_filepath,output_dir,filename):
	dst_clean_dir = os.path.join(DATA_ROOT_DIR, output_dir)
	if not os.path.exists(dst_clean_dir):
		os.makedirs(dst_clean_dir)
	out_filepath = os.path.join(dst_clean_dir, filename)
	print('Downsampling : {}'.format(input_filepath))
	subprocess.run(
		'sox {} -r 16k {}'
		.format(input_filepath, out_filepath),
		shell=True, check=True)
	return out_filepath
def slice_signal(filepath, window_size, stride, sample_rate):
	"""
	Helper function for slicing the audio file
	by window size with [stride] percent overlap (default 50%).
	"""
	wav, sr = librosa.load(filepath, sr=sample_rate)
	n_samples = wav.shape[0]  # contains simple amplitudes
	hop = int(window_size * stride)
	slices = []
	for end_idx in range(window_size, len(wav), hop):
		start_idx = end_idx - window_size
		slice_sig = wav[start_idx:end_idx]
		slices.append(slice_sig)
	return slices



def process_and_serialize(clean_filepath,noisy_filepath,filename):
	"""
	Serialize the sliced signals and save on separate folder.
	"""
	start_time = time.time()  # measure the time
	window_size = 2 ** 14  # about 1 second of samples
	sample_rate = 16000
	stride = 1

	if not os.path.exists(SER_DST_PATH):
		print('Creating new destination folder for new data')
		os.makedirs(SER_DST_PATH)

	# the path for source data (16k downsampled)
	
	clean_sliced = slice_signal(clean_filepath, window_size, stride, sample_rate)
	noisy_sliced = slice_signal(noisy_filepath, window_size, stride, sample_rate)
	# ex) p293_154.wav_5.npy denotes 5th slice of p293_154.wav file
	for idx, slice_tuple in enumerate(zip(clean_sliced, noisy_sliced)):
		pair = np.array([slice_tuple[0], slice_tuple[1]])
		np.save(os.path.join(SER_DST_PATH, '{}_{}'.format(filename, idx)), arr=pair)
	

	# measure the time it took to process
	end_time = time.time()
	print('Total elapsed time for preprocessing : {}'.format(end_time - start_time))

def preprocess(clean_audio_name,noisy_audio_name):
	
	clean_path = resample_16(clean_audio_name,DST_CLEAN_TRAIN_DIR,'cleann.wav')
	noisy_path = resample_16(noisy_audio_name,DST_NOISY_TRAIN_DIR,'noisyy.wav')
	process_and_serialize(clean_path,noisy_path,'audio')
if __name__ == '__main__':
	"""
	Uncomment each function call that suits your needs.
	"""

	preprocess('sp.wav','sp.wav')

	# verify_data()
	print(3)
