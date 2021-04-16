"""
Here we define the discriminator and generator for SEGAN.
After definition of each modules, run the training.
"""

import time
import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch import optim
import torch.nn.functional as F
import numpy as np
from scipy.io import wavfile
from data_generator import AudioSampleGenerator
from vbnorm import VirtualBatchNorm1d
from tensorboardX import SummaryWriter
import soundfile as sf


# device we're using
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# define folders for output data
in_path = ''
out_path_root = 'segan_data_out'
ser_data_fdr = 'ser_data'  # serialized data
gen_data_fdr = 'gen_data'  # folder for saving generated data
checkpoint_fdr = 'checkpoint'  # folder for saving models, optimizer states, etc.
tblog_fdr = 'logs'  # summary data for tensorboard
# time info is used to distinguish dfferent training sessions
run_time = time.strftime('%Y%m%d_%H%M', time.gmtime())  # 20180625_1742
# output path - all outputs (generated data, logs, model checkpoints) will be stored here
# the directory structure is as: "[curr_dir]/segan_data_out/[run_time]/"
out_path = os.path.join(os.getcwd(), out_path_root, run_time)
tblog_path = os.path.join(os.getcwd(), tblog_fdr, run_time)  # summary data for tensorboard
# SOME TRAINING PARAMETERS #
batch_size = 32
d_learning_rate = 0.0001
g_learning_rate = 0.0001
g_lambda = 100  # regularizer for generator
use_devices = [0, 1, 2, 3]
sample_rate = 16000
num_gen_examples = 32  # number of generated audio examples displayed per epoch    SHOULD BE EQUAL TO BATCH SIZE
total_examples_per_epoch = 5
num_epochs = 86
epoch = 26  ## EPOCH TO TEST

class PhaseShuffle(nn.Module):
	"""
	Performs phase shuffling, i.e. shifting feature axis of a 3D tensor
	by a random integer in {-n, n} and performing reflection padding where
	necessary.
	"""
	# Copied from https://github.com/jtcramer/wavegan/blob/master/wavegan.py#L8
	def __init__(self, shift_factor):
		super(PhaseShuffle, self).__init__()
		self.shift_factor = shift_factor

	def forward(self, x):
		if self.shift_factor == 0:
			return x
		# uniform in (L, R)
		k_list = torch.Tensor(x.shape[0]).random_(0, 2 * self.shift_factor + 1) - self.shift_factor
		k_list = k_list.numpy().astype(int)

		# Combine sample indices into lists so that less shuffle operations
		# need to be performed
		k_map = {}
		for idx, k in enumerate(k_list):
			k = int(k)
			if k not in k_map:
				k_map[k] = []
			k_map[k].append(idx)

		# Make a copy of x for our output
		x_shuffle = x.clone()

		# Apply shuffle to each sample
		for k, idxs in k_map.items():
			if k > 0:
				x_shuffle[idxs] = F.pad(x[idxs][..., :-k], (k, 0), mode='reflect')
			else:
				x_shuffle[idxs] = F.pad(x[idxs][..., -k:], (0, -k), mode='reflect')

		assert x_shuffle.shape == x.shape, "{}, {}".format(x_shuffle.shape,
													   x.shape)
		return x_shuffle

class SelfAttention(nn.Module):
	"""
		Self attention module
	"""
	# Copied from https://github.com/jtcramer/wavegan/blob/master/wavegan.py#L8

	def __init__(self,in_dim,activation):
		super(SelfAttention,self).__init__()
		self.chanel_in = in_dim
		self.activation = activation
		
		self.query_conv = nn.Conv1d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
		self.key_conv = nn.Conv1d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
		self.value_conv = nn.Conv1d(in_channels = in_dim , out_channels = in_dim , kernel_size= 1)
		self.gamma = nn.Parameter(torch.zeros(1))

		self.softmax  = nn.Softmax(dim=-1) #
	def forward(self,x):
		"""
			inputs :
				x : input feature maps( B X C X S)
			returns :
				out : self attention value + input feature 
				attention: B X N X N (N is S)
		"""
		m_batchsize,C,S = x.size()
		proj_query  = self.query_conv(x).view(m_batchsize,-1,S).permute(0,2,1) # B X CX(N)
		proj_key =  self.key_conv(x).view(m_batchsize,-1,S) # B X C x (S)
		energy =  torch.bmm(proj_query,proj_key) # transpose check
		attention = self.softmax(energy) # BX (N) X (N) 
		proj_value = self.value_conv(x).view(m_batchsize,-1,S) # B X C X N

		out = torch.bmm(proj_value,attention.permute(0,2,1) )
		out = out.view(m_batchsize,C,S)
		
		out = self.gamma*out + x
		
		return out,attention

class Generator(nn.Module):
	def __init__(self, d = 32, c = 1):
		super().__init__()
		
		
		## Input B*16384*c

		self.conv1 = nn.Conv1d(in_channels = c, out_channels=d,kernel_size = 25, stride = 4, padding = 11) #B * 4096 * d
		self.enc_prelu1 = nn.PReLU()
		self.conv2 = nn.Conv1d( d, 2*d, 25, 4, 11) #B * 1024 * 2d
		self.enc_prelu2 = nn.PReLU()
		self.self_att_1 = SelfAttention(2*d,'relu')

		self.conv3 = nn.Conv1d( 2*d, 4*d, 25, 4, 11) #B * 256 * 4d
		self.enc_prelu3 = nn.PReLU()
		self.conv4 = nn.Conv1d( 4*d, 8*d, 25, 4, 11) #B * 64 * 8d
		self.enc_prelu4 = nn.PReLU()
		self.conv5 = nn.Conv1d( 8*d, 16*d, 25, 4, 11) #B * 16 * 16d
		self.enc_prelu5 = nn.PReLU()
		self.self_att_2 = SelfAttention(16*d,'relu')

		#Encoded thought vector  B*16*16d

		#decoder
		self.tconv1 = nn.ConvTranspose1d(in_channels=16*d, out_channels=8*d, kernel_size=25, stride=4, padding=11,output_padding = 1) #B * 64 * 8d
		self.dec_prelu1 = nn.PReLU()
		self.tconv2 = nn.ConvTranspose1d(16*d, 4*d, 25, 4, 11, 1) #B * 256 * 4d
		self.dec_prelu2 = nn.PReLU()
		self.tconv3 = nn.ConvTranspose1d(8*d, 2*d, 25, 4, 11, 1) # B * 1024 * 2d
		self.dec_prelu3 = nn.PReLU()
		self.tconv4 = nn.ConvTranspose1d(4*d, d, 25, 4, 11, 1) # B * 4096 * d
		self.dec_prelu4 = nn.PReLU()
		self.tconv_final = nn.ConvTranspose1d(2*d, c, 25, 4, 11, 1) # B * 16384 * c
		self.dec_tanh = nn.Tanh()

		self.init_weights() 

	def init_weights(self):
		"""
			Initialize weights for convolution layers using Xavier initialization.
		"""
		for m in self.modules():
			if isinstance(m, nn.Conv1d) or isinstance(m, nn.ConvTranspose1d):
				nn.init.xavier_normal_(m.weight.data)

	def forward(self,x):
		"""
		Forward pass of generator.

		Args:
		x: input batch (signal)

		"""
		##Encode

		e1 = self.conv1(x)#B * 4096 * d
		e2 = self.conv2(self.enc_prelu1(e1)) #B * 1024 * 2d
		e2, a1 = self.self_att_1(e2)
		e3 = self.conv3(self.enc_prelu2(e2)) #B * 256 * 4d
		e4 = self.conv4(self.enc_prelu3(e3)) #B * 64 * 8d
		e5 = self.conv5(self.enc_prelu4(e4)) #B * 16 * 16d
		#print("{}  {}  {}  {}  {}".format(e1.size(),e2.size(),e3.size(),e4.size(),e5.size()))
		encoded = self.enc_prelu5(e5)
		att_encoded, a2= self.self_att_2(encoded)
		#print(encoded.size())

		# B * 16 * 16d
		### decoding step
		d4 = self.tconv1(att_encoded) #B * 64 * 8d
		#print(d4.size())
		# dx_c : concatenated with skip-connected layer's output & passed nonlinear layer
		d4_c = self.dec_prelu1(torch.cat((d4, e4), dim=1))  #B * 64 * 16d
		d3 = self.tconv2(d4_c)                              #B * 256 * 4d
		d3_c = self.dec_prelu2(torch.cat((d3, e3), dim=1))  #B * 256 * 8d
		d2 = self.tconv3(d3_c)                              #B * 1024 * 2d
		d2_c = self.dec_prelu3(torch.cat((d2, e2), dim=1))  #B * 1024 * 4d
		d1 = self.tconv4(d2_c)                              #B * 4096 * d
		d1_c = self.dec_prelu4(torch.cat((d1, e1), dim=1))  #B * 4096 * 2d
		decoded = self.dec_tanh(self.tconv_final(d1_c))     #B * 16384 * c
 
		return decoded

class Discriminator(nn.Module):
	def init_weights(self):
		"""
		Initialize weights for convolution layers using Xavier initialization.
		"""
		for m in self.modules():
			if isinstance(m, nn.Conv1d):
				nn.init.xavier_normal_(m.weight.data)
	def __init__(self,d = 32, c = 1, dropout_drop = 0.5,shift_factor = 2):
		super().__init__()
		negative_slope = 0.03



		# in : (B * 16384 * c) * 2
		self.conv1 = nn.Conv1d(in_channels=2*c, out_channels=d, kernel_size=25, stride=4, padding=11)   # out : 8192 x 32
		self.vbn1 = VirtualBatchNorm1d(d)
		self.lrelu1 = nn.LeakyReLU(negative_slope)
		self.dropout1 = nn.Dropout(dropout_drop)

		self.conv2 = nn.Conv1d( d, 2*d, 25, 4, 11) #B * 1024 * 2d
		self.vbn2 = VirtualBatchNorm1d(2*d)
		self.lrelu2 = nn.LeakyReLU(negative_slope)

		self.conv3 = nn.Conv1d( 2*d, 4*d, 25, 4, 11) #B * 256 * 4d
		self.dropout2 = nn.Dropout(dropout_drop)
		self.vbn3 = VirtualBatchNorm1d(4*d)
		self.lrelu3 = nn.LeakyReLU(negative_slope)

		self.conv4 = nn.Conv1d( 4*d, 8*d, 25, 4, 11) #B * 64 * 8d
		self.vbn4 = VirtualBatchNorm1d(8*d)
		self.lrelu4 = nn.LeakyReLU(negative_slope)
		self.dropout3 = nn.Dropout(dropout_drop)

		self.conv5 = nn.Conv1d( 8*d, 16*d, 25, 4, 11) #B * 16 * 16d
		self.vbn5 = VirtualBatchNorm1d(16*d)
		self.lrelu5 = nn.LeakyReLU(negative_slope)

		self.conv_final = nn.Conv1d(16*d, 1, kernel_size=1, stride=1) # B * 16 * 1
		self.lrelu_final = nn.LeakyReLU(negative_slope)
		self.fully_connected = nn.Linear(in_features=16, out_features=1)  # B * 1
		self.sigmoid = nn.Sigmoid()

		self.ps1 = PhaseShuffle(shift_factor)
		self.ps2 = PhaseShuffle(shift_factor)
		self.ps3 = PhaseShuffle(shift_factor)
		self.ps4 = PhaseShuffle(shift_factor)
		self.ps5 = PhaseShuffle(shift_factor)

		#initialize weights
		self.init_weights()

	def forward(self, x, ref_x):
			"""
			Forward pass of discriminator.

			Args:
				x: batch
				ref_x: reference batch for virtual batch norm
			"""
			# reference pass
			ref_x = self.conv1(ref_x)
			ref_x = self.dropout1(ref_x)
			#ref_x, mean1, meansq1 = self.vbn1(ref_x, None, None)
			ref_x = self.lrelu1(ref_x)
			ref_x = self.ps1(ref_x)

			ref_x = self.conv2(ref_x)
			#ref_x, mean2, meansq2 = self.vbn2(ref_x, None, None)
			ref_x = self.lrelu2(ref_x)
			ref_x = self.ps2(ref_x)

			ref_x = self.conv3(ref_x)
			ref_x = self.dropout2(ref_x)
			#ref_x, mean3, meansq3 = self.vbn3(ref_x, None, None)
			ref_x = self.lrelu3(ref_x)
			ref_x = self.ps3(ref_x)

			ref_x = self.conv4(ref_x)
			ref_x = self.dropout3(ref_x)
			#ref_x, mean4, meansq4 = self.vbn4(ref_x, None, None)
			ref_x = self.lrelu4(ref_x)
			ref_x = self.ps4(ref_x)

			ref_x = self.conv5(ref_x)
			#ref_x, mean5, meansq5 = self.vbn5(ref_x, None, None)
			ref_x = self.lrelu5(ref_x)
			ref_x = self.ps5(ref_x)
			
			# further pass no longer needed

			# train pass
			x = self.conv1(x)
			x = self.dropout1(x)
			#x, _, _ = self.vbn1(x, mean1, meansq1)
			x = self.lrelu1(x)
			x = self.ps1(x)

			x = self.conv2(x)
			#x, _, _ = self.vbn2(x, mean2, meansq2)
			x = self.lrelu2(x)
			x = self.ps2(x)

			x = self.conv3(x)
			x = self.dropout2(x)
			#x, _, _ = self.vbn3(x, mean3, meansq3)
			x = self.lrelu3(x)
			x = self.ps3(x)

			x = self.conv4(x)
			x = self.dropout3(x)
			#x, _, _ = self.vbn4(x, mean4, meansq4)
			x = self.lrelu4(x)
			x = self.ps4(x)

			x = self.conv5(x)
			#x, _, _ = self.vbn5(x, mean5, meansq5)
			x = self.lrelu5(x)
			x = self.ps5(x)
			
			x = self.conv_final(x)
			x = self.lrelu_final(x)
			# reduce down to a scalar value
			x = torch.squeeze(x)
			x = self.fully_connected(x)
			# return self.sigmoid(x)
			return x

def split_pair_to_vars(sample_batch_pair):
	"""
	Splits the generated batch data and creates combination of pairs.
	Input argument sample_batch_pair consists of a batch_size number of
	[clean_signal, noisy_signal] pairs.

	This function creates three pytorch Variables - a clean_signal, noisy_signal pair,
	clean signal only, and noisy signal only.
	It goes through preemphasis preprocessing before converted into variable.

	Args:
		sample_batch_pair(torch.Tensor): batch of [clean_signal, noisy_signal] pairs
	Returns:
		batch_pairs_var(Variable): batch of pairs containing clean signal and noisy signal
		clean_batch_var(Variable): clean signal batch
		noisy_batch_var(Varialbe): noisy signal batch
	"""
	# pre-emphasis
	sample_batch_pair = sample_batch_pair.numpy()

	batch_pairs_var = torch.from_numpy(sample_batch_pair).type(torch.FloatTensor).to(device)  # [40 x 2 x 16384]
	clean_batch = np.stack([pair[0].reshape(1, -1) for pair in sample_batch_pair])
	clean_batch_var = torch.from_numpy(clean_batch).type(torch.FloatTensor).to(device)
	noisy_batch = np.stack([pair[1].reshape(1, -1) for pair in sample_batch_pair])
	noisy_batch_var = torch.from_numpy(noisy_batch).type(torch.FloatTensor).to(device)
	#print(" I came to create batch and this is output B size : {}".format(len(clean_batch_var)))
	return batch_pairs_var, clean_batch_var, noisy_batch_var

### INITIALIZING MODEL
discriminator = torch.nn.DataParallel(Discriminator().to(device), device_ids=(0,))
#print(discriminator)
print('Discriminator created')
generator = torch.nn.DataParallel(Generator().to(device), device_ids=(0,))

#print(generator)
print('Generator created')
sample_generator = AudioSampleGenerator(os.path.join(in_path, ser_data_fdr))

print('DataLoader created')
states = torch.load('segan_data_out/20201121_1316/checkpoint/state-'+str(epoch)+'.pkl')
discriminator.load_state_dict(states['discriminator'])
generator.load_state_dict(states['generator'])



# Function that gives inference on entire ser_data folder
def infer():
	extra,clean_data,noisy_data = sample_generator.all_audio(batch_size)
	no_of_batches = clean_data.shape[0]
	print(no_of_batches)
	print("***********")
	# print(clean_data.shape)
	print(extra)
	#print(y.shape)
	ftc = clean_data
	ftn = noisy_data
	clean_data = torch.from_numpy(clean_data)
	noisy_data = torch.from_numpy(noisy_data)
	
	print('Ho gaya mera')

	# test_noise_filenames, fixed_test_clean, fixed_test_noise = \
	# 	sample_generator.fixed_test_audio(num_gen_examples)
	# print(fixed_test_noise.shape)

	# ftc = fixed_test_clean
	# ftn = fixed_test_noise
	# fixed_test_clean = torch.from_numpy(fixed_test_clean)
	# fixed_test_noise = torch.from_numpy(fixed_test_noise)
	# y = torch.from_numpy(y)
	path = os.path.join(os.getcwd(),'Inference/'+str(epoch)+'/')
	if not os.path.exists(path):
		os.makedirs(path)
	generated_sample = np.empty(0)
	original_sample = np.empty(0)
	noisy_sample = np.empty(0)
	for batch in range(no_of_batches):

		fake_speech = generator(noisy_data[batch])
		print(fake_speech)
		fake_speech_data = fake_speech.data.cpu().numpy()
		cnt = 0
		
		for idx in range(num_gen_examples):
			if batch + 1 == no_of_batches and idx+1 >num_gen_examples-extra:
				break
			#print(fake_speech_data[idx].shape)
			generated_sample = np.append(generated_sample,fake_speech_data[idx])
			original_sample = np.append(original_sample,ftc[batch][idx])
			noisy_sample = np.append(noisy_sample,ftn[batch][idx])
			#gen_fname = test_noise_filenames[idx]
			# 12 23 34 45 56 6
		#print(len(generated_sample))
		# write to file
	sf.write(str(path)+'qgenerate-'+str(batch)+'--d.wav',generated_sample.T,sample_rate)
	sf.write(str(path)+'qnois-'+str(batch)+'--y.wav',  noisy_sample.T,sample_rate)
	sf.write(str(path)+'qorigina-'+str(batch)+'--l.wav',  original_sample.T,sample_rate)
		# if cnt > total_examples_per_epoch:
			# 	break
			# cnt = cnt + 1	

	

if __name__ == '__main__':
	infer()