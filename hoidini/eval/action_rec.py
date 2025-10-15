import torch
import numpy as np
import torch.nn as nn
import os
import sys
sys.path.append('.')
sys.path.append('..')

import torch.nn.functional as F
import torch.autograd as autograd

class LSTM_Action_Classifier(nn.Module):
	def __init__(self, joints_dim=44, hidden_dim=128, label_size=29, batch_size=1, num_layers=2, kernel_size=3):   #LSTMClassifier(48, 128, 8, 1, 2, 3)
		super(LSTM_Action_Classifier, self).__init__()
		self.hidden_dim = hidden_dim
		self.batch_size = batch_size
		self.num_layers = num_layers
		joints_dim2d = joints_dim 
		
		self.lstm2_2 = nn.LSTM(joints_dim2d, hidden_dim, num_layers=self.num_layers)
		self.conv1_2_2 = nn.Conv1d(1, 1, kernel_size, stride=1, padding=1)
		self.lstm2_3 = nn.LSTM(joints_dim2d, hidden_dim, num_layers=self.num_layers)
		self.conv1_2_3 = nn.Conv1d(1, 1, kernel_size, stride=1, padding=1)
		self.sig = nn.Sigmoid()
		self.hidden2_2 = self.init_hidden2_2()
		self.hidden2_3 = self.init_hidden2_3()
		
		self.hidden2label = nn.Linear(hidden_dim, label_size)
	
	def init_hidden2_1(self):
		# the first is the hidden h
		# the second is the cell  c
		return (autograd.Variable(torch.zeros(self.num_layers, self.batch_size, self.hidden_dim).cuda()),
				autograd.Variable(torch.zeros(self.num_layers, self.batch_size, self.hidden_dim).cuda()))
	def init_hidden2_2(self):
		# the first is the hidden h
		# the second is the cell  c
		return (autograd.Variable(torch.zeros(self.num_layers, self.batch_size, self.hidden_dim).cuda()),
				autograd.Variable(torch.zeros(self.num_layers, self.batch_size, self.hidden_dim).cuda()))
	def init_hidden2_3(self):
		# the first is the hidden h
		# the second is the cell  c
		return (autograd.Variable(torch.zeros(self.num_layers, self.batch_size, self.hidden_dim).cuda()),
				autograd.Variable(torch.zeros(self.num_layers, self.batch_size, self.hidden_dim).cuda()))
	
	def predict(self, joints3d_vec):
		x3 = joints3d_vec
		x2 = x3.view(-1, 22, 3)
		x2_2 = x2[:,:,0:2].contiguous().view(-1, 1, 44)
		x2_3 = x2[:,:,[0,2]].contiguous().view(-1, 1, 44)
		lstm_out2_2, self.hidden2_2_ = self.lstm2_2(x2_2, self.hidden2_2)
		lstm_out2_3, self.hidden2_3_ = self.lstm2_3(x2_3, self.hidden2_3)
		t2_2 = lstm_out2_2[-1].view(self.batch_size,1,-1)
		t2_3 = lstm_out2_3[-1].view(self.batch_size,1,-1)
		y2_2 = self.conv1_2_2(t2_2)
		y2_3 = self.conv1_2_3(t2_3)
		y3 = y2_2+y2_3
		y3 = y3.contiguous().view(-1, self.hidden_dim)        
		y4  = self.hidden2label(y3)
		y_pred = self.sig(y4)
		return  y_pred, torch.tanh(y3)*0.1


	def forward(self, joints3d_vec, y):
		x3 = joints3d_vec
		x2 = x3.view(-1, 22, 3)
		x2_2 = x2[:,:,0:2].contiguous().view(-1, 1, 44)
		x2_3 = x2[:,:,[0,2]].contiguous().view(-1, 1, 44)
		lstm_out2_2, self.hidden2_2_ = self.lstm2_2(x2_2, self.hidden2_2)
		lstm_out2_3, self.hidden2_3_ = self.lstm2_3(x2_3, self.hidden2_3)
		t2_2 = lstm_out2_2[-1].view(self.batch_size,1,-1)
		t2_3 = lstm_out2_3[-1].view(self.batch_size,1,-1)
		y2_2 = self.conv1_2_2(t2_2)
		y2_3 = self.conv1_2_3(t2_3)
		y3 = y2_2+y2_3
		y3 = y3.contiguous().view(-1, self.hidden_dim)        
		y4  = self.hidden2label(y3)
		y_pred = self.sig(y4)
		loss = F.binary_cross_entropy(y_pred, y.float())
		return loss, y_pred
	
	def predict2(self, joints3d_vec):
		# x3 = joints3d_vec
		x2 = joints3d_vec.view(-1, 22, 3)
		x2_2 = x2[:, :, 0:2].contiguous().view(-1, 1, 44)
		x2_3 = x2[:, :, [0, 2]].contiguous().view(-1, 1, 44)
		lstm_out2_2, _ = self.lstm2_2(x2_2, self.hidden2_2)
		lstm_out2_3, _ = self.lstm2_3(x2_3, self.hidden2_3)
		# Return the last output of the first LSTM layers as embedding
		embedding = torch.cat([
			torch.tanh(lstm_out2_2[-1]),
			torch.tanh(lstm_out2_3[-1])
		], dim=-1) * 0.1
		# For y_pred, use the same as predict
		t2_2 = lstm_out2_2[-1].view(self.batch_size, 1, -1)
		t2_3 = lstm_out2_3[-1].view(self.batch_size, 1, -1)
		y2_2 = self.conv1_2_2(t2_2)
		y2_3 = self.conv1_2_3(t2_3)
		y3 = y2_2 + y2_3
		y3 = y3.contiguous().view(-1, self.hidden_dim)
		y4 = self.hidden2label(y3)
		y_pred = self.sig(y4)
		return y_pred, embedding


class LSTM_Action_Classifier_Hoidini(nn.Module):
	def __init__(self, n_features, hidden_dim=128, label_size=29, num_layers=2):
		super(LSTM_Action_Classifier_Hoidini, self).__init__()
		self.hidden_dim = hidden_dim
		self.num_layers = num_layers
		self.lstm = nn.LSTM(n_features, hidden_dim, num_layers=self.num_layers, batch_first=True)
		self.hidden2label = nn.Linear(hidden_dim, label_size)
		self.sig = nn.Sigmoid()

	def forward(self, x, y):
		# x: [batch_size, seq_len, n_features]
		lstm_out, _ = self.lstm(x)
		# Use the last output for classification
		last_out = lstm_out[:, -1, :]
		logits = self.hidden2label(last_out)
		y_pred = self.sig(logits)
		loss = F.binary_cross_entropy(y_pred, y.float())
		return loss, y_pred
	
	def predict(self, x):
		# x: [batch_size, seq_len, n_features]
		lstm_out, _ = self.lstm(x)
		last_out = lstm_out[:, -1, :]
		logits = self.hidden2label(last_out)
		y_pred = self.sig(logits)
		embedding = torch.tanh(last_out) * 0.1
		return y_pred, embedding
	
	def predict2(self, x):
		# x: [batch_size, seq_len, n_features]
		lstm_out, _ = self.lstm(x)
		# Use the last output of the first LSTM as embedding
		embedding = torch.tanh(lstm_out[:, -1, :]) * 0.1
		# For y_pred, use the same as predict
		last_out = lstm_out[:, -1, :]
		logits = self.hidden2label(last_out)
		y_pred = self.sig(logits)
		return y_pred, embedding


class Conv_Action_Classifier_Hoidini(nn.Module):
	def __init__(self, n_features, hidden_dim=128, label_size=29, num_layers=2):
		super(Conv_Action_Classifier_Hoidini, self).__init__()
		self.hidden_dim = hidden_dim
		self.num_layers = num_layers
		self.n_features = n_features
		self.label_size = label_size
		# Conv1d expects (N, C, L) where C=n_features, L=seq_len
		self.conv = nn.Sequential(
			nn.Conv1d(self.n_features, self.hidden_dim, kernel_size=3, padding=1),
			nn.ReLU(),
			nn.AvgPool1d(kernel_size=4, stride=2, padding=1),
			*[nn.Sequential(
				nn.Conv1d(self.hidden_dim, self.hidden_dim, kernel_size=3, padding=1),
				nn.ReLU()
			) for _ in range(self.num_layers)],
			nn.AvgPool1d(kernel_size=4, stride=2, padding=1),
			nn.Conv1d(self.hidden_dim, self.hidden_dim, kernel_size=3, padding=1),
			nn.ReLU()
		)
		self.hidden2label = nn.Linear(self.hidden_dim, self.label_size)
		self.sig = nn.Sigmoid()

	def forward(self, x, y):
		# x: [batch_size, seq_len, n_features] -> [batch_size, n_features, seq_len]
		x = x.transpose(1, 2)
		x = self.conv(x)  # [batch_size, hidden_dim, new_seq_len]
		x = x.mean(dim=2)  # Global average pooling over time
		logits = self.hidden2label(x)
		y_pred = self.sig(logits)
		loss = F.binary_cross_entropy(y_pred, y.float())
		return loss, y_pred

	def predict(self, x):
		# x: [batch_size, seq_len, n_features] -> [batch_size, n_features, seq_len]
		x = x.transpose(1, 2)
		x = self.conv(x)
		x = x.mean(dim=2)
		logits = self.hidden2label(x)
		y_pred = self.sig(logits)
		embedding = torch.tanh(x) * 0.1
		return y_pred, embedding
	
	def predict2(self, x):
		# x: [batch_size, seq_len, n_features] -> [batch_size, n_features, seq_len]
		x = x.transpose(1, 2)
		# Pass through the first Conv1d + ReLU + AvgPool1d
		conv1 = self.conv[0](x)
		relu1 = self.conv[1](conv1)
		pool1 = self.conv[2](relu1)
		embedding = torch.tanh(pool1.mean(dim=2)) * 0.1  # Global avg pool over time
		# For y_pred, use the same as predict
		x_full = self.conv(x)
		x_full = x_full.mean(dim=2)
		logits = self.hidden2label(x_full)
		y_pred = self.sig(logits)
		return y_pred, embedding


class Transformer_Action_Classifier_Hoidini(nn.Module):
	def __init__(self, n_features, hidden_dim=128, label_size=29, num_layers=2, nhead=4, dim_feedforward=256, dropout=0.1):
		super(Transformer_Action_Classifier_Hoidini, self).__init__()
		self.hidden_dim = hidden_dim
		self.num_layers = num_layers
		self.n_features = n_features
		self.label_size = label_size
		self.input_proj = nn.Linear(n_features, hidden_dim)
		encoder_layer = nn.TransformerEncoderLayer(
			d_model=hidden_dim,
			nhead=nhead,
			dim_feedforward=dim_feedforward,
			dropout=dropout,
			batch_first=True
		)
		self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
		self.hidden2label = nn.Linear(hidden_dim, label_size)
		self.sig = nn.Sigmoid()

	def forward(self, x, y):
		# x: [batch_size, seq_len, n_features]
		x = self.input_proj(x)  # [batch_size, seq_len, hidden_dim]
		x = self.transformer_encoder(x)  # [batch_size, seq_len, hidden_dim]
		x = x.mean(dim=1)  # Global average pooling over time
		logits = self.hidden2label(x)
		y_pred = self.sig(logits)
		loss = F.binary_cross_entropy(y_pred, y.float())
		return loss, y_pred

	def predict(self, x):
		# x: [batch_size, seq_len, n_features]
		x = self.input_proj(x)
		x = self.transformer_encoder(x)
		x = x.mean(dim=1)
		logits = self.hidden2label(x)
		y_pred = self.sig(logits)
		embedding = torch.tanh(x) * 0.1
		return y_pred, embedding
	
	def predict2(self, x):
		# x: [batch_size, seq_len, n_features]
		x_proj = self.input_proj(x)
		# Pass through the first transformer encoder layer only
		first_layer = self.transformer_encoder.layers[0]
		x_first = first_layer(x_proj)
		embedding = torch.tanh(x_first.mean(dim=1)) * 0.1
		# For y_pred, use the same as predict
		x_full = self.transformer_encoder(x_proj)
		x_full = x_full.mean(dim=1)
		logits = self.hidden2label(x_full)
		y_pred = self.sig(logits)
		return y_pred, embedding



class TransformerEmb_Action_Classifier_Hoidini(nn.Module):
	def __init__(self, n_features, hidden_dim=128, label_size=29, num_layers=2, nhead=4, dim_feedforward=256, dropout=0.1, max_seq_len=512):
		super(Transformer_Action_Classifier_Hoidini, self).__init__()
		self.hidden_dim = hidden_dim
		self.num_layers = num_layers
		self.n_features = n_features
		self.label_size = label_size
		self.input_proj = nn.Linear(n_features, hidden_dim)
		self.max_seq_len = max_seq_len
		# Remove learnable pos_embedding, add sinusoidal encoding function
		encoder_layer = nn.TransformerEncoderLayer(
			d_model=hidden_dim,
			nhead=nhead,
			dim_feedforward=dim_feedforward,
			dropout=dropout,
			batch_first=True
		)
		self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
		self.hidden2label = nn.Linear(hidden_dim, label_size)
		self.sig = nn.Sigmoid()

	def get_sinusoid_encoding(self, seq_len, d_model, device):
		position = torch.arange(seq_len, dtype=torch.float, device=device).unsqueeze(1)
		div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float, device=device) * -(np.log(10000.0) / d_model))
		pe = torch.zeros(seq_len, d_model, device=device)
		pe[:, 0::2] = torch.sin(position * div_term)
		pe[:, 1::2] = torch.cos(position * div_term)
		return pe.unsqueeze(0)  # [1, seq_len, d_model]

	def forward(self, x, y):
		# x: [batch_size, seq_len, n_features]
		x = self.input_proj(x)  # [batch_size, seq_len, hidden_dim]
		seq_len = x.size(1)
		# Add sinusoidal positional encoding
		pos_emb = self.get_sinusoid_encoding(seq_len, self.hidden_dim, x.device)
		x = x + pos_emb
		x = self.transformer_encoder(x)  # [batch_size, seq_len, hidden_dim]
		x = x.mean(dim=1)  # Global average pooling over time
		logits = self.hidden2label(x)
		y_pred = self.sig(logits)
		loss = F.binary_cross_entropy(y_pred, y.float())
		return loss, y_pred

	def predict(self, x):
		# x: [batch_size, seq_len, n_features]
		x = self.input_proj(x)
		seq_len = x.size(1)
		pos_emb = self.get_sinusoid_encoding(seq_len, self.hidden_dim, x.device)
		x = x + pos_emb
		x = self.transformer_encoder(x)
		x = x.mean(dim=1)
		logits = self.hidden2label(x)
		y_pred = self.sig(logits)
		embedding = torch.tanh(x) * 0.1
		return y_pred, embedding

	def predict2(self, x):
		# x: [batch_size, seq_len, n_features]
		x_proj = self.input_proj(x)
		seq_len = x_proj.size(1)
		pos_emb = self.get_sinusoid_encoding(seq_len, self.hidden_dim, x.device)
		x_proj = x_proj + pos_emb
		# Pass through the first transformer encoder layer only
		first_layer = self.transformer_encoder.layers[0]
		x_first = first_layer(x_proj)
		embedding = torch.tanh(x_first.mean(dim=1)) * 0.1
		# For y_pred, use the same as predict
		x_full = self.transformer_encoder(x_proj)
		x_full = x_full.mean(dim=1)
		logits = self.hidden2label(x_full)
		y_pred = self.sig(logits)
		return y_pred, embedding
