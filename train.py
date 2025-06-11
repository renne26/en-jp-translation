from spacy.lang.en import English
from spacy.lang.ja import Japanese

from tqdm import tqdm

import numpy as np
import random
import time

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from vocab import prepare_data
from model import EncoderRNN, AttnDecoderRNN
from helper import timeSince, showPlot, showAttention

MAX_LENGTH = 10
SOS_TOKEN = 0
EOS_TOKEN = 1

HIDDEN_SIZE = 128
BATCH_SIZE = 512
EPOCHS = 80

ja_text = "スリバン人です"
en_text = "it's suliban."

def indexesFromSentence(vocab, sentence):
  doc = vocab.tokenizer(sentence)
  tokenList = [vocab.word2index[str(word)] for word in doc]
  tokenList = tokenList[:(MAX_LENGTH - 1)]
  
  return tokenList

def tensorFromSentence(vocab, sentence):
  indexes = indexesFromSentence(vocab, sentence)
  indexes.append(EOS_TOKEN)
  return torch.tensor(indexes, dtype=torch.long, device=device).view(1, -1)

def get_dataloader(batch_size):
  input_vocab, output_vocab, pairs = prepare_data('english', 'japanese', English(), Japanese())

  n = len(pairs)
  input_ids = np.zeros((n, MAX_LENGTH), dtype=np.uint64)
  target_ids = np.zeros((n, MAX_LENGTH), dtype=np.uint64)

  with tqdm(total=n, desc='Indexing Pairs') as progress_bar:
    for idx, (inp, tgt) in enumerate(pairs):
      inp_ids = indexesFromSentence(input_vocab, inp)
      tgt_ids = indexesFromSentence(output_vocab, tgt)
      inp_ids.append(EOS_TOKEN)
      tgt_ids.append(EOS_TOKEN)
      input_ids[idx, :len(inp_ids)] = inp_ids
      target_ids[idx, :len(tgt_ids)] = tgt_ids
      progress_bar.update(1)
  
  train_data = TensorDataset(torch.tensor(input_ids, dtype=torch.long, device=device), torch.tensor(target_ids, dtype=torch.long, device=device))

  train_sampler = RandomSampler(train_data)
  train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)
  return input_vocab, output_vocab, pairs, train_dataloader

def train_epoch(dataloader, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion):
  total_loss = 0

  with tqdm(total=len(dataloader), desc='Training') as progress_bar:
    for data in dataloader:
      input_tensor, target_tensor = data

      encoder_optimizer.zero_grad()
      decoder_optimizer.zero_grad()

      encoder_outputs, encoder_hidden = encoder(input_tensor)
      decoder_outputs, _, _ = decoder(encoder_outputs, encoder_hidden, target_tensor)

      loss = criterion(decoder_outputs.view(-1, decoder_outputs.size(-1)), target_tensor.view(-1))
      loss.backward()

      encoder_optimizer.step()
      decoder_optimizer.step()

      total_loss += loss.item()
      progress_bar.update(1)
  
  return total_loss / len(dataloader)

def train(train_dataloader, encoder, decoder, n_epochs, learning_rate=0.001, print_every=100, plot_every=100):
  start = time.time()
  plot_losses = []
  print_loss_total = 0
  plot_loss_total = 0

  encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
  decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)
  criterion = nn.NLLLoss()

  for epoch in range(1, n_epochs + 1):
    loss = train_epoch(train_dataloader, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion)
    print_loss_total += loss
    plot_loss_total += loss

    if epoch % print_every == 0:
      print_loss_avg = print_loss_total / print_every
      print_loss_total = 0
      print(f'{timeSince(start, epoch / n_epochs)} Epoch: {epoch} / {n_epochs} ({epoch / n_epochs * 100}%) Average Loss: {print_loss_avg}')
    
    if epoch % plot_every == 0:
      plot_loss_avg = plot_loss_total / plot_every
      plot_losses.append(plot_loss_avg)
      plot_loss_total = 0
  
  showPlot(plot_losses)

def evaluate(encoder, decoder, sentence, input_vocab, output_vocab):
  with torch.no_grad():
    input_tensor = tensorFromSentence(input_vocab, sentence)

    encoder_outputs, encoder_hidden = encoder(input_tensor)
    decoder_outputs, decoder_hidden, decoder_attn = decoder(encoder_outputs, encoder_hidden)

    _, topi = decoder_outputs.topk(1)
    decoded_ids = topi.squeeze()

    decoded_words = []
    for idx in decoded_ids:
      if idx.item() == EOS_TOKEN:
        decoded_words.append('<EOS>')
        break

      decoded_words.append(output_vocab.index2word[idx.item()])
  
  return decoded_words, decoder_attn

def evaluateRandomly(encoder, decoder, pairs, n=10):
  for i in range(n):
    pair = random.choice(pairs)
    output_words, attention = evaluate(encoder, decoder, pair[0], input_vocab, output_vocab)
    output_sentence = ' '.join(output_words)
    print(f'Input: {pair[0]}')
    print(f'Output: {output_sentence}')
    print(f'Target: {pair[1]}')
    print()
    showAttention(pair[0], output_words, attention[0, :len(output_words), :], f'attn_plot{i}')

input_vocab, output_vocab, pairs, train_dataloader = get_dataloader(BATCH_SIZE)
encoder = EncoderRNN(input_vocab.n_words, HIDDEN_SIZE).to(device)
decoder = AttnDecoderRNN(HIDDEN_SIZE, output_vocab.n_words, SOS_TOKEN).to(device)

train(train_dataloader, encoder, decoder, EPOCHS, print_every=5, plot_every=5)

encoder.eval()
decoder.eval()
evaluateRandomly(encoder, decoder, pairs)