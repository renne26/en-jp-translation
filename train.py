from tqdm import tqdm
import os

import numpy as np
import time

import torch
import torch.nn as nn
from torch import optim
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_float32_matmul_precision('medium')
scaler = torch.amp.GradScaler()

from model import EncoderRNN, AttnDecoderRNN
from helper import timeSince, showPlot, showAttention, get_dataloader, decodePairs

def train_epoch(dataloader, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion):
  encoder.train()
  decoder.train()

  with tqdm(total=len(dataloader), desc='Training') as progress_bar:
    for data in dataloader:
      input_tensor, target_tensor = data

      with torch.autocast(device_type='cuda', dtype=torch.float16):
        encoder_outputs, encoder_hidden = encoder(input_tensor)
        decoder_outputs, _, _ = decoder(encoder_outputs, encoder_hidden, target_tensor)

        loss = criterion(decoder_outputs.view(-1, decoder_outputs.size(-1)), target_tensor.view(-1))
      
      scaler.scale(loss).backward()
      scaler.unscale_(encoder_optimizer)

      scaler.step(encoder_optimizer)
      scaler.step(decoder_optimizer)
      scaler.update()
      encoder_optimizer.zero_grad(set_to_none=True)
      decoder_optimizer.zero_grad(set_to_none=True)

      progress_bar.update(1)

def train(n_epochs, batch_size, hidden_size, learning_rate, plot_every, max_length, sos_token, eos_token, unk_token, trainCheckpoint=False):
  plot_losses = []
  best_dev_loss = 999
  epoch_start = 1

  input_vocab, output_vocab, dataloaders = get_dataloader(batch_size, max_length, eos_token)
  encoder = EncoderRNN(input_vocab.n_words, hidden_size).to(device)
  decoder = AttnDecoderRNN(hidden_size, output_vocab.n_words, sos_token, max_length, device).to(device)
  encoder_optimizer = optim.AdamW(encoder.parameters(), lr=learning_rate)
  decoder_optimizer = optim.AdamW(decoder.parameters(), lr=learning_rate)
  criterion = nn.NLLLoss()
  
  if not os.path.exists('./Checkpoints'):
    os.makedirs('Checkpoints')
    
  elif (trainCheckpoint == True and os.path.isfile(f'./Checkpoints/model_{hidden_size}_{batch_size}_{learning_rate}_{max_length}')):
    checkpoint = torch.load(f'./Checkpoints/model_{hidden_size}_{batch_size}_{learning_rate}_{max_length}.pt')
    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    decoder.load_state_dict(checkpoint['decoder_state_dict'])
    encoder_optimizer.load_state_dict(checkpoint['encoder_optimizer_state_dict'])
    decoder_optimizer.load_state_dict(checkpoint['decoder_optimizer_state_dict'])
    scaler.load_state_dict(checkpoint['scaler_state_dict'])
    epoch_start = checkpoint['epoch']
    best_dev_loss = checkpoint['dev_loss']
    plot_losses = checkpoint['plot_losses']
    print(f'Continuing from epoch: {epoch_start}')
    print(f'Best Dev loss: {best_dev_loss}')
  
  else:
    print('Checkpoint not found, starting training loop from scratch')
  
  start = time.time()
  
  for epoch in range(epoch_start, n_epochs + 1):
    train_epoch(dataloaders['train'], encoder, decoder, encoder_optimizer, decoder_optimizer, criterion)

    dev_loss = evaluate(encoder, decoder, criterion, input_vocab, output_vocab, eos_token, unk_token, dataloaders['dev'], 'Dev', epoch, epoch % plot_every == 0, 3)
    plot_losses.append(dev_loss)

    torch.save({
      'epoch': epoch,
      'encoder_state_dict': encoder.state_dict(),
      'decoder_state_dict': decoder.state_dict(),
      'encoder_optimizer_state_dict': encoder_optimizer.state_dict(),
      'decoder_optimizer_state_dict': decoder_optimizer.state_dict(),
      'scaler_state_dict': scaler.state_dict(),
      'dev_loss': dev_loss,
      'plot_losses': plot_losses
    }, f'./Checkpoints/model_{hidden_size}_{batch_size}_{learning_rate}_{max_length}.pt')

    if (dev_loss < best_dev_loss):
      print(f'Best model saved at epoch: {epoch}')
      best_dev_loss = dev_loss
      torch.save({
        'encoder_state_dict': encoder.state_dict(),
        'decoder_state_dict': decoder.state_dict()
      }, f'./Checkpoints/best_model_{hidden_size}_{batch_size}_{learning_rate}_{max_length}.pt')

    print(f'{timeSince(start, epoch / n_epochs)} Epoch: {epoch} / {n_epochs} ({epoch / n_epochs * 100}%) Dev Loss: {dev_loss} Best Dev Loss: {best_dev_loss}')
    print()
  
  showPlot(plot_losses)
  best_checkpoint = torch.load(f'./Checkpoints/best_model_{hidden_size}_{batch_size}_{learning_rate}_{max_length}.pt')
  encoder.load_state_dict(best_checkpoint['encoder_state_dict'])
  decoder.load_state_dict(best_checkpoint['decoder_state_dict'])
  
  return encoder, decoder, criterion, input_vocab, output_vocab, dataloaders

def predict(encoder, decoder, input_tensor, output_vocab, eos_token, unk_token):
  with torch.no_grad():
    encoder_outputs, encoder_hidden = encoder(input_tensor.view(1, -1))
    decoder_outputs, decoder_hidden, decoder_attn = decoder(encoder_outputs, encoder_hidden)

    _, topi = decoder_outputs.topk(1)
    decoded_ids = topi.squeeze()

    decoded_words = []
    for idx in decoded_ids:
      if idx.item() == unk_token:
        decoded_words.append('<UNK>')

      elif idx.item() == eos_token:
        decoded_words.append('<EOS>')
        break
      
      else:
        decoded_words.append(output_vocab.index2word[idx.item()])
  
  del encoder_outputs, encoder_hidden, decoder_outputs, decoder_hidden
  return decoded_words, decoder_attn

def evaluate(encoder, decoder, criterion, input_vocab, output_vocab, eos_token, unk_token, dataloader, dataloaderType, epoch, plot, n=10):
  dev_loss = 0
  encoder.eval()
  decoder.eval()

  for data in dataloader:
    input_tensor, target_tensor = data

    encoder_outputs, encoder_hidden = encoder(input_tensor)
    decoder_outputs, decoder_hidden, decoder_attn = decoder(encoder_outputs, encoder_hidden, target_tensor)
    dev_loss = criterion(decoder_outputs.view(-1, decoder_outputs.size(-1)), target_tensor.view(-1)).item() + dev_loss

  del data, input_tensor, target_tensor, encoder_outputs, encoder_hidden, decoder_outputs, decoder_hidden, decoder_attn
  torch.cuda.empty_cache()
  dev_loss = round(dev_loss / len(dataloader), 5)

  if(plot):
    for i in range(n):
      pair = dataloader.dataset[np.random.randint(0, len(dataloader.dataset))]
      inputPair = pair[0]
      targetPair = pair[1]

      output_words, attention = predict(encoder, decoder, inputPair, output_vocab, eos_token, unk_token)
      output_sentence = ' '.join(output_words)

      input_sentence, target_sentence = decodePairs(inputPair, targetPair, input_vocab, output_vocab, unk_token, eos_token)

      print(f'Input: {input_sentence}')
      print(f'Output: {output_sentence}')
      print(f'Target: {target_sentence}')
      print()
      showAttention(input_sentence, output_words, attention[0, :len(output_words), :], f'attn_plot_{i + 1}', dataloaderType, epoch)
      del attention
  
  return dev_loss