import time
import math
import os
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.font_manager import FontProperties

from spacy.lang.en import English
from spacy.lang.ja import Japanese

from tqdm import tqdm
import pickle

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from vocab import prepare_data

plt.switch_backend('agg')
prop = FontProperties()
prop.set_file('./Fonts/NotoSansJP.ttf')

def indexesFromSentence(vocab, sentence, max_length):
  doc = vocab.tokenizer(sentence)
  tokenList = [vocab.getIndex(str(word)) for word in doc]
  tokenList = tokenList[:(max_length - 1)]
  
  return tokenList

def tensorFromSentence(vocab, sentence, max_length, eos_token):
  indexes = indexesFromSentence(vocab, sentence, max_length)
  indexes.append(eos_token)
  return torch.tensor(indexes, dtype=torch.long, device=device).view(1, -1)

def get_dataloader(batch_size, max_length, eos_token):
  dataloaders = {}
  input_vocab, output_vocab, datasets = prepare_data('english', 'japanese', English(), Japanese())

  for datasetType in datasets:

    if not os.path.exists(f'./Data/processed'):
      os.makedirs(f'./Data/processed')

    if (os.path.isfile(f'./Data/processed/input_ids_{datasetType}.pkl') and os.path.isfile(f'./Data/processed/target_ids_{datasetType}.pkl')):
      print(f'Loading Input and Target IDs for {datasetType}')

      with open(f'./Data/processed/input_ids_{datasetType}.pkl', 'rb') as file:
        input_ids = pickle.load(file)
      
      with open(f'./Data/processed/target_ids_{datasetType}.pkl', 'rb') as file:
        target_ids = pickle.load(file)
    else:
      dataset = datasets[datasetType]
      n = len(dataset)
      input_ids = np.zeros((n, max_length), dtype=np.uint64)
      target_ids = np.zeros((n, max_length), dtype=np.uint64)

      with tqdm(total=n, desc=f'Indexing Pairs for {datasetType} set') as progress_bar:
        for idx, (inp, tgt) in enumerate(dataset):
          inp_ids = indexesFromSentence(input_vocab, inp, max_length)
          tgt_ids = indexesFromSentence(output_vocab, tgt, max_length)
          inp_ids.append(eos_token)
          tgt_ids.append(eos_token)
          input_ids[idx, :len(inp_ids)] = inp_ids
          target_ids[idx, :len(tgt_ids)] = tgt_ids
          progress_bar.update(1)
        
        with open(f'./Data/processed/input_ids_{datasetType}.pkl', 'wb') as file:
          pickle.dump(input_ids, file)
        
        with open(f'./Data/processed/target_ids_{datasetType}.pkl', 'wb') as file:
          pickle.dump(target_ids, file)

    data = TensorDataset(torch.tensor(input_ids, dtype=torch.long, device=device), torch.tensor(target_ids, dtype=torch.long, device=device))

    sampler = RandomSampler(data)
    dataloader = DataLoader(data, sampler=sampler, batch_size=batch_size)
    dataloaders[datasetType] = dataloader
  
  return input_vocab, output_vocab, dataloaders

def asMinutes(s):
  m = math.floor(s / 60)
  s -= m * 60
  return '%dm %ds' % (m, s)

def timeSince(since, percent):
  now = time.time()
  s = now - since
  es = s / (percent)
  rs = es - s
  return 'Elapsed Time: %s Estimated Time: (- %s)' % (asMinutes(s), asMinutes(rs))

def showPlot(points):
  if not os.path.exists('./Plots'):
    os.makedirs('Plots')
  
  if not os.path.exists(f'./Plots/loss'):
    os.makedirs(f'./Plots/loss')

  plt.figure()
  fig, ax = plt.subplots()
  ax.set_title('Loss')
  ax.set_ylim(min(points) - 0.5, max(points) + 0.5)
  ax.set_xticks(range(len(points)))
  loc = ticker.MultipleLocator(base=0.2)
  ax.yaxis.set_major_locator(loc)
  plt.plot(points)

  plt.savefig('./plots/loss/loss_plot.png')
  plt.close()

def showAttention(input_sentence, output_words, attentions, file_name, dataloaderType):
  if not os.path.exists('./Plots'):
    os.makedirs('Plots')
  
  if not os.path.exists(f'./Plots/{dataloaderType}'):
    os.makedirs(f'./Plots/{dataloaderType}')

  fig = plt.figure(figsize=(5, 5))
  ax = fig.add_subplot(111)
  cax = ax.matshow(attentions.cpu().numpy(), cmap='bone')
  fig.colorbar(cax)
  xLabels = input_sentence.split(' ')
  yLabels = output_words
  
  ax.set_xticks(range(len(xLabels)))
  ax.set_yticks(range(len(yLabels)))
  ax.set_xticklabels(xLabels, rotation=90)
  ax.set_yticklabels(yLabels, fontproperties=prop)

  ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
  ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

  plt.savefig(f'./Plots/{dataloaderType}/{file_name}')
  plt.close()