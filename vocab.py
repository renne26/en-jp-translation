import pandas
from tqdm import tqdm
import os
import pickle

class Vocab:
  def __init__(self, name, tokenizer):
    self.name = name
    self.word2index = {}
    self.word2count = {}
    self.index2word = {0: "SOS", 1: "EOS", 2: "UNK"}
    self.n_words = 2
    self.tokenizer = tokenizer
  
  def addSentence(self, sentence):
    doc = self.tokenizer(sentence)
    for token in doc:
      self.addWord(str(token))
  
  def addWord(self, word):
    if word not in self.word2index:
      self.word2index[word] = self.n_words
      self.word2count[word] = 1
      self.index2word[self.n_words] = word
      self.n_words += 1
    else:
      self.word2count[word] += 1
  
  def getIndex(self, word):
    if word not in self.word2index:
      return 2
    else:
      return self.word2index[word]

def read_data(lang1, lang2, tokenizer1, tokenizer2):
  datasets = {}
  input_vocab = Vocab(lang1, tokenizer1)
  output_vocab = Vocab(lang2, tokenizer2)

  for dataset in os.listdir('./Data/split'):
    pairs = []

    with pandas.read_csv(f'./Data/split/{dataset}', sep='\t', header=None, names=['english', 'japanese'], chunksize=10000, quoting=3) as reader:
      for i, data in enumerate(reader):
        print(f'Building pairs for chunk {i + 1}')
        pairs = pairs + data.to_numpy().tolist()

      print()
    
    datasets[dataset] = pairs
  
  return input_vocab, output_vocab, datasets

def prepare_data(lang1, lang2, tokenizer1, tokenizer2):
  if (os.path.isfile('./Vocab/input_vocab.pkl') and os.path.isfile('./Vocab/output_vocab.pkl')):
    _, _, datasets = read_data(lang1, lang2, tokenizer1, tokenizer2)

    print('Loading Vocab')
    with open('./Vocab/input_vocab.pkl', 'rb') as file:
      input_vocab = pickle.load(file)
    
    with open('./Vocab/output_vocab.pkl', 'rb') as file:
      output_vocab = pickle.load(file)
  else:
    input_vocab, output_vocab, datasets = read_data(lang1, lang2, tokenizer1, tokenizer2)

    with tqdm(total=len(datasets['train']), desc='Building Vocab') as progress_bar:
      for pair in datasets['train']:
        input_vocab.addSentence(pair[0])
        output_vocab.addSentence(pair[1])
        progress_bar.update(1)
  
    if not os.path.exists(f'./Vocab/'):
      os.makedirs(f'./Vocab/')
    print('Saving Vocab')
    with open('./Vocab/input_vocab.pkl', 'wb') as file:
      pickle.dump(input_vocab, file)
    
    with open('./Vocab/output_vocab.pkl', 'wb') as file:
      pickle.dump(output_vocab, file)

  print('Counted words:')
  print(input_vocab.name, input_vocab.n_words)
  print(output_vocab.name, output_vocab.n_words)
  print()

  return input_vocab, output_vocab, datasets