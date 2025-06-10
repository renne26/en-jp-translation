import pandas

class Vocab:
  def __init__(self, name, tokenizer):
    self.name = name
    self.word2index = {}
    self.word2count = {}
    self.index2word = {0: "SOS", 1: "EOS"}
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

def read_data(lang1, lang2, tokenizer1, tokenizer2):
  pairs = []
  input_vocab = Vocab(lang1, tokenizer1)
  output_vocab = Vocab(lang2, tokenizer2)

  with pandas.read_csv('./Data/split/train', sep='\t', header=None, names=['english', 'japanese'], chunksize=10000, quoting=3) as reader:
    for i, data in enumerate(reader):
      print(f'Building pairs for chunk {i + 1}')
      pairs = pairs + data.to_numpy().tolist()
  
  return input_vocab, output_vocab, pairs

def prepare_data(lang1, lang2, tokenizer1, tokenizer2):
  input_vocab, output_vocab, pairs = read_data(lang1, lang2, tokenizer1, tokenizer2)

  for pair in pairs:
    input_vocab.addSentence(pair[0])
    output_vocab.addSentence(pair[1])
  
  print("Counted words:")
  print(input_vocab.name, input_vocab.n_words)
  print(output_vocab.name, output_vocab.n_words)
  return input_vocab, output_vocab, pairs