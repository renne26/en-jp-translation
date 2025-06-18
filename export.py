import torch
import torch.nn as nn
import os
import json
import onnx
import onnxruntime
import numpy as np
from helper import decodePairs

def indexesFromSentence(input_vocab, sentence, max_length):
  doc = input_vocab.tokenizer(sentence)
  tokenList = [input_vocab.getIndex(str(word)) for word in doc]
  tokenList = tokenList[:(max_length - 1)]

  return tokenList

def decodeOutput(output_vocab, decoded_ids, unk_token, eos_token):
  decoded_words = []
  for idx in decoded_ids:
    if idx.item() == unk_token:
      decoded_words.append('<UNK>')
    
    elif idx.item() == eos_token:
      decoded_words.append('<EOS>')
      break

    else:
      decoded_words.append(output_vocab.index2word[idx.item()])
  
  return decoded_words

class Model(nn.Module):
  def __init__(self, encoder, decoder, input_vocab, output_vocab, max_length, eos_token, unk_token):
    super(Model, self).__init__()
    self.encoder = encoder
    self.decoder = decoder
    self.input_vocab = input_vocab
    self.output_vocab = output_vocab
    self.unk_token = unk_token
    self.eos_token = eos_token
    self.max_length = max_length

  def forward(self, input_tensor):
    encoder_outputs, encoder_hidden = self.encoder(input_tensor.view(1, -1))
    decoder_outputs, decoder_hidden, decoder_attn = self.decoder(encoder_outputs, encoder_hidden)
    _, topi = decoder_outputs.topk(1)
    decoded_ids = topi.squeeze()

    return decoded_ids

def exportModel(encoder, decoder, input_vocab, output_vocab, max_length, eos_token, unk_token, testLoader):
  if not os.path.exists(f'./Exports/'):
    os.makedirs(f'./Exports/')
    os.makedirs(f'./Exports/Models/')
    os.makedirs(f'./Exports/Vocab/')
  
  pair = testLoader.dataset[np.random.randint(0, len(testLoader.dataset))]
  example_inputs = pair[0].cpu()
  decoder.device = 'cpu'
  print('Exporting model to ONNX format')
  model = Model(encoder, decoder, input_vocab, output_vocab, max_length, eos_token, unk_token).cpu()
  onnx_program = torch.onnx.export(model, example_inputs, input_names=['Input'], dynamo=True, opset_version=21)
  onnx_program.optimize()

  model_export_dir = './Exports/Models/en_jp_translation_model.onnx'
  onnx_program.save(model_export_dir)

  print('Exporting vocab')
  input_vocab_json = {
    'name': input_vocab.name,
    'word2index': input_vocab.word2index,
    'word2count': input_vocab.word2count,
    'index2word': input_vocab.index2word,
    'n_words': input_vocab.n_words
  }

  output_vocab_json = {
    'name': output_vocab.name,
    'word2index': output_vocab.word2index,
    'word2count': output_vocab.word2count,
    'index2word': output_vocab.index2word,
    'n_words': output_vocab.n_words
  }

  with open('./Exports/Vocab/input_vocab.json', 'w') as file:
    json.dump(input_vocab_json, file)
  
  with open('./Exports/Vocab/output_vocab.json', 'w') as file:
    json.dump(output_vocab_json, file)

  print('Loading model to validate predictions')
  onnx_model = onnx.load(model_export_dir)
  onnx.checker.check_model(onnx_model)

  decoded_onnx_input, _ = decodePairs(pair[0], pair[1], input_vocab, output_vocab, unk_token, eos_token)

  print(f'Input length: {len(pair[0])}')
  print(f'Sample input: {decoded_onnx_input}')

  ort_session = onnxruntime.InferenceSession(model_export_dir, providers=['CPUExecutionProvider'])
  onnx_input = {'Input': pair[0].cpu().numpy()}
  onnx_outputs = ort_session.run(None, onnx_input)[0]
  torch_outputs = model(pair[0].cpu())

  print(len(torch_outputs) == len(onnx_outputs))
  print(f'Torch outputs: {decodeOutput(output_vocab, torch_outputs.detach().numpy(), unk_token, eos_token)}')
  print(f'ONNX outputs: {decodeOutput(output_vocab, onnx_outputs, unk_token, eos_token)}')
  print()

  print('Testing prediction from raw text')
  text = 'Hello World'
  print(f'Raw Text: {text}')
  onnx_input = np.zeros(max_length, dtype=np.int64)
  onnx_input_ids = indexesFromSentence(input_vocab, text, max_length)
  onnx_input_ids.append(eos_token)
  onnx_input[:len(onnx_input_ids)] = onnx_input_ids
  onnx_outputs = ort_session.run(None, {'Input': np.array(onnx_input)})[0]
  print(f'Output: {decodeOutput(output_vocab, onnx_outputs, unk_token, eos_token)}')