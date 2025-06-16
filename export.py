import torch
import torch.nn as nn
import os
import onnx
import onnxruntime
import numpy as np
from helper import decodePairs

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
  
  def topk(self, array, k, axis=-1, sorted=True):
    partitioned_ind = (np.argpartition(array, -k, axis=axis).take(indices=range(-k, 0), axis=axis))
    partitioned_scores = np.take_along_axis(array, partitioned_ind, axis=axis)

    if sorted:
      sorted_trunc_ind = np.flip(np.argsort(partitioned_scores, axis=axis), axis=axis)
      ind = np.take_along_axis(partitioned_ind, sorted_trunc_ind, axis=axis)
      scores = np.take_along_axis(partitioned_scores, sorted_trunc_ind, axis=axis)
    else:
      ind = partitioned_ind
      scores = partitioned_scores

    return scores, ind

  def indexesFromSentence(self, input_vocab, sentence):
    doc = input_vocab.tokenizer(sentence)
    tokenList = [input_vocab.getIndex(str(word)) for word in doc]
    tokenList = tokenList[:(self.max_length - 1)]

    return tokenList
  
  def tensorFromSentence(self, input_vocab, sentence, eos_token):
    indexes = self.indexesFromSentence(input_vocab, sentence)
    indexes.append(eos_token)
    return torch.tensor(indexes, dtype=torch.long).view(1, -1)

  def decodeOutput(self, decoder_outputs):
    _, topi = self.topk(decoder_outputs, 1)
    decoded_ids = topi.squeeze()

    decoded_words = []
    for idx in decoded_ids:
      if idx.item() == self.unk_token:
        decoded_words.append('<UNK>')
      
      elif idx.item() == self.eos_token:
        decoded_words.append('<EOS>')
        break

      else:
        decoded_words.append(self.output_vocab.index2word[idx.item()])
    
    return decoded_words

  def forward(self, input_tensor):
    encoder_outputs, encoder_hidden = self.encoder(input_tensor.view(1, -1))
    decoder_outputs, decoder_hidden, decoder_attn = self.decoder(encoder_outputs, encoder_hidden)

    return decoder_outputs

def exportModel(encoder, decoder, input_vocab, output_vocab, max_length, eos_token, unk_token, testLoader):
  if not os.path.exists(f'./Models/'):
    os.makedirs(f'./Models/')
  
  pair = testLoader.dataset[np.random.randint(0, len(testLoader.dataset))]
  example_inputs = pair[0].cpu()
  decoder.device = 'cpu'
  print('Exporting model to ONNX format')
  model = Model(encoder, decoder, input_vocab, output_vocab, max_length, eos_token, unk_token).cpu()
  onnx_program = torch.onnx.export(model, example_inputs, input_names=['Input'], dynamo=True, opset_version=21)
  onnx_program.optimize()

  onnx_program.save('./Models/en_jp_translation_model.onnx')

  onnx_model = onnx.load('./Models/en_jp_translation_model.onnx')
  onnx.checker.check_model(onnx_model)

  decoded_onnx_input, _ = decodePairs(pair[0], pair[1], input_vocab, output_vocab, unk_token, eos_token)

  print(f'Input length: {len(pair[0])}')
  print(f'Sample input: {decoded_onnx_input}')

  ort_session = onnxruntime.InferenceSession('./Models/en_jp_translation_model.onnx', providers=['CPUExecutionProvider'])
  onnx_input = {'Input': pair[0].cpu().numpy()}
  onnx_outputs = ort_session.run(None, onnx_input)[0]
  torch_outputs = model(pair[0].cpu())

  print(len(torch_outputs) == len(onnx_outputs))
  print(f'Torch outputs: {onnx_model.decodeOutput(torch_outputs.detach().numpy())}')
  print(f'ONNX outputs: {onnx_model.decodeOutput(onnx_outputs)}')