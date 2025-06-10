import time
import math
import os
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.font_manager import FontProperties

plt.switch_backend('agg')
prop = FontProperties()
prop.set_file('./Fonts/NotoSansJP.ttf')

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
  plt.figure()
  fig, ax = plt.subplots()
  loc = ticker.MultipleLocator(base=0.2)
  ax.yaxis.set_major_locator(loc)
  plt.plot(points)

  if not os.path.exists('./Plots'):
    os.makedirs('Plots')

  plt.savefig('./plots/loss_plot.png')

def showAttention(input_sentence, output_words, attentions, file_name):
  fig = plt.figure(figsize=(5, 5))
  ax = fig.add_subplot(111)
  cax = ax.matshow(attentions.cpu().numpy(), cmap='bone')
  fig.colorbar(cax)
  xLabels = input_sentence.split(' ') + ['<EOS>']
  yLabels = output_words
  
  ax.set_xticks(range(len(xLabels)))
  ax.set_yticks(range(len(yLabels)))
  ax.set_xticklabels(xLabels, rotation=90)
  ax.set_yticklabels(yLabels, fontproperties=prop)

  ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
  ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

  plt.savefig(f'./Plots/{file_name}')