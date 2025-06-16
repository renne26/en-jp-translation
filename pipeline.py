from preprocess import preprocess
from train import train, evaluate
from export import exportModel

MAX_LENGTH = 32
SOS_TOKEN = 0
EOS_TOKEN = 1
UNK_TOKEN = 2

HIDDEN_SIZE = 128
BATCH_SIZE = 256
EPOCHS = 80
LEARNING_RATE = 0.005
PLOT_EVERY = 5

if __name__ == '__main__':
    print('Starting preprocessing of data')
    preprocess()

    print('Starting training')
    encoder, decoder, criterion, input_vocab, output_vocab, dataloaders = train(EPOCHS, BATCH_SIZE, HIDDEN_SIZE, LEARNING_RATE, PLOT_EVERY, MAX_LENGTH, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN, trainCheckpoint=False)

    print('Starting evaluation on test set')
    encoder.eval()
    decoder.eval()
    evaluate(encoder, decoder, criterion, input_vocab, output_vocab, EOS_TOKEN, UNK_TOKEN, dataloaders['test'], 'test', True)

    exportModel(encoder.cpu(), decoder.cpu(), input_vocab, output_vocab, MAX_LENGTH, EOS_TOKEN, UNK_TOKEN, dataloaders['test'])