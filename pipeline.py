from preprocess import preprocess
from train import train, evaluateRandomly
from export import export

MAX_LENGTH = 32
SOS_TOKEN = 0
EOS_TOKEN = 1
UNK_TOKEN = 2

HIDDEN_SIZE = 128
BATCH_SIZE = 256
EPOCHS = 80
LEARNING_RATE = 0.005
EVAL_EVERY = 5

if __name__ == '__main__':
    preprocess()

    encoder, decoder, input_vocab, output_vocab, dataloaders = train(EPOCHS, BATCH_SIZE, HIDDEN_SIZE, LEARNING_RATE, EVAL_EVERY, MAX_LENGTH, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN, trainCheckpoint=False)

    encoder.eval()
    decoder.eval()
    evaluateRandomly(encoder, decoder, input_vocab, output_vocab, EOS_TOKEN, UNK_TOKEN, dataloaders['test'], 'Test')