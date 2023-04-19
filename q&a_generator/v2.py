import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.data import Field, BucketIterator
from torchtext import data
import random
from nltk.tokenize import word_tokenize


class AnswerIdentifier(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, output_dim, n_layers, dropout):
        super(AnswerIdentifier, self).__init__()
        self.embedding = nn.Embedding(input_dim, embedding_dim)
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, n_layers, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        embedded = self.dropout(self.embedding(src))
        outputs, (hidden, cell) = self.rnn(embedded)
        prediction = self.fc(outputs.squeeze(0))
        return prediction

class Encoder(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, n_layers, dropout):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(input_dim, embedding_dim)
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, n_layers, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        embedded = self.dropout(self.embedding(src))
        outputs, (hidden, cell) = self.rnn(embedded)
        return hidden, cell


class Decoder(nn.Module):
    def __init__(self, output_dim, embedding_dim, hidden_dim, n_layers, dropout):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(output_dim, embedding_dim)
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, n_layers, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, cell):
        input = input.unsqueeze(0)
        embedded = self.dropout(self.embedding(input))
        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))
        prediction = self.fc(output.squeeze(0))
        return prediction, hidden, cell

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        batch_size = trg.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim

        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)
        hidden, cell = self.encoder(src)

        input = trg[0, :]
        for t in range(1, trg_len):
            output, hidden, cell = self.decoder(input, hidden, cell)
            outputs[t] = output
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.argmax(1)
            input = trg[t] if teacher_force else top1

        return outputs


def train(model, iterator, optimizer, criterion, clip):
    model.train()
    epoch_loss = 0

    for _, batch in enumerate(iterator):
        src = batch.src
        trg = batch.trg

        optimizer.zero_grad()
        output = model(src, trg)

        output_dim = output.shape[-1]
        output = output[1:].view(-1, output_dim)
        trg = trg[1:].view(-1)

        loss = criterion(output, trg)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(iterator)


def evaluate(model, iterator, criterion):
    model.eval()
    epoch_loss = 0

    with torch.no_grad():
        for _, batch in enumerate(iterator):
            src = batch.src
            trg = batch.trg

            output = model(src, trg, 0)  # turn off teacher forcing

            output_dim = output.shape[-1]
            output = output[1:].view(-1, output_dim)
            trg = trg[1:].view(-1)

            loss = criterion(output, trg)
            epoch_loss += loss.item()

    return epoch_loss / len(iterator)

def tokenize_src(text):
    return word_tokenize(text)

def tokenize_trg(text):
    return word_tokenize(text)

# Define the fields for the source and target texts
SRC = Field(tokenize=tokenize_src, init_token='<sos>', eos_token='<eos>', lower=True)
TRG = Field(tokenize=tokenize_trg, init_token='<sos>', eos_token='<eos>', lower=True)

# Load and split the dataset
train_data, valid_data, test_data = data.TabularDataset.splits(
    path='path_to_data_folder',
    train='train.csv',
    validation='valid.csv',
    test='test.csv',
    format='csv',
    fields=[('src', SRC), ('trg', TRG)]
)

# Build the vocabularies for the source and target texts
SRC.build_vocab(train_data, min_freq=2)
TRG.build_vocab(train_data, min_freq=2)


### Hyperparameters
input_dim = len(SRC.vocab)
output_dim = len(TRG.vocab)
embedding_dim = 128
hidden_dim = 128
n_layers = 2
dropout = 0.3
batch_size = 32
device = 'mps' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks

# Create the iterators
train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
    (train_data, valid_data, test_data),
    batch_size=batch_size,
    device=device,
    sort_key=lambda x: len(x.src),
    sort_within_batch=True
)

# Instantiate the models
encoder = Encoder(input_dim, embedding_dim, hidden_dim, n_layers, dropout)
decoder = Decoder(output_dim, embedding_dim, hidden_dim, n_layers, dropout)
seq2seq = Seq2Seq(encoder, decoder, device).to(device)

# Set up training parameters
optimizer = optim.Adam(seq2seq.parameters())
criterion = nn.CrossEntropyLoss(ignore_index=TRG.vocab.stoi[TRG.pad_token])

# Train the model
num_epochs = 10
clip = 1
best_valid_loss = float('inf')

for epoch in range(num_epochs):
    train_loss = train(seq2seq, train_iterator, optimizer, criterion, clip)
    valid_loss = evaluate(seq2seq, valid_iterator, criterion)

    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(seq2seq.state_dict(), 'best_model.pt')

    print(f"Epoch: {epoch + 1}, Train Loss: {train_loss:.3f}, Valid Loss: {valid_loss:.3f}")
    
seq2seq.load_state_dict(torch.load('best_model.pt'))
test_loss = evaluate(seq2seq, test_iterator, criterion)
print(f"Test Loss: {test_loss:.3f}")

