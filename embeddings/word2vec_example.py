import torch
from torch import nn, optim
from torch.autograd import Variable
import torch.nn.functional as F
import numpy

CONTEXT_SIZE = 2  # 2 words to the left, 2 to the right
EMBED_DIM = 100  # dimension of embedding
raw_text = "We are about to study the idea of a computational process. Computational processes are abstract beings that inhabit computers. As they evolve, processes manipulate other abstract things called data. The evolution of a process is directed by a pattern of rules called a program. People create programs to direct processes. In effect, we conjure the spirits of the computer with our spells.".split(
    ' ')

vocab = set(raw_text)
# print("vocab",vocab)
# print("vocab len",len(vocab))

word_to_idx = {word: i for i, word in enumerate(vocab)}  # dict

# print("word_to_idx",word_to_idx)

data = []
for i in range(CONTEXT_SIZE, len(raw_text) - CONTEXT_SIZE):
    context = [raw_text[i - 2], raw_text[i - 1], raw_text[i + 1], raw_text[i + 2]]
    target = raw_text[i]
    data.append((context, target))


# print("data",data) # tuple(list of text before and after, word)

class CBOW(nn.Module):
    def __init__(self, n_word, n_dim, context_size):
        super(CBOW, self).__init__()
        self.embedding = nn.Embedding(n_word, n_dim)
        # before and after is 2 words, 2*2 intotal, mutiply the dimensiton of embedding
        # input is the embedding list of nearby words(2 before and 2 after)
        # output is defined by experience
        self.linear1 = nn.Linear(2 * context_size * n_dim, 128)
        # input is the output of last linear lay, output is the weight of each words
        self.linear2 = nn.Linear(128, n_word)

    def forward(self, x):  # x is the list of index of before and after words, as the format of Variable
        # print("x", x)
        x = self.embedding(x)
        # print("embedding_x", x)
        x = x.view(1, -1)
        x = self.linear1(x)
        x = F.relu(x, inplace=True)
        x = self.linear2(x)
        x = F.log_softmax(x)
        # print("out_x",x)
        return x


model = CBOW(len(word_to_idx), EMBED_DIM, CONTEXT_SIZE)

# print("model",model)

if torch.cuda.is_available():
    model = model.cuda()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=1e-3)

for epoch in range(10):
    print('epoch {}'.format(epoch))
    print('*' * 10)
    running_loss = 0
    for word in data:
        context, target = word
        context = Variable(
            torch.LongTensor([word_to_idx[i] for i in context]))  # convert list of words to list of indexs
        target = Variable(torch.LongTensor([word_to_idx[target]]))
        if torch.cuda.is_available():
            context = context.cuda()
            target = target.cuda()
        # forward
        out = model(context)
        # print("out", out)
        # print("target", target)
        loss = criterion(out, target)  # out is a list of prosibility of each words, target is the index of correct one
        running_loss += loss.data[0]
        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print('loss: {:.6f}'.format(running_loss / len(data)))

# save
# print("embedding",type(model.embedding))
# print("embedding",model.embedding())
embedding = model.embedding.weight.data.numpy()  # array
# print("model.embedding.weight",model.embedding.weight) # Parameter [torch.FloatTensor of size 49x100]
# print("model.embedding.weight.data",model.embedding.weight.data) # FloatTensor [torch.FloatTensor of size 49x100]
# print("embedding",embedding)
embedding_file_path = "embedding_array"
file = open(embedding_file_path, 'w')
file.write('%d %d\n' % (len(embedding), EMBED_DIM))
for w, i in word_to_idx.items():
    e = embedding[i]
    e = ' '.join(map(lambda x: str(x), e))
    file.write('%s %s\n' % (w, e))

# read embedding

f = open(embedding_file_path)
f.readline()
all_embeddings = []
all_words = []
word2id = dict()
for i, line in enumerate(f):
    line = line.strip().split(' ')
    word = line[0]
    embedding = [float(x) for x in line[1:]]
    # assert len(embedding) == EMBED_DIM
    all_embeddings.append(embedding)
    all_words.append(word)
    word2id[word] = i
all_embeddings = numpy.array(all_embeddings)

print("all_embeddings[idea]", all_embeddings[word2id["idea"]])

# Persistance
# import pickle
# pickle.dump(model.embedding, open("embedding.pickle", "w"))

import cPickle as pickle

pickle.dump(model.embedding, open("embedding.pickle", "w"), True)  # press

ebeddings = pickle.load(open("embedding.pickle", "r"))

# print type(ebeddings)
# print(ebeddings.weight[0])

