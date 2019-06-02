{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "1.0.1.post2\n"
     ]
    }
   ],
   "source": [
    "from __future__ import absolute_import\n",
    "from __future__ import division\n",
    "from __future__ import print_function\n",
    "\n",
    "import os\n",
    "os.environ['CUDA_LAUNCH_BLOCKING'] = \"1\"\n",
    "\n",
    "import argparse\n",
    "import json\n",
    "import random\n",
    "import time\n",
    "\n",
    "import tqdm\n",
    "import pickle\n",
    "from torchvision import datasets, transforms, models\n",
    "import loader\n",
    "import models\n",
    "import numpy as np\n",
    "\n",
    "import torch\n",
    "import torch.functional as F\n",
    "import torch.nn as nn\n",
    "import torch.optim as optim\n",
    "from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence\n",
    "\n",
    "import pandas as pd\n",
    "from sklearn.model_selection import train_test_split\n",
    "import numpy as np\n",
    "import unicodedata\n",
    "import re\n",
    "import datetime as dt\n",
    "print(torch.__version__)\n",
    "import shutil\n",
    "#import convlstm"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "print(str(dt.datetime.now())+\" Process Started\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Set random seeds\n",
    "random.seed(7)\n",
    "torch.manual_seed(7)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "def findcp(val):\n",
    "    if val == 'e':\n",
    "        ecp = []\n",
    "        for i in os.listdir('/scratch/skp454/AST/MainDir/models/'):\n",
    "            if 'encoder.pth' in i:\n",
    "                ecp.append(int(i.split('_')[0]))\n",
    "        return str(max(ecp)+1)\n",
    "    if val == 'd':\n",
    "        dcp = []\n",
    "        for i in os.listdir('/scratch/skp454/AST/MainDir/models/'):\n",
    "            if 'decoder.pth' in i:\n",
    "                dcp.append(int(i.split('_')[0]))\n",
    "        return str(max(dcp)+1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "def print_output(out_path, output_list):\n",
    "    \"\"\"\n",
    "    Function that creates an output file with all of the output lines in\n",
    "    the output list.\n",
    "    Args:\n",
    "        out_path: path to write the output file.\n",
    "        output_list: list containing all the output items \n",
    "    \"\"\"\n",
    "    output_file = open(out_path,\"a+\")\n",
    "    output_file.write(\"\\n\\n\")\n",
    "    output_file.write(\"\\n\".join(output_list))\n",
    "    output_file.close()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "def save_checkpoint(state, is_best, mtype, filename='checkpoint.pth.tar'):\n",
    "    torch.save(state, filename)\n",
    "    if is_best:\n",
    "        shutil.copyfile(filename, mtype+'_best.pth.tar')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# set device\n",
    "device = torch.device(\"cuda\" if torch.cuda.is_available() else \"cpu\")\n",
    "print(device)\n",
    "#device = \"cpu\""
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "with (open('/scratch/skp454/AST/MainDir/data/'+'dictionary.pkl', \"rb\")) as openfile:\n",
    "    dictionary = pickle.load(openfile, encoding = 'latin1')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [],
   "source": [
    "dev_file = '/scratch/skp454/AST/MainDir/data/test_set.pkl'\n",
    "train_file = '/scratch/skp454/AST/MainDir/data/train_set.pkl'\n",
    "#train_file = '/scratch/skp454/AST/MainDir/data/test_set.pkl'"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "CPU times: user 9 µs, sys: 0 ns, total: 9 µs\n",
      "Wall time: 18.1 µs\n"
     ]
    }
   ],
   "source": [
    "BATCH_SIZE = 16\n",
    "units = 256\n",
    "embedding_dim = 256"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "print(str(dt.datetime.now())+\" Loading Data\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "hello\n",
      "CPU times: user 3.82 s, sys: 13.2 s, total: 17 s\n",
      "Wall time: 51 s\n"
     ]
    }
   ],
   "source": [
    "# Loaders\n",
    "train_ldr = loader.make_loader(train_file, BATCH_SIZE)\n",
    "dev_ldr = loader.make_loader(dev_file, BATCH_SIZE)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "print(str(dt.datetime.now())+\" Data Loading Complete\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "def trans_input(inputs):\n",
    "    inputs = np.array(inputs)\n",
    "    inputs = np.swapaxes(inputs,1,3)\n",
    "    inputs = np.swapaxes(inputs,2,3)\n",
    "    return torch.Tensor(inputs)\n",
    "\n",
    "def trans_label(labels):\n",
    "    labels = np.array(labels)\n",
    "    #labels = np.swapaxes(labels,1,2)\n",
    "    return torch.Tensor(labels)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "class Encoder(nn.Module):\n",
    "    def __init__(self, batch_sz, enc_units):\n",
    "        super(Encoder, self).__init__()\n",
    "        self.batch_sz = batch_sz\n",
    "        self.enc_units = enc_units\n",
    "        self.feature = nn.Sequential(\n",
    "            nn.Conv2d(3, 32, kernel_size=3, stride=(1,2), padding=0),\n",
    "            nn.BatchNorm2d(32),\n",
    "            nn.Conv2d(32, 64, kernel_size=3, stride=(1,2), padding=0),\n",
    "            nn.BatchNorm2d(64))\n",
    "        self.recurrent = nn.Sequential(\n",
    "            nn.LSTM(4864, self.enc_units, 3, batch_first=True,  bidirectional =False, dropout = 0.5))\n",
    "        \n",
    "    def forward(self, x, device):\n",
    "        x = self.feature(x) \n",
    "        self.hidden = self.initialize_hidden_state(device)\n",
    "        x = x.permute(0,3,1,2)\n",
    "        x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3])\n",
    "        output, self.hidden = self.recurrent(x)\n",
    "        return output, self.hidden\n",
    "\n",
    "    def initialize_hidden_state(self, device):\n",
    "        return torch.zeros((1, self.batch_sz, self.enc_units)).to(device)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# encoder = Encoder(BATCH_SIZE, units)\n",
    "# encoder.to(device)\n",
    "# enc_output, [enc_hidden, enc_cell] = encoder(inputs.to(device), device)\n",
    "# print(enc_output.size())\n",
    "# print(enc_hidden.size())# max_length, batch_size, enc_units"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "class Decoder(nn.Module):\n",
    "    def __init__(self, vocab_size, dec_units, enc_units, batch_sz, embedding_dim):\n",
    "        super(Decoder, self).__init__()\n",
    "        self.vocab_size = vocab_size\n",
    "        self.embedding_dim = embedding_dim\n",
    "        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim)\n",
    "        self.batch_sz = batch_sz\n",
    "        self.dec_units = dec_units\n",
    "        self.enc_units = enc_units\n",
    "        self.recurrent = nn.Sequential(\n",
    "            nn.LSTM(512, self.enc_units, 3, batch_first=True,  bidirectional =False, dropout = 0.5))\n",
    "        self.fc = nn.Linear(self.enc_units, self.vocab_size)\n",
    "        \n",
    "        # used for attention\n",
    "        self.W1 = nn.Linear(self.enc_units, self.dec_units)\n",
    "        self.W2 = nn.Linear(self.enc_units, self.dec_units)\n",
    "        self.V = nn.Linear(self.enc_units, 1)\n",
    "    \n",
    "    def forward(self, x, hidden, enc_output):\n",
    "        \n",
    "        hidden_with_time_axis = hidden.permute(1, 0, 2)[:,2,:].unsqueeze(1)\n",
    "        score = torch.tanh(self.W1(enc_output) + self.W2(hidden_with_time_axis))\n",
    "        \n",
    "        #score = torch.tanh(self.W2(hidden_with_time_axis) + self.W1(enc_output))\n",
    "        # attention_weights shape == (batch_size, max_length, 1)\n",
    "        # we get 1 at the last axis because we are applying score to self.V\n",
    "        attention_weights = torch.softmax(self.V(score), dim=1)\n",
    "        \n",
    "        # context_vector shape after sum == (batch_size, hidden_size)\n",
    "        context_vector = attention_weights * enc_output\n",
    "        context_vector = torch.sum(context_vector, dim=1)\n",
    "        \n",
    "        # x shape after passing through embedding == (batch_size, 1, embedding_dim)\n",
    "        x = x.type(torch.cuda.LongTensor)\n",
    "        x = x.unsqueeze(1)\n",
    "        x = x.to(device)\n",
    "        x = self.embedding(x)\n",
    "\n",
    "        # x shape after concatenation == (batch_size, 1, embedding_dim + hidden_size)\n",
    "        #x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)\n",
    "        # ? Looks like attention vector in diagram of source\n",
    "        x = torch.cat((context_vector.unsqueeze(1), x), -1)\n",
    "        \n",
    "        # passing the concatenated vector to the GRU\n",
    "        # output: (batch_size, 1, hidden_size)\n",
    "        output, [state, cell] = self.recurrent(x)\n",
    "        \n",
    "        \n",
    "        # output shape == (batch_size * 1, hidden_size)\n",
    "        output =  output.view(-1, output.size(2))\n",
    "        \n",
    "        # output shape == (batch_size * 1, vocab)\n",
    "        x = self.fc(output)\n",
    "        #x = torch.softmax(x, dim=1)\n",
    "        return x, state, attention_weights\n",
    "    \n",
    "    def initialize_hidden_state(self):\n",
    "        return torch.zeros((1, self.batch_sz, self.dec_units))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "print(str(dt.datetime.now())+\" Training Begins\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# decoder = Decoder(len(dictionary.keys()), units, units, BATCH_SIZE, embedding_dim)\n",
    "# decoder = decoder.to(device)\n",
    "\n",
    "# #print(enc_hidden.squeeze(0).shape)\n",
    "# dec_hidden = enc_hidden#.squeeze(0)\n",
    "# dec_input = labels[:,0]\n",
    "# print(\"Decoder Input: \", dec_input.shape)\n",
    "# print(\"--------\")\n",
    "\n",
    "# for t in range(1, labels.size(1)):\n",
    "#     # enc_hidden: 1, batch_size, enc_units\n",
    "#     # output: max_length, batch_size, enc_units\n",
    "#     predictions, dec_hidden, _ = decoder(dec_input.to(device), \n",
    "#                                          dec_hidden.to(device), \n",
    "#                                          enc_output.to(device))\n",
    "\n",
    "\n",
    "#     print(\"Prediction: \", predictions.shape)\n",
    "#     print(\"Decoder Hidden: \", dec_hidden.shape)\n",
    "    \n",
    "#     #loss += loss_function(y[:, t].to(device), predictions.to(device))\n",
    "    \n",
    "#     dec_input = labels[:,t]\n",
    "#     print(dec_input.shape)\n",
    "#     break"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "criterion = nn.CrossEntropyLoss()\n",
    "\n",
    "def loss_function(real, pred):\n",
    "    \"\"\" Only consider non-zero inputs in the loss; mask needed \"\"\"\n",
    "    #mask = 1 - np.equal(real, 0) # assign 0 to all above 0 and 1 to all 0s\n",
    "    #print(mask)\n",
    "    mask = real.ge(1).type(torch.cuda.FloatTensor)\n",
    "    real = real.type(torch.cuda.LongTensor)\n",
    "#     mask = real.ge(1).type(torch.FloatTensor)\n",
    "#     real = real.type(torch.LongTensor)\n",
    "    loss_ = criterion(pred, real) * mask \n",
    "    \n",
    "    return torch.mean(loss_)\n",
    "    "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "## TODO: Combine the encoder and decoder into one class\n",
    "encoder = Encoder(BATCH_SIZE, units)\n",
    "decoder = Decoder(len(dictionary.keys())+1, units, units, BATCH_SIZE, embedding_dim)\n",
    "\n",
    "#encoder = nn.DataParallel(encoder)\n",
    "#decoder = nn.DataParallel(decoder)\n",
    "\n",
    "encoder.to(device)\n",
    "decoder.to(device)\n",
    "\n",
    "optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), \n",
    "                       lr=0.001)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "scrolled": false
   },
   "outputs": [],
   "source": [
    "EPOCHS = 30\n",
    "\n",
    "tloss = 10 \n",
    "\n",
    "for epoch in range(EPOCHS):\n",
    "    \n",
    "    #### Train on train set\n",
    "    start = time.time()\n",
    "    \n",
    "    encoder.train()\n",
    "    decoder.train()\n",
    "    \n",
    "    total_loss = 0\n",
    "    batch = 1\n",
    "    for inputs, labels in train_ldr:\n",
    "        inputs, labels = trans_input(inputs).to(device), trans_label(labels).to(device)\n",
    "        loss = 0\n",
    "        enc_output, [enc_hidden, enc_cell] = encoder(inputs.to(device), device)\n",
    "        del inputs\n",
    "        del enc_cell\n",
    "        dec_hidden = enc_hidden\n",
    "        del enc_hidden\n",
    "        dec_input = labels[:,0]\n",
    "        \n",
    "        for t in range(1, labels.size(1)):\n",
    "            predictions, dec_hidden, _ = decoder(dec_input.to(device), \n",
    "                                                 dec_hidden.to(device), \n",
    "                                                 enc_output.to(device))\n",
    "            del dec_input\n",
    "            loss += loss_function(labels[:,t], predictions.to(device))\n",
    "            dec_input = labels[:,t]\n",
    "            \n",
    "        batch_loss = (loss / int(labels.size(1)))\n",
    "        total_loss += batch_loss\n",
    "        \n",
    "        optimizer.zero_grad()\n",
    "        \n",
    "        loss.backward()\n",
    "\n",
    "        ### UPDATE MODEL PARAMETERS\n",
    "        optimizer.step()\n",
    "        \n",
    "        if batch % 5 == 0:\n",
    "            print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1,\n",
    "                                                         batch,\n",
    "                                                         batch_loss.detach().item()))\n",
    "        batch += 1\n",
    "       \n",
    "    del labels\n",
    "    del dec_input\n",
    "    del dec_hidden\n",
    "    \n",
    "    torch.cuda.empty_cache()\n",
    "    \n",
    "    ### Evaluation on dev_set\n",
    "    with torch.no_grad():\n",
    "        encoder.eval()\n",
    "        decoder.eval()\n",
    "        dev_total_loss = 0\n",
    "        dev_batch = 0\n",
    "        for inputs, labels in dev_ldr:\n",
    "            inputs, labels = trans_input(inputs).to(device), trans_label(labels).to(device)\n",
    "            loss = 0\n",
    "            enc_output, [enc_hidden, enc_cell] = encoder(inputs.to(device), device)\n",
    "            del inputs\n",
    "            del enc_cell\n",
    "            dec_hidden = enc_hidden\n",
    "            del enc_hidden\n",
    "            dec_input = labels[:,0]\n",
    "\n",
    "            for t in range(1, labels.size(1)):\n",
    "                predictions, dec_hidden, _ = decoder(dec_input.to(device), \n",
    "                                                     dec_hidden.to(device), \n",
    "                                                     enc_output.to(device))\n",
    "                del dec_input\n",
    "                loss += loss_function(labels[:,t], predictions.to(device))\n",
    "                dec_input = labels[:,t]\n",
    "\n",
    "            dev_batch_loss = (loss / int(labels.size(1)))\n",
    "            dev_total_loss += dev_batch_loss\n",
    "            if dev_batch % 5 == 0:\n",
    "                print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1,\n",
    "                                                         dev_batch,\n",
    "                                                         dev_batch_loss.detach().item()))\n",
    "            dev_batch+=1\n",
    "            \n",
    "\n",
    "        lis=[]\n",
    "        lis.append('Time{}:'.format(dt.datetime.now()))\n",
    "        lis.append('Epoch {} Train Loss {:.4f} Dev Loss {:.4f}'.format(epoch + 1,\n",
    "                   total_loss / len(train_ldr), dev_total_loss / len(dev_ldr)))\n",
    "        lis.append('Time taken for 1 epoch {} sec'.format(time.time() - start))\n",
    "\n",
    "\n",
    "        print_output('logs.txt', lis)\n",
    "        print(lis)\n",
    "\n",
    "        is_best = tloss > (dev_total_loss / len(dev_ldr))\n",
    "\n",
    "        if is_best:\n",
    "            tloss = (dev_total_loss / len(dev_ldr))\n",
    "\n",
    "    if epoch % 1 == 0:\n",
    "        # save encoder\n",
    "        save_checkpoint({\n",
    "                             'epoch': epoch + 1,\n",
    "                             'arch': 'encoder',\n",
    "                             'state_dict': encoder.state_dict(),\n",
    "                             'best_acc': tloss,\n",
    "                             'optimizer' : optimizer.state_dict(),\n",
    "                         }, is_best, mtype ='encoder', filename= \\\n",
    "            '/scratch/skp454/AST/MainDir/models/'+findcp('e')+'_encoder.pth.tar')\n",
    "\n",
    "        # save decoder\n",
    "        save_checkpoint({\n",
    "                             'epoch': epoch + 1,\n",
    "                             'arch': 'decoder',\n",
    "                             'state_dict': decoder.state_dict(),\n",
    "                             'best_acc': tloss,\n",
    "                             'optimizer' : optimizer.state_dict(),\n",
    "                         }, is_best, mtype ='decoder', filename= \\\n",
    "            '/scratch/skp454/AST/MainDir/models/'+findcp('d')+'_decoder.pth.tar')"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.6.3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
