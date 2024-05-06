import warnings
warnings.filterwarnings("ignore")
warnings.simplefilter(action='ignore', category=FutureWarning)

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
import time
import random
import math
import copy
import sklearn.metrics as metrics
import tensorflow as tf
import numpy as np
import pickle
from scipy.interpolate import splev, splrep


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print('=================================== {} ==================================='.format(device))

def load_data(data_dir, fname, train_rate):
    ir = 3
    before = 2
    after = 2
    # normalize
    scaler = lambda arr: (arr - np.min(arr)) / (np.max(arr) - np.min(arr))
    tm = np.arange(0, (before + 1 + after) * 60, step=1 / float(ir))

    with open(os.path.join(data_dir, fname), 'rb') as f:
        apnea_ecg = pickle.load(f)

    o_train, y_train = apnea_ecg["o_train"], apnea_ecg["y_train"]
    groups_train = apnea_ecg["groups_train"]

    x_train5, x_train3, x_train1 = [], [], []
    for i in range(len(o_train)):
        (rri_tm, rri_signal), (ampl_tm, ampl_siganl) = o_train[i]
        # Curve interpolation
        rri_interp_signal = splev(tm, splrep(rri_tm, scaler(rri_signal), k=3), ext=1)
        ampl_interp_signal = splev(tm, splrep(ampl_tm, scaler(ampl_siganl), k=3), ext=1)
        x_train5.append([rri_interp_signal, ampl_interp_signal])  # 5-minute-long segment
        x_train3.append([rri_interp_signal[180:720], ampl_interp_signal[180:720]])  # 3-minute-long segment
        x_train1.append([rri_interp_signal[360:540], ampl_interp_signal[360:540]])  # 1-minute-long segment


    x_training5,x_training3,x_training1,y_training,groups_training = [],[],[],[],[]
    x_val5,x_val3,x_val1,y_val,groups_val = [],[],[],[],[]

    trainlist = random.sample(range(len(o_train)),int(len(o_train)*train_rate))
    num = [i for i in range(len(o_train))]
    vallist = set(num) - set(trainlist)
    vallist = list(vallist)
    for i in trainlist:
        x_training5.append(x_train5[i])
        x_training3.append(x_train3[i])
        x_training1.append(x_train1[i])
        y_training.append(y_train[i])
        groups_training.append(groups_train[i])
    for i in vallist:
        x_val5.append(x_train5[i])
        x_val3.append(x_train3[i])
        x_val1.append(x_train1[i])
        y_val.append(y_train[i])
        groups_val.append(groups_train[i])

    x_training5 = np.array(x_training5, dtype="float32").transpose((0, 2, 1))
    x_training3 = np.array(x_training3, dtype="float32").transpose((0, 2, 1))
    x_training1 = np.array(x_training1, dtype="float32").transpose((0, 2, 1))
    y_training = np.array(y_training, dtype="float32")
    x_val5 = np.array(x_val5, dtype="float32").transpose((0, 2, 1))
    x_val3 = np.array(x_val3, dtype="float32").transpose((0, 2, 1))
    x_val1 = np.array(x_val1, dtype="float32").transpose((0, 2, 1))
    y_val = np.array(y_val, dtype="float32")

    x_test5,x_test3,x_test1 = [],[],[]
    o_test, y_test = apnea_ecg["o_test"], apnea_ecg["y_test"]
    groups_test = apnea_ecg["groups_test"]
    for i in range(len(o_test)):
        (rri_tm, rri_signal), (ampl_tm, ampl_siganl) = o_test[i]
        # Curve interpolation
        rri_interp_signal = splev(tm, splrep(rri_tm, scaler(rri_signal), k=3), ext=1)
        ampl_interp_signal = splev(tm, splrep(ampl_tm, scaler(ampl_siganl), k=3), ext=1)
        x_test5.append([rri_interp_signal, ampl_interp_signal])
        x_test3.append([rri_interp_signal[180:720], ampl_interp_signal[180:720]])
        x_test1.append([rri_interp_signal[360:540], ampl_interp_signal[360:540]])

    x_test5 = np.array(x_test5, dtype="float32").transpose((0, 2, 1))
    x_test3 = np.array(x_test3, dtype="float32").transpose((0, 2, 1))
    x_test1 = np.array(x_test1, dtype="float32").transpose((0, 2, 1))
    y_test = np.array(y_test, dtype="float32")

    x_training5 = x_training5.transpose(0, 2, 1)
    x_training3 = x_training3.transpose(0, 2, 1)
    x_training1 = x_training1.transpose(0, 2, 1)
    x_val5 = x_val5.transpose(0, 2, 1)
    x_val3 = x_val3.transpose(0, 2, 1)
    x_val1 = x_val1.transpose(0, 2, 1)
    x_test5 = x_test5.transpose(0, 2, 1)
    x_test3 = x_test3.transpose(0, 2, 1)
    x_test1 = x_test1.transpose(0, 2, 1)

    return x_training5, x_training3, x_training1, y_training, groups_training, \
           x_val5, x_val3, x_val1, y_val, groups_val, \
           x_test5, x_test3, x_test1, y_test, groups_test


class RandomDataset(Dataset):
    def __init__(self, X3, X2, X1, Y, Length, Classes):
        self.X3 = X3
        self.X2 = X2
        self.X1 = X1
        self.Y = Y
        self.Length = Length
        self.Classes = Classes

    def __getitem__(self, index):
        classes = self.Classes[index]
        x3 = torch.tensor(self.X3[index, :, :]).float()
        x2 = torch.tensor(self.X2[index, :, :]).float()
        x1 = torch.tensor(self.X1[index, :, :]).float()
        y = torch.tensor(self.Y[index, :]).type(torch.LongTensor).float()
        return x3, x2, x1, y, index, classes

    def __len__(self):
        return self.Length


def expand_mask(mask):
    assert mask.ndim > 2, "Mask must be at least 2-dimensional with seq_length x seq_length"
    if mask.ndim == 3:
        mask = mask.unsqueeze(1)
    while mask.ndim < 4:
        mask = mask.unsqueeze(0)
    return mask

def scaled_dot_product(q, k, v, mask=None):
    d_k = q.size()[-1]
    attn_logits = torch.matmul(q, k.transpose(-2, -1))
    attn_logits = attn_logits / math.sqrt(d_k)
    if mask is not None:
        attn_logits = attn_logits.masked_fill(mask == 0, -9e15)
    attention = F.softmax(attn_logits, dim=-1)
    values = torch.matmul(attention, v)
    return values, attention

class MultiheadAttention(nn.Module):

    def __init__(self, input_dim, embed_dim, num_heads):
        super().__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be 0 modulo number of heads."

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Stack all weight matrices 1...h together for efficiency
        # Note that in many implementations you see "bias=False" which is optional
        self.qkv_proj = nn.Linear(input_dim, 3*embed_dim)
        self.o_proj = nn.Linear(embed_dim, embed_dim)

        self._reset_parameters()

    def _reset_parameters(self):
        # Original Transformer initialization, see PyTorch documentation
        nn.init.xavier_uniform_(self.qkv_proj.weight)
        self.qkv_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.o_proj.weight)
        self.o_proj.bias.data.fill_(0)

    def forward(self, x, mask=None, return_attention=False):
        batch_size, seq_length, _ = x.size()
        if mask is not None:
            mask = expand_mask(mask)
        qkv = self.qkv_proj(x)

        # Separate Q, K, V from linear output
        qkv = qkv.reshape(batch_size, seq_length, self.num_heads, 3*self.head_dim)
        qkv = qkv.permute(0, 2, 1, 3) # [Batch, Head, SeqLen, Dims]
        q, k, v = qkv.chunk(3, dim=-1)
        # print('qkv size:', q.size(), k.size(), v.size())

        # Determine value outputs
        values, attention = scaled_dot_product(q, k, v, mask=mask)
        values = values.permute(0, 2, 1, 3) # [Batch, SeqLen, Head, Dims]
        values = values.reshape(batch_size, seq_length, self.embed_dim)
        o = self.o_proj(values)

        if return_attention:
            return o, attention
        else:
            return o

class MTIF(nn.Module):
    def __init__(self, kernel_size, drop_rate, dilation_rate):
        super(MTIF, self).__init__()
        self.kernel_size = kernel_size
        self.drop_rate = drop_rate
        self.dilation_rate = dilation_rate

        self.layer0 = nn.Sequential(
            nn.Conv1d(in_channels=2, out_channels=64, kernel_size=self.kernel_size, padding='same'),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(p=self.drop_rate)
        )
        self.layer1_1 = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=self.kernel_size, dilation=self.dilation_rate, padding='same'),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(p=self.drop_rate),
        )
        self.layer1_2 = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=self.kernel_size, dilation=self.dilation_rate, padding='same'),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(p=self.drop_rate),
        )
        self.layer1_3 = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=self.kernel_size, dilation=self.dilation_rate, padding='same'),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(p=self.drop_rate),
        )
        self.layer1_4 = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=self.kernel_size, dilation=self.dilation_rate, padding='same'),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(p=self.drop_rate),
        )
        self.layer2_1 = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=self.kernel_size, dilation=self.dilation_rate, padding='same'),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(p=self.drop_rate),
        )
        self.layer2_2 = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=self.kernel_size, dilation=self.dilation_rate, padding='same'),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(p=self.drop_rate),
        )
        self.layer2_3 = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=self.kernel_size, dilation=self.dilation_rate, padding='same'),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(p=self.drop_rate),
        )
        self.layer2_4 = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=self.kernel_size, dilation=self.dilation_rate, padding='same'),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(p=self.drop_rate),
        )

    def forward(self, x):
        x = self.layer0(x)
        x = self.layer1_1(x)
        x = self.layer1_2(x)
        x = self.layer1_3(x)
        x = self.layer1_4(x)
        x = self.layer2_1(x)
        x = self.layer2_2(x)
        x = self.layer2_3(x)
        x = self.layer2_4(x)
        return x

class DAN_MTIF(nn.Module):
    def __init__(self, num_heads, drop_rate, dilation_rate):
        super(DAN_MTIF, self).__init__()
        self.num_heads = num_heads
        self.drop_rate = drop_rate
        self.dilation_rate = dilation_rate

        self.CNN1 = MTIF(kernel_size=3, drop_rate=self.drop_rate, dilation_rate=self.dilation_rate)
        self.CNN2 = MTIF(kernel_size=7, drop_rate=self.drop_rate, dilation_rate=self.dilation_rate)
        self.CNN3 = MTIF(kernel_size=11, drop_rate=self.drop_rate, dilation_rate=self.dilation_rate)

        self.ATT = MultiheadAttention(input_dim=64, embed_dim=64, num_heads=self.num_heads)

        self.Dropout = nn.Dropout(p=0.5)

        self.Classifier = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
        )

    def forward(self, x1, x3, x5):
        x1 = self.CNN1(x1)
        x3 = self.CNN2(x3)
        x5 = self.CNN3(x5)
        x1 = x1.permute(2, 0, 1)
        x3 = x3.permute(2, 0, 1)
        x5 = x5.permute(2, 0, 1)
        x = torch.cat([x1, x3, x5], dim=0)
        attx = self.ATT(x)
        x = x + attx
        x = x.permute(1, 2, 0)
        x = F.avg_pool1d(x, kernel_size=x.size()[2])
        x = x.view(x.shape[0], -1)
        x = self.Dropout(x)
        x = self.Classifier(x)
        return x



num_heads = 2
drop_rate = 0.3
dilation_rate = 2
patience = 2
factor = 0.2
lr = 0.001
batch_size = 32
start_epoch = 0
num_epochs = 100
end_epoch = num_epochs

data_dir = './dataset'
fname = 'apnea-ecg.pkl'
log_dir = './model'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

flag_train = True
flag_test = True
BEST_ACC = False
BEST_LOSS = True



train_rate_set = [0.7]

for train_rate in train_rate_set:

    x_train5, x_train3, x_train1, y_train, groups_train, \
    x_val5, x_val3, x_val1, y_val, groups_val, \
    x_test5, x_test3, x_test1, y_test, groups_test = load_data(data_dir, fname, train_rate)

    # Convert to two categories
    y_train = tf.keras.utils.to_categorical(y_train, num_classes=2)
    y_val = tf.keras.utils.to_categorical(y_val, num_classes=2)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes=2)

    # Load Dataset
    ntrain = y_train.shape[0]
    RandomDataset_train = RandomDataset(x_train5, x_train3, x_train1, y_train, ntrain, groups_train)
    dataloader_train = DataLoader(dataset=RandomDataset_train, batch_size=batch_size, num_workers=0,
                                   shuffle=True)
    nval = y_val.shape[0]
    RandomDataset_valid = RandomDataset(x_val5, x_val3, x_val1, y_val, nval, groups_val)
    dataloader_valid = DataLoader(dataset=RandomDataset_valid, batch_size=batch_size, num_workers=0,
                                   shuffle=True)
    ntest = y_test.shape[0]
    RandomDataset_test = RandomDataset(x_test5, x_test3, x_test1, y_test, ntest, groups_test)
    dataloader_test = DataLoader(dataset=RandomDataset_test, batch_size=batch_size, num_workers=0,
                                   shuffle=False)

    # load model
    model = DAN_MTIF(num_heads=num_heads, drop_rate=drop_rate, dilation_rate=dilation_rate)
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=patience, verbose=True, factor=factor)
    criteria = torch.nn.CrossEntropyLoss()

    # training
    def model_train(model, optimizer, scheduler, num_epochs, criteria):
        since = time.time()
        best_acc = 0.0
        best_loss = 999

        best_model_acc = model.state_dict()
        best_model_loss = model.state_dict()

        for epoch in range(num_epochs):
            print('Epoch {}/{}'.format(epoch+1, num_epochs), '-' * 100)

            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                k = 0
                running_loss = 0.0
                running_pred = []
                running_true = []

                if (phase == 'train'):
                    model.train()
                    dataloader = dataloader_train
                    len_data = len(dataloader)
                    print("time:", time.time() - since)

                    for sample in dataloader:
                        k += 1

                        x_min5, x_min3, x_min1, labels, _, _ = sample
                        x_min5 = x_min5.to(device)
                        x_min3 = x_min3.to(device)
                        x_min1 = x_min1.to(device)
                        labels = labels.to(device)

                        optimizer.zero_grad()

                        with torch.set_grad_enabled(phase == 'train'):
                            outputs = model(x_min1, x_min3, x_min5)
                            loss = criteria(outputs, labels)
                            # backward:
                            loss.backward()
                            optimizer.step()

                            preds = torch.argmax(outputs, dim=1)
                            labels = labels[:, 1]

                            running_loss += loss.item() * x_min1.size(0)

                            preds = preds.cpu().detach().numpy().squeeze()
                            labels = labels.cpu().detach().numpy().squeeze()

                            running_pred += list(preds)
                            running_true += list(labels)


                    epoch_loss = (running_loss * 1.0) / len_data
                    epoch_acc = accuracy_score(running_true, running_pred)


                if (phase == 'val'):
                    model.eval()
                    dataloader = dataloader_valid

                    for sample in dataloader:
                        k += 1

                        x_min5, x_min3, x_min1, labels, _, _ = sample
                        x_min5 = x_min5.to(device)
                        x_min3 = x_min3.to(device)
                        x_min1 = x_min1.to(device)
                        labels = labels.to(device)

                        with torch.set_grad_enabled(False):
                            outputs = model(x_min1, x_min3, x_min5)
                            loss = criteria(outputs, labels)

                            preds = torch.argmax(outputs, dim=1)
                            labels = labels[:, 1]

                            running_loss += loss.item() * x_min1.size(0)

                            preds = preds.cpu().detach().numpy().squeeze()
                            labels = labels.cpu().detach().numpy().squeeze()

                            running_pred += list(preds)
                            running_true += list(labels)


                    epoch_loss = (running_loss * 1.0) / len_data
                    epoch_acc = accuracy_score(running_true, running_pred)

                    scheduler.step(epoch_loss)


                # deep copy the model
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_acc = copy.deepcopy(model.state_dict())
                    torch.save(best_model_acc, log_dir + '/best_acc.pth')

                if phase == 'val' and epoch_loss < best_loss:
                    best_loss = epoch_loss
                    best_model_loss = copy.deepcopy(model.state_dict())
                    torch.save(best_model_loss, log_dir + '/best_loss.pth')

                print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            print()

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        print('Best val Loss: {:4f}'.format(best_loss))
        print('Best val Acc: {:4f}'.format(best_acc))
        print()

        return model


    def model_test(model):
        for phase in ['test']:
            model.eval()
            dataloader = dataloader_test

            # print(phase)
            print('model evaluation with {} dataset'.format(phase), '-' * 100)

            y_true_all = []
            y_prob_all = []
            y_pred_all = []
            class_list_all = []

            for sample in dataloader:

                x_min5, x_min3, x_min1, y_true, _, class_list = sample
                x_min5 = x_min5.to(device)
                x_min3 = x_min3.to(device)
                x_min1 = x_min1.to(device)
                y_true = y_true.to(device)

                # outputs
                outputs = model(x_min1, x_min3, x_min5)
                m = nn.Softmax()
                outputs = m(outputs)

                y_prob = outputs[:, 1]
                y_pred = torch.argmax(outputs, dim=1)
                y_true = y_true[:, 1]

                y_prob = y_prob.cpu().detach().numpy().squeeze()
                y_pred = y_pred.cpu().detach().numpy().squeeze()
                y_true = y_true.cpu().detach().numpy().squeeze()

                y_prob_all += list(y_prob)
                y_true_all += list(y_true)
                y_pred_all += list(y_pred)
                class_list_all += class_list


            Acc = accuracy_score(y_true_all, y_pred_all)
            fpr, tpr, threshold = metrics.roc_curve(y_true_all, y_prob_all)
            AUC = metrics.auc(fpr, tpr)
            f1 = f1_score(y_true_all, y_pred_all, average='binary')
            confusion = confusion_matrix(y_true_all, y_pred_all, labels=(1, 0))

            print("ACC: {}, AUC: {}, F1: {}".format(Acc, AUC, f1))
            print("Confusion Matrix: \n")
            print(confusion)
            print()

        return model


    if flag_train:
        model_ft = model_train(model, optimizer, scheduler, num_epochs, criteria)
    if flag_test and BEST_ACC:
        model.load_state_dict(torch.load(log_dir + '/best_acc.pth'))
        evaluate_result = model_test(model)
    if flag_test and BEST_LOSS:
        model.load_state_dict(torch.load(log_dir + '/best_loss.pth'))
        evaluate_result = model_test(model)



