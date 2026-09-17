
import torch.nn.functional as F
import torch

import numpy as np

train = 'data/Train.txt'
val = 'data/Val.txt'
test = 'data/Test.txt'
root = 'data/cardiofeats/'

batch_size = 1

device = torch.device('cpu')#('cuda')

from representation_eng.cardio_loader import Cardio_LDR
dataset_tr = Cardio_LDR(train, root)
dl = torch.utils.data.DataLoader(dataset_tr, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)
    
dataset = Cardio_LDR(val, root)
vdl = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)

dataset_ts = Cardio_LDR(test, root)
tst = torch.utils.data.DataLoader(dataset_ts, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)
dataloader = {'train':dl, 'val':vdl, 'test':tst}



def get_representation(phase):
    k = 0 
    eps= 1e-8
    
    for feat, output, index, name in dataloader[phase]:
            
            vid_feats = feat[0].to(device)
            
            #print('vid fetas', vid_feats.size())

            f = vid_feats
           
            out = output.to(device)
            index = index.to(device)

            X = torch.cat((X, f), dim=0) if k > 0 else f
            Y = torch.cat((Y, out[0]), dim=0) if k > 0 else out[0]
            X_idx = torch.cat((X_idx, index), dim=0) if k > 0 else index
            
            k += 1
    print(X.size(), Y.size(), X_idx.size())
    if phase == 'test':
            
            print(name)
            return X, Y, X_idx, name
    else:
        return X, Y, X_idx

