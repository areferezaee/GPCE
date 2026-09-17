import torch
import torch.utils.data as data_utl

import numpy as np
import random

import os
#import lintel

import json

import torch


class Cardio_LDR(data_utl.Dataset):

    def __init__(self, split_file, root):
        with open(split_file, 'r') as f:
            self.data = f.readlines()
        
        self.maindata = []
        self.feats = []
        self.all_feats = []
        self.all_normgrads = []
        self.normgrads = []
        self.all_gts = []
        self.gts = []
        self.all_feats2 = []
        for o in range(len(self.data)):
            line = self.data[o].strip()
            #print(line, o)
            if not line:
                continue
            parts = line.split()
            entry = parts[0]
            # Determine feature file names for each dataset type
            feature_name = 'z_'+entry +'.pt'
            #ablation = 'QFormer_'+entry
            
            #normalized_grad = 'normalized_grad_z_'+entry
            gt_name = 'z_gt_'+entry+'.pt'
            #feature_names = [entry + '.avi.pt'] if not entry.endswith('.avi.pt') else [entry]
            #print(feature_name, entry)
            #for feat_name in feature_names:
            feat_path = os.path.join(root, feature_name)
            
            #feat_path2 = os.path.join(secondroot, ablation)
            #normalized_grad_path = os.path.join(root, normalized_grad)
            gt_path = os.path.join(root, gt_name)
            #print(feat_path)
            if not os.path.exists(feat_path):
                    print(feat_path)
                    continue
            totalfeat = torch.load(feat_path)
            #totalfeat2 = torch.load(feat_path2)
            #totalnormgrads = torch.load(normalized_grad_path)
            totalgts = torch.load(gt_path)
            self.all_feats.append(totalfeat)
            #self.all_feats2.append(totalfeat2)
            #self.all_normgrads.append(totalnormgrads)
            self.all_gts.append(totalgts)
            self.maindata.append(line)
            self.feats.append(feature_name)
            #self.normgrads.append(normalized_grad)
            self.gts.append(gt_name)
            #print(line, totalfeat.size())
        
       
        
        self.split_file = split_file
        self.root = root
        
        
        
    def __getitem__(self, index):
        eta = 1.0
        feat = self.feats[index] 
        #normalized_gradname = self.normgrads[index]
        gts = self.gts[index]

        #normalized_gradz= self.all_normgrads[index]
        gtz = self.all_gts[index]
        #self.delta_z = get_delta_Z(normalized_gradz, compress='False')
        #self.delta_z = -eta * normalized_gradz
        
        #print(self.delta_z.size())
        dff = self.all_feats[index]
        self.delta_z = gtz - dff
        #dff2 = self.all_feats2[index]
        #print(dff.shape, dff2.shape)
        #dff = torch.cat((dff, dff2), dim = 2)
        
        return dff, self.delta_z, index, feat
        
    def __len__(self):
        return len(self.maindata)#.keys())

    


    
