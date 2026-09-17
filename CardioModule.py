#Arefe-MAH
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader#, non_deterministic
from tqdm import trange
from utils import *
from torchvision import transforms
from Error_Estimator.err_est import ModelErrorEstimator
from io import BytesIO
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import gpytorch
import time
from sklearn.metrics import (
accuracy_score,
precision_score,
recall_score,
f1_score,
roc_auc_score,
confusion_matrix,
classification_report
)

from sklearn.preprocessing import label_binarize
from argparse import ArgumentParser
import torch
import time


from representation_eng.cardio import get_representation


# =========================
# CardioModule
# =========================

class CardioModule:
    def __init__(self, args):
        
        self.args = args
        set_seed(self.args.seed)
        
        self.n_token = 1
        self.device = torch.device('cpu')#('cuda')
        self.batch_size = args.batch_size
        self.status= 'train'
        self.leng = 3072
        self.div = 1
        if args.Only_TEST == 'Y':
           self.experiment = True
        else:
           self.experiment = False
        
        print(f"Using device: {self.device}")

        #======= Data =======
        

        if self.args.dataset =='Cardio':


          self.X_train, self.Y_train, self.Xtrain_idx = get_representation('train')
          self.X_train = self.X_train.reshape([self.X_train.size(0),self.X_train.size(1)*self.X_train.size(2)])
          self.Y_train = self.Y_train.reshape([self.Y_train.size(0),self.Y_train.size(1)*self.Y_train.size(2)])
          self.Y_train= self.Y_train.to(torch.float32)
          
          self.X_val, self.Y_val, self.Xval_idx = get_representation('val')
          self.X_val = self.X_val.reshape([self.X_val.size(0),self.X_val.size(1)*self.X_val.size(2)])
          self.Y_val = self.Y_val.reshape([self.Y_val.size(0),self.Y_val.size(1)*self.Y_val.size(2)])
          self.Y_val= self.Y_val.to(torch.float32)
          
          self.X_test, self.Y_test, self.Xtest_idx, self.name = get_representation('test')
          self.ztest = self.X_test
          self.X_test = self.X_test.reshape([self.X_test.size(0),self.X_test.size(1)*self.X_test.size(2)])
          self.Y_test = self.Y_test.reshape([self.Y_test.size(0),self.Y_test.size(1)*self.Y_test.size(2)])
          self.Y_test= self.Y_test.to(torch.float32)
          
          self.Y_test_major = self.Y_test
          
          self.args.feature_length = torch.tensor([self.X_train.size(1)])#, self.X_train.size(2)])
          print('for Xbar',self.X_train.size(1))#*self.X_train.size(2))
        self.allGT=[]
        self.allpreds=[]
        #======= Model =======
        # build initial model
        
        self.ERR_Estimator = ModelErrorEstimator(self.args, self.device, pretrained=False)
        self.ERR_Estimator.to(self.device)
        # ==========
        # Initialization Train
        # ==========
        self.epochs = 20
        self.writer = SummaryWriter()
        self.start_time = time.time()
        
        self.params_optimizer = optim.Adam(self.ERR_Estimator.parameters(), lr=1e-3)
    # =========================
    # train-val-test 
    # =========================
    
    # =========================
    # Checkpoint
    # =========================
    def save_checkpoint(self, epoch, path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

        # Store command-line/model settings in a simple serializable dict.
        saved_args = dict(vars(self.args))
        saved_args.pop("feature_length", None)

        checkpoint = {
            "epoch": epoch,

            # IMPORTANT:
            # ModelErrorEstimator creates some GP/VI submodules dynamically.
            # Saving the whole model preserves those registered modules so a
            # separate test process can restore the exact trained structure.
            "model": self.ERR_Estimator,

            # Keep state_dict too, for inspection/backward compatibility.
            "model_state_dict": self.ERR_Estimator.state_dict(),

            "optimizer_state_dict": (
                self.params_optimizer.state_dict()
                if self.params_optimizer is not None
                else None
            ),
            "feature_length": int(self.args.feature_length[0]),
            "args": saved_args,
        }

        torch.save(checkpoint, path+'cardio_err_estimator'+str(epoch)+'.pth')
        print(f"Checkpoint saved to: {path}")
    def train(self, strategy):
       
       #max_grad = 100
       self.ERR_Estimator.train()
       total_loss = 0.0
       if strategy == 'joint':
          self.params_optimizer.zero_grad()
       with torch.autograd.detect_anomaly():

             if strategy == 'hierarchical':
                self.params_optimizer.zero_grad()
             loss = self.ERR_Estimator(self.X_train, self.Y_train.squeeze(1), self.Xtrain_idx, mode='train', tk_idx=0)
             
             if strategy == 'hierarchical':
                loss.backward(retain_graph=True)
                self.params_optimizer.step()
             
             total_loss += loss
       # optimize GP hyper-parameters 
       tot = self.n_token*self.div
       tot_loss = total_loss/tot
       if strategy == 'joint':
         
         tot_loss.backward(retain_graph=True )
         self.params_optimizer.step()
       
       return tot_loss
       

    def val(self, strategy):
      self.ERR_Estimator.eval()
      total_loss1 = 0.0
      total_loss2 = 0.0
      total_loss3 = 0.0
      
          
      loss1, loss2, loss3, mu = self.ERR_Estimator(self.X_val, self.Y_val, self.Xval_idx, mode='val', tk_idx=0) 
      print(self.Y_val.mean().item(), self.Y_val.std().item())
      print('mu v mean & std:', mu.mean().item(), mu.std().item())
      
      total_loss1 += loss1
      total_loss2 += loss2
      total_loss3 += loss3
      tot = self.n_token*self.div
      torch.save(self.ERR_Estimator,'checkpoints/cardio_err_estimator.pth')
      return total_loss1/tot, total_loss2/tot, total_loss3/tot
      

    def test(self, strategy):

      self.ERR_Estimator.eval()
      total_loss1 = 0.0
      total_loss2 = 0.0
      total_loss3 = 0.0
      
         
      loss1, loss2, loss3, mu = self.ERR_Estimator(self.X_test, self.Y_test, self.Xtest_idx, mode='test', tk_idx=0)
          
      total_loss1 += loss1
      total_loss2 += loss2
      total_loss3 += loss3
          
     
      
      pred_z = mu
      
      print(self.Y_test.mean().item(), self.Y_test.std().item())
      print('mu t mean & std:', pred_z.mean().item(), pred_z.std().item())
      tot = self.n_token*self.div
      return total_loss1/tot, total_loss2/tot, total_loss3/tot, pred_z

    ##################################################metrics comp#####################################################################
    def major_comp_loss(self, pred_z):
        
        
        mainY = self.Y_test_major
        predY = pred_z
        mse = torch.mean((mainY - predY) ** 2)
        mae = torch.mean(torch.abs(mainY - predY))
        print(predY.size())
        predY = predY.reshape([predY.size(1), 96, 3072])
        #alpha2 = predY.std()/self.ztest.std()
        #torch.save(predY, 'Pred_Y_cardio/pred_bar_z_'+ self.name[0])
        #torch.save(predY, 'dlta_bar_Z_cardio/seconddlta_bar_z_'+ self.name[0])
        torch.save(predY, 'dlta_bar_Z_cardio/dlta_bar_z_'+ self.name[0])
        return mse, mae

    # === execute ===
    def execute(self):
      path = args.checkpoints_path
      trained_model = args.checkpoints
      strategy = self.args.train_strategy
      for epoch in range(self.epochs):
        if epoch == (self.epochs - 1):
           self.status= 'test'
           start = time.perf_counter()
           loss1, loss2, loss3, predz = self.test(strategy)
           end = time.perf_counter()
           print(f"Time: {end - start:.6f} seconds")
           mse, mae = self.major_comp_loss(predz) 
           print('TEST:loss1, loss2, loss3', loss1, loss2, loss3)
           print('Pred bar z acc:mse, mae:', mse, mae)
        elif self.experiment:
            self.ERR_Estimator = torch.load(path + trained_model, weights_only=False)
            start = time.perf_counter()
            loss1, loss2, loss3, predz = self.test(strategy)
            end = time.perf_counter()
            print(f"Time: {end - start:.6f} seconds")
            mse, mae = self.major_comp_loss(predz) 
            print('TEST:loss1, loss2, loss3', loss1, loss2, loss3)
            print('Pred bar z acc:mse, mae:', mse, mae)
            break
        else:
  
           self.status= 'train'
           loss = self.train(strategy)
           print('TRAIN:loss:', loss)
           self.status= 'val'
           loss1, loss2, loss3 = self.val(strategy)
           print('VAL:loss1, loss2, loss3', loss1, loss2, loss3)
           #if epoch < self.epochs -2 :
           #  self.save_checkpoint(epoch, path)
               
      

           


# =========================
# Initialization
# =========================
Modulename = '_CardioModule_'
if Modulename == '_CardioModule_':
   print('Module is Running...')
   parser = argparse.ArgumentParser(description='Err_Est GP - trainer')
   parser.add_argument('--script-name', default='Err_Est')
   parser.add_argument('--dataset', type=str, default='ARID')
   parser.add_argument('--optimizer', default='adam')

   parser.add_argument('-lr', default=1e-5, type=float, help='learning rate')
   parser.add_argument('--natural-lr', default=.1, type=float,
                    help='natural GA learning rate. If not using stochastic updates - may use a value of 1.')
   parser.add_argument('--batch_size', type=int, default=1, help='batch size')
   parser.add_argument('--test-batch-size', type=int, default=1, help='test batch size')
   parser.add_argument('--train_strategy', type=str, default='joint')
   parser.add_argument('--checkpoints_path', type=str, default='checkpoints/')
   parser.add_argument('--checkpoints', type=str, default='cardio_err_estimator.pth')
   parser.add_argument('--Only_TEST', type=str, default='N', choices={'Y','N'})


   parser.add_argument('--kernel-function', type=str, default=['LinearKernel', 'LinearKernel', 'LinearKernel', 'LinearKernel', 'LinearKernel', 'LinearKernel'])
   parser.add_argument('--num-inducing-points', type=int, default=11)

   parser.add_argument('--outputscale', type=float, default=1., help='output scale')
   parser.add_argument('--eval-every', type=int, default=1, help='num. epochs between test set eval')
   parser.add_argument('--seed', default=42, type=int, help='random seed')
   parser.add_argument('--num-workers', default=1, type=int, help='num wortkers')
   parser.add_argument('--gpus', type=str, default='0')

   args = parser.parse_args()
   runner = CardioModule(args)
   runner.execute()

