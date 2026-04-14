from datetime import datetime
import shutil
import os
import numpy as np

def get_time():
    time = str(datetime.now())
    time = time.replace('-','_')
    time = time.replace(' ','_')
    time = time.replace(':','_')[:19]
    return time

def save_code(save_path):
    files = os.listdir('./')
    files = [file for file in files if file[-3:] == ".py"]
    for file in files:
        shutil.copy(file, save_path + file)
        print(file + ' is copied into saving directory!')

class lr_decay_calculator():
    def __init__(self, data_set_length, training_length, lr_decay_length, decay_offset, secondary_decay_offset):
        self.data_set_length = data_set_length
        self.training_length = training_length
        self.lr_decay_length = lr_decay_length
        self.decay_offset = decay_offset
        self.secondary_decay_offset = secondary_decay_offset
        self.iterator = int(0)
        
    def get_lrd(self, iteration, epoch):
        
        i = self.iterator
        
        check = epoch * self.data_set_length + iteration
        
        if i != check:
            print("LR DECAY CALCULATION ERROR")
 
        if i < self.lr_decay_length:
            lam = 3.0
            mmin = np.exp(- lam)
            mmax = 1 - mmin
            lr_decay = (((np.exp(- lam * (i / self.lr_decay_length))) - mmin) / mmax) * (1 - self.decay_offset) + self.decay_offset
        elif i >= self.lr_decay_length:
            lr_decay = self.decay_offset - ((i - self.lr_decay_length) / (self.training_length - self.lr_decay_length)) * (self.decay_offset - self.secondary_decay_offset)
        else:
            lr_decay = self.secondary_decay_offset
            
        return lr_decay
    
    def update_iterator(self):
        self.iterator += 1


