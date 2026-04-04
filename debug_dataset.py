import os
import torch
import attributes
from dataset import gns_dataset

device = torch.device('cpu')
data_parameters_pack = attributes.attribute(device)

test_set = gns_dataset(data_parameters_pack, [], device)
full_path = test_set.ds_path + test_set.test_folder
print("Target path:", full_path)
print("Files in target:", os.listdir(full_path))

test_set.load_dataset(full_path)
print("Length of read dataset:", len(test_set.dataset))
print("Iterator shape:", test_set.iterator.shape)
