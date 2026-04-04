import os
import dataset
import normalizer
import attributes as attr
from dataset import gns_dataset

device = 'cpu'

network_attributes_pack, training_attributes_pack = attr.attribute(device)
training_parameters_pack, testing_parameters_pack, data_parameters_pack, optimizer_parameters_pack = training_attributes_pack

node_input_size = network_attributes_pack.node_encoder.input_size
edge_input_size = network_attributes_pack.edge_encoder.input_size
target_input_size = network_attributes_pack.decoder.output_size

node_normalizer = normalizer.online_normalizer('node_normalizer', node_input_size, device=device)
edge_normalizer = normalizer.online_normalizer('edge_normalizer', edge_input_size, device=device)
target_normalizer = normalizer.online_normalizer('target_normalizer', target_input_size, device=device)

normalizer_pack = [node_normalizer, edge_normalizer, target_normalizer]

ds = dataset.gns_dataset(data_parameters_pack, normalizer_pack, device)

local_data_path = r"C:\Projects\GNN_Particle_Sim_1\training_40_"

print("loading from:", local_data_path)
print("files:", os.listdir(local_data_path)[:10])

ds.load_dataset(local_data_path + "\\")

data_pack = ds.get_data(0, ds.contact_distance, False)

print("edge_a shape:", data_pack.edgepack.edge_a.shape)
print("reverse_edge_idx shape:", data_pack.edgepack.reverse_edge_idx.shape)
print("pairwise edges:", data_pack.edgepack.pairwise_mask.sum().item())