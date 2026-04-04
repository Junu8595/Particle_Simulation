import torch
import numpy as np
import random

import attributes as attr
import dataset
import normalizer
import graph_model as gm

seed = 7777
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


def grad_stats(model):
    total_norm_sq = 0.0
    max_abs_grad = 0.0
    nan_grad = False
    inf_grad = False
    grad_param_count = 0

    for name, param in model.named_parameters():
        if param.grad is None:
            continue

        g = param.grad.detach()
        grad_param_count += 1

        if torch.isnan(g).any():
            nan_grad = True
        if torch.isinf(g).any():
            inf_grad = True

        param_norm = g.norm(2).item()
        total_norm_sq += param_norm ** 2
        max_abs_grad = max(max_abs_grad, g.abs().max().item())

    total_norm = total_norm_sq ** 0.5
    return {
        "grad_param_count": grad_param_count,
        "total_grad_norm": total_norm,
        "max_abs_grad": max_abs_grad,
        "nan_grad": nan_grad,
        "inf_grad": inf_grad,
    }


def main():
    print("===== TRAIN STEP DEBUG START =====")
    print("device:", device)

    # 1. attribute packs
    network_attributes_pack, training_attributes_pack = attr.attribute(device)
    training_parameters_pack, testing_parameters_pack, data_parameters_pack, optimizer_parameters_pack = training_attributes_pack

    node_input_size = network_attributes_pack.node_encoder.input_size
    edge_input_size = network_attributes_pack.edge_encoder.input_size
    target_input_size = network_attributes_pack.decoder.output_size

    print(f"node_encoder input_size = {node_input_size}")
    print(f"edge_encoder input_size = {edge_input_size}")
    print(f"decoder output_size    = {target_input_size}")

    assert node_input_size == 25, f"Unexpected node input size: {node_input_size}"
    assert edge_input_size == 7, f"Unexpected edge input size: {edge_input_size}"
    assert target_input_size == 3, f"Unexpected decoder output size: {target_input_size}"

    # 2. normalizers
    # 현재 dataset.py 기준:
    # node_features  : 25차원
    # edge_features  : 7차원 = [dist, dx_local(3), dv_local(3)]
    # target         : 3차원 acceleration
    node_normalizer = normalizer.online_normalizer(
        'node_normalizer', node_input_size, 10**6, 1e-6, './', device
    )
    edge_normalizer = normalizer.online_normalizer(
        'edge_normalizer', edge_input_size, 10**6, 1e-6, './', device
    )
    target_normalizer = normalizer.online_normalizer(
        'target_normalizer', target_input_size, 10**6, 1e-6, './', device
    )
    normalizer_pack = [node_normalizer, edge_normalizer, target_normalizer]

    # 3. dataset load
    ds = dataset.gns_dataset(data_parameters_pack, normalizer_pack, device)
    ds.load_dataset(ds.ds_path + ds.training_folder)

    print("dataset length:", len(ds))

    # 4. graph model
    graph = gm.Graph(
        network_attributes_pack,
        training_parameters_pack,
        device,
        train_flag=True
    )
    graph.graph_net = graph.graph_net.to(device)
    graph.graph_net.train()

    # 5. one batch load
    idx = 0
    contact_distance = data_parameters_pack.contact_distance
    rotate_flag = False

    print("\n===== LOAD ONE TRAIN BATCH =====")
    data_pack = ds.get_data(idx, contact_distance, rotate_flag)

    print("node_features shape :", data_pack.nodepack.node_features.shape)
    print("edge_features shape :", data_pack.edgepack.edge_features.shape)
    print("edge_a shape        :", data_pack.edgepack.edge_a.shape)
    print("edge_b shape        :", data_pack.edgepack.edge_b.shape)
    print("edge_c shape        :", data_pack.edgepack.edge_c.shape)
    print("target shape        :", data_pack.targetpack.normalized_target.shape)

    assert data_pack.nodepack.node_features.shape[1] == node_input_size, \
        f"Node feature dim mismatch: {data_pack.nodepack.node_features.shape[1]} vs {node_input_size}"
    assert data_pack.edgepack.edge_features.shape[1] == edge_input_size, \
        f"Edge feature dim mismatch: {data_pack.edgepack.edge_features.shape[1]} vs {edge_input_size}"
    assert data_pack.targetpack.normalized_target.shape[1] == target_input_size, \
        f"Target dim mismatch: {data_pack.targetpack.normalized_target.shape[1]} vs {target_input_size}"

    assert data_pack.edgepack.receivers.shape[0] == data_pack.edgepack.senders.shape[0], \
        "Receivers / senders length mismatch"
    assert data_pack.edgepack.edge_a.shape[0] == data_pack.edgepack.edge_features.shape[0], \
        "edge_a count mismatch"
    assert data_pack.edgepack.reverse_edge_idx.shape[0] == data_pack.edgepack.edge_features.shape[0], \
        "reverse_edge_idx count mismatch"

    # 6. parameter snapshot before update
    with torch.no_grad():
        first_param_before = None
        for p in graph.graph_net.parameters():
            first_param_before = p.detach().clone()
            break

    # 7. zero grad
    graph.zero_grad()

    # 8. forward
    print("\n===== FORWARD =====")
    output, loss_info = graph.forward(data_pack, train_flag=True, grid_flag=False)
    print("output shape:", output.shape)
    print("loss info   :", loss_info)
    print("internal loss tensor finite:", torch.isfinite(graph.loss).item())

    assert output.shape[0] == data_pack.nodepack.node_features.shape[0], \
        "Output node count mismatch"
    assert output.shape[1] == target_input_size, \
        "Output feature dim mismatch"
    assert torch.isfinite(graph.loss), "Loss is not finite before backward."
    assert not torch.isnan(output).any(), "NaN detected in forward output."
    assert not torch.isinf(output).any(), "Inf detected in forward output."

    # 9. backward only
    print("\n===== BACKWARD =====")
    graph.loss.backward()

    stats = grad_stats(graph.graph_net)
    print("grad_param_count:", stats["grad_param_count"])
    print("total_grad_norm :", stats["total_grad_norm"])
    print("max_abs_grad    :", stats["max_abs_grad"])
    print("nan_grad        :", stats["nan_grad"])
    print("inf_grad        :", stats["inf_grad"])

    assert stats["grad_param_count"] > 0, "No gradients were produced."
    assert not stats["nan_grad"], "NaN detected in gradients."
    assert not stats["inf_grad"], "Inf detected in gradients."

    # 10. optimizer step manually
    print("\n===== OPTIMIZER STEP =====")
    clipped_grad_norm = torch.nn.utils.clip_grad_norm_(
        graph.graph_net.parameters(),
        max_norm=optimizer_parameters_pack.grad_limit,
        norm_type=2
    )
    print("returned clipped grad norm:", float(clipped_grad_norm))

    graph.graph_optimizer.step()

    # 11. parameter changed?
    with torch.no_grad():
        first_param_after = None
        for p in graph.graph_net.parameters():
            first_param_after = p.detach().clone()
            break

        param_change = (first_param_after - first_param_before).abs().max().item()

    print("max change in first parameter tensor:", param_change)
    assert param_change > 0.0, "Parameters did not update."

    # 12. optional second forward after one step
    print("\n===== SECOND FORWARD AFTER UPDATE =====")
    graph.zero_grad()
    output2, loss_info2 = graph.forward(data_pack, train_flag=True, grid_flag=False)
    print("output2 shape:", output2.shape)
    print("loss info2   :", loss_info2)
    print("second loss finite:", torch.isfinite(graph.loss).item())

    assert torch.isfinite(graph.loss), "Loss is not finite after one optimizer step."
    assert not torch.isnan(output2).any(), "NaN detected in output after update."
    assert not torch.isinf(output2).any(), "Inf detected in output after update."

    print("\n===== TRAIN STEP DEBUG SUCCESS =====")


if __name__ == "__main__":
    main()