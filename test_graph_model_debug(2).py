import torch
import normalizer

from attributes import attribute
from dataset import gns_dataset
from graph_model import Graph


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device = {device}")

    network_attributes, training_attributes = attribute(device)

    trainingparameterpack = training_attributes[0]
    dataparameterpack = training_attributes[2]
    optimizerparameterpack = training_attributes[3]

    print("[INFO] edge_encoder input_size =", network_attributes.edge_encoder.input_size)
    print("[INFO] node_encoder input_size =", network_attributes.node_encoder.input_size)

    # normalizer 생성
    node_input_size = network_attributes.node_encoder.input_size
    edge_input_size = network_attributes.edge_encoder.input_size
    target_input_size = network_attributes.decoder.output_size

    saving_path = "./debug_tmp/"
    norm_acc_length = optimizerparameterpack.norm_acc_length

    node_normalizer = normalizer.online_normalizer(
        'node_normalizer', node_input_size, norm_acc_length, 1e-6, saving_path, device
    )
    edge_normalizer = normalizer.online_normalizer(
        'mesh_normalizer', edge_input_size, norm_acc_length, 1e-6, saving_path, device
    )
    target_normalizer = normalizer.online_normalizer(
        'target_normalizer', target_input_size, norm_acc_length, 1e-6, saving_path, device
    )

    normalizer_pack = [node_normalizer, edge_normalizer, target_normalizer]

    # dataset 생성 + 로드
    ds = gns_dataset(dataparameterpack, normalizer_pack, device)
    ds.load_dataset(ds.ds_path + ds.training_folder)

    print("[INFO] dataset length =", len(ds))

    # datapack 생성
    datapack = ds.get_data(0, ds.contact_distance, False)

    print("\n===== DATAPACK SHAPES =====")
    print("node_features      :", datapack.nodepack.node_features.shape)
    print("particle_indices   :", datapack.nodepack.particle_indices.shape)
    print("next_particle_idx  :", datapack.nodepack.next_particle_indices.shape)

    print("edge_features      :", datapack.edgepack.edge_features.shape)
    print("receivers          :", datapack.edgepack.receivers.shape)
    print("senders            :", datapack.edgepack.senders.shape)

    if hasattr(datapack.edgepack, "edge_a"):
        print("edge_a             :", datapack.edgepack.edge_a.shape)
        print("edge_b             :", datapack.edgepack.edge_b.shape)
        print("edge_c             :", datapack.edgepack.edge_c.shape)
        print("reverse_edge_idx   :", datapack.edgepack.reverse_edge_idx.shape)
        print("pairwise_mask      :", datapack.edgepack.pairwise_mask.shape)
        print("b_degenerate_mask  :", datapack.edgepack.b_degenerate_mask.shape)
        print("c_degenerate_mask  :", datapack.edgepack.c_degenerate_mask.shape)

    print("normalized_target  :", datapack.targetpack.normalized_target.shape)
    print("target_acc         :", datapack.targetpack.target_acc.shape)
    print("target_vel         :", datapack.targetpack.target_vel.shape)
    print("target_pos         :", datapack.targetpack.target_pos.shape)

    print("\n===== BASIC CHECKS =====")
    print("edge feature dim   :", datapack.edgepack.edge_features.shape[1], "(expected 7)")
    print("node feature dim   :", datapack.nodepack.node_features.shape[1], "(expected 25)")
    print("receiver/sender eq :", datapack.edgepack.receivers.shape[0] == datapack.edgepack.senders.shape[0])

    # graph model 생성
    graph = Graph(
        network_attributes,
        trainingparameterpack,
        device,
        train_flag=True
    )

    graph.graph_net = graph.graph_net.to(device)

    print("\n===== ENCODER TEST =====")
    graph.encoder(datapack)
    print("latent_node        :", graph.latent_node.shape)
    print("latent_edge        :", graph.latent_edge.shape)

    print("\n===== FULL FORWARD TEST =====")
    output, loss_average = graph.forward(datapack, train_flag=True, grid_flag=False)
    print("output             :", output.shape)
    print("loss_average       :", loss_average)
    print("loss finite        :", torch.isfinite(graph.loss).item())
    
    print("\n===== NODE COUNT CONSISTENCY CHECK =====")

    # forward 1
    output1, _ = graph.forward(datapack, train_flag=True, grid_flag=False)
    n1 = output1.shape[0]

    # forward 2 (같은 datapack으로 다시)
    output2, _ = graph.forward(datapack, train_flag=True, grid_flag=False)
    n2 = output2.shape[0]

    print("forward1 output nodes :", n1)
    print("forward2 output nodes :", n2)

    if n1 != n2:
        print("[ERROR] output node count changed!")

    # datapack 내부 확인
    print("\n--- datapack internal ---")
    print("node_features        :", datapack.nodepack.node_features.shape)
    print("particle_indices     :", datapack.nodepack.particle_indices.shape)
    print("next_particle_indices:", datapack.nodepack.next_particle_indices.shape)

    # graph 내부 상태 확인
    print("\n--- graph internal ---")
    print("latent_node shape    :", graph.latent_node.shape)
    print("latent_edge shape    :", graph.latent_edge.shape)

    # particle index 기준 출력 비교
    print("\n--- particle index check ---")
    print("max particle idx     :", datapack.nodepack.particle_indices.max().item())
    print("num particles        :", datapack.nodepack.particle_indices.shape[0])

    # decoder가 particle만 쓰는 경우 대비
    print("\n--- output vs particle ---")
    print("output1 size         :", n1)
    print("particle size        :", datapack.nodepack.particle_indices.shape[0])

    print("\n===== TRAIN STEP TEST =====")

    graph.graph_optimizer.zero_grad()

    output, loss_average = graph.forward(datapack, train_flag=True, grid_flag=False)

    print("before backward - loss       :", graph.loss.item())
    print("before backward - loss_avg   :", loss_average)

    graph.loss.backward()
    print("backward success             : True")

    total_grad_norm_sq = 0.0
    num_params_with_grad = 0
    found_nan_grad = False

    for name, param in graph.graph_net.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.data.norm(2).item()
            total_grad_norm_sq += grad_norm ** 2
            num_params_with_grad += 1

            if not torch.isfinite(param.grad).all():
                found_nan_grad = True
                print(f"[GRAD ERROR] non-finite gradient in: {name}")

    total_grad_norm = total_grad_norm_sq ** 0.5

    print("params with grad             :", num_params_with_grad)
    print("total grad norm (before clip):", total_grad_norm)
    print("non-finite grad exists       :", found_nan_grad)

    clip_threshold = optimizerparameterpack.grad_limit
    clipped_grad_norm = torch.nn.utils.clip_grad_norm_(
        graph.graph_net.parameters(),
        max_norm=clip_threshold,
        norm_type=2
    )

    print("clip threshold               :", clip_threshold)
    print("returned grad norm           :", clipped_grad_norm)

    graph.graph_optimizer.step()
    print("optimizer step success       : True")

    found_nan_param = False
    for name, param in graph.graph_net.named_parameters():
        if not torch.isfinite(param.data).all():
            found_nan_param = True
            print(f"[PARAM ERROR] non-finite parameter in: {name}")

    print("non-finite param exists      :", found_nan_param)

    output2, loss_average2 = graph.forward(datapack, train_flag=True, grid_flag=False)

    print("after step - output          :", output2.shape)
    print("after step - loss            :", graph.loss.item())
    print("after step - loss_avg        :", loss_average2)
    print("after step - loss finite     :", torch.isfinite(graph.loss).item())

    print("\n===== SUCCESS =====")


if __name__ == "__main__":
    main()