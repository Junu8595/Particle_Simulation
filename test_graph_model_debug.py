import os
import torch
import dataset
import normalizer
import attributes as attr
import graph_model as gm

# 1. 환경 설정
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# 2. 어트리뷰트 및 파라미터 로드
network_attributes_pack, training_attributes_pack = attr.attribute(device)
training_parameters_pack, testing_parameters_pack, data_parameters_pack, optimizer_parameters_pack = training_attributes_pack

# 3. 노말라이저 세팅
node_input_size = network_attributes_pack.node_encoder.input_size
edge_input_size = network_attributes_pack.edge_encoder.input_size

# 참고: 타겟 노말라이저는 여전히 가속도(3차원) 기준입니다.
target_input_size = 3 

node_normalizer = normalizer.online_normalizer('node_normalizer', node_input_size, device=device)
edge_normalizer = normalizer.online_normalizer('edge_normalizer', edge_input_size, device=device)
target_normalizer = normalizer.online_normalizer('target_normalizer', target_input_size, device=device)

normalizer_pack = [node_normalizer, edge_normalizer, target_normalizer]

# 4. 데이터셋 로드 (1set 샘플 데이터 경로)
ds = dataset.gns_dataset(data_parameters_pack, normalizer_pack, device)
local_data_path = r"C:\Projects\GNN_Particle_Sim_1\training_40_" # 실제 데이터 경로로 수정해주세요
ds.load_dataset(local_data_path)

# 첫 번째 데이터팩 가져오기
data_pack = ds.get_data(0, ds.contact_distance, False)
print("\n--- Data Pack Info ---")
print("Node features shape:", data_pack.nodepack.node_features.shape)
print("Edge features shape:", data_pack.edgepack.edge_features.shape)
print("Target shape:", data_pack.targetpack.normalized_target.shape)
print("Pairwise mask sum (P-P edges):", data_pack.edgepack.pairwise_mask.sum().item())
print("Total edges:", data_pack.edgepack.pairwise_mask.shape[0])

# 5. 모델 초기화
print("\n--- Initializing Graph Model ---")
# train_flag=True 로 설정하여 디코더에서 Loss가 계산되도록 합니다.
graph = gm.Graph(network_attributes_pack, training_parameters_pack, device, train_flag=True)

# 모델 내부 구조가 우리가 의도한 대로(edge_decoder_pp, pm) 잘 생성되었는지 확인
print("Graph Sub-networks:")
for idx, module in enumerate(graph.graph_net.sub_nets):
    print(f" - SubNet {idx}")

# 6. Forward Pass 및 Loss 확인
print("\n--- Running Forward Pass ---")
try:
    # forward 실행
    output, loss = graph.forward(data_pack)
    
    print("[SUCCESS] Forward pass successful!")
    print(f"Output shape: {output.shape} (Expected: [N, 3])")
    print(f"Loss (Normalized): {loss[0]:.6f}")
    print(f"Loss (Absolute MAE/RMSE): {loss[1]:.6f}")
    
    # 텐서에 NaN이 있는지 체크 (물리 시뮬레이터에서 매우 중요)
    if torch.isnan(output).any():
        print("[WARNING] NaN detected in model output!")
    else:
        print("[SUCCESS] No NaNs detected in output.")
        
except Exception as e:
    print("[ERROR] Error during forward pass:")
    print(e)
    import traceback
    traceback.print_exc()