import torch
import os
import dataset
import attributes as attr

def main():
    # 1. Device 설정 (굽는 과정은 CPU로 해도 무방하지만, 기본 세팅 유지)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Bake device: {device}")
    
    # 2. attributes.py에서 파라미터 가져오기 (graph_main.py와 동일한 방식)
    network_attributes_pack, training_attributes_pack = attr.attribute(device)
    training_parameters_pack, testing_parameters_pack, data_parameters_pack, optimizer_parameters_pack = training_attributes_pack

    # (주의) Normalizer는 구울 때 쓰지 않거나 더미를 넣어도 되지만, 구조 유지를 위해 임시 생성
    import normalizer
    node_input_size = network_attributes_pack.node_encoder.input_size
    edge_input_size = network_attributes_pack.edge_encoder.input_size
    target_input_size = network_attributes_pack.edge_decoder_pp.output_size
    
    node_normalizer = normalizer.online_normalizer('node_normalizer', node_input_size, 1, 1e-6, './', 'cpu')
    edge_normalizer = normalizer.online_normalizer('edge_normalizer', edge_input_size, 1, 1e-6, './', 'cpu')
    target_normalizer = normalizer.online_normalizer('target_normalizer', target_input_size, 1, 1e-6, './', 'cpu')
    normalizer_pack = [node_normalizer, edge_normalizer, target_normalizer]

    # 3. 드디어 data_set 정의!
    data_set = dataset.gns_dataset(data_parameters_pack, normalizer_pack, device)
    data_set.load_dataset(data_set.ds_path + data_set.training_folder)

    # Issue #3 Fix: bake_mode=True → normalizer/log-transform 생략, raw feature 저장
    data_set.bake_mode = True

    # 4. 저장할 폴더 생성
    save_dir = 'baked_training_data'
    os.makedirs(save_dir, exist_ok=True)
    
    dataset_length = data_set.__len__()
    print(f"Total data to bake: {dataset_length}")

    # 5. 데이터 굽기 시작 (기저벡터 계산 등 무거운 연산을 여기서 다 끝냄)
    for i in range(dataset_length):
        out_path = os.path.join(save_dir, f'step_{i}.pt')
        if os.path.isfile(out_path):
            if i % 100 == 0:
                print(f"Skip {i}/{dataset_length} (already exists)")
            continue
        data_pack = data_set.get_data(i, data_parameters_pack.contact_distance, rotate_flag=False)
        torch.save(data_pack, out_path)
        if i % 100 == 0:
            print(f"Baked {i}/{dataset_length}...")

    print("Baking Complete! 데이터 굽기 완료!")

if __name__ == '__main__':
    main()