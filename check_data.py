import numpy as np

# 파일 로드
data = np.load('/home/ssdl/PJW/Particle_Simulation/training_40_/train/00000_particle_id.npy')

# 데이터의 타입과 구조(shape) 확인
print(f"Data Shape: {data.shape}")
print(f"Data Type: {data.dtype}")

# 데이터의 앞부분(Head) 출력 (처음 5개 행)
print("--- Head ---")
print(data[:5])