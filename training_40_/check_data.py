import numpy as np

# 파일 경로를 입력하세요 (예: 이미지 속 00034_particle_positions.npy)
file_path = r"C:\Projects\GNN_Particle_Sim_1\training_40_\00000_particle_positions.npy"

# 데이터 불러오기
data = np.load(file_path)

# 데이터 확인
#print("데이터 타입:", type(data))
#print("데이터 형태(Shape):", data.shape)
#print("데이터 내용:\n", data)

a = 29000+40000+80000+33000+50000+150000
b = 6000*22
print("합계:", a)
print("b:", b)
print("차이:", 1000000-a - b)