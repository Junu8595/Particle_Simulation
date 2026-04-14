import torch
import numpy as np
import matplotlib.pyplot as plt

def calculate_rmse(pred_pos, gt_pos):
    """
    각 타임스텝별 예측 위치와 실제 위치 간의 RMSE를 계산합니다.
    입력 형태: [time_steps, num_particles, 3]
    """
    squared_error = (pred_pos - gt_pos) ** 2
    rmse_per_step = torch.sqrt(torch.mean(squared_error, dim=(1, 2)))
    return rmse_per_step.cpu().numpy()

def calculate_linear_momentum(vel, mass=1.0):
    """
    각 타임스텝별 시스템 전체의 선운동량 크기를 계산합니다.
    입력 형태: [time_steps, num_particles, 3]
    """
    momentum_vectors = torch.sum(vel * mass, dim=1) 
    momentum_magnitude = torch.norm(momentum_vectors, dim=1)
    return momentum_magnitude.cpu().numpy()

# conv_pos, conv_vel 인자를 제거하고 GT와 Anti(Ours)만 받도록 수정
def plot_performance_comparison(gt_pos, gt_vel, anti_pos, anti_vel):
    """
    Ground Truth와 새 모델(Ours)의 결과를 비교합니다.
    """
    time_steps = gt_pos.shape[0]
    steps = np.arange(time_steps)

    # 1. 지표 계산
    rmse_anti = calculate_rmse(anti_pos, gt_pos)
    mom_gt = calculate_linear_momentum(gt_vel)
    mom_anti = calculate_linear_momentum(anti_vel)

    # 2. 그래프 그리기
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # [Subplot 1] RMSE 오차 누적 그래프
    axes[0].plot(steps, rmse_anti, label='Prediction', color='blue', linewidth=2)
    axes[0].set_title('Position RMSE')
    axes[0].set_xlabel('Time Step')
    axes[0].set_ylabel('RMSE')
    axes[0].legend()
    axes[0].grid(True)

    # [Subplot 2] 선운동량 보존 그래프
    axes[1].plot(steps, mom_gt, label='Ground Truth', color='black', linestyle=':', linewidth=2)
    axes[1].plot(steps, mom_anti, label='Prediction', color='blue', linewidth=2)
    axes[1].set_title('Linear Momentum Conservation')
    axes[1].set_xlabel('Time Step')
    axes[1].set_ylabel('P')
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig('performance_comparison.png', dpi=300)
    plt.show()

if __name__ == "__main__":
    print("Loading rollout data...")
    
    # 렌더링이 완료된 후 생성된 npy 파일의 경로
    # (주의: 실제 저장된 폴더명과 일치하는지 확인하세요!)
    path = r'C:/Users/AISDL_PJW/Particle_Simulation/test_result_2026_04_11_14_58_34_epoch140/'
    
    # Numpy 배열로 로드
    pred_data = np.load(path + 'pred.npy')
    targ_data = np.load(path + 'targ.npy')

    # 계산 함수들이 Torch 텐서를 사용하므로 변환
    pred_tensor = torch.from_numpy(pred_data)
    targ_tensor = torch.from_numpy(targ_data)

    # 데이터 슬라이싱: [Pos(0:3), Vel(3:6), Acc(6:9), ID(9)]
    anti_pos = pred_tensor[:, :, 0:3]
    anti_vel = pred_tensor[:, :, 3:6]
    
    gt_pos = targ_tensor[:, :, 0:3]
    gt_vel = targ_tensor[:, :, 3:6]

    min_steps = min(anti_pos.shape[0], gt_pos.shape[0])
    anti_pos = anti_pos[:min_steps]
    anti_vel = anti_vel[:min_steps]
    gt_pos = gt_pos[:min_steps]
    gt_vel = gt_vel[:min_steps]

    print("Generating performance plots...")
    plot_performance_comparison(gt_pos, gt_vel, anti_pos, anti_vel)
    print("Saved 'performance_comparison.png' successfully!")