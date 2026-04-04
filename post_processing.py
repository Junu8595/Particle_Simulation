import torch
import numpy as np
import matplotlib.pyplot as plt

def calculate_rmse(pred_pos, gt_pos):
    """
    각 타임스텝별 예측 위치와 실제 위치 간의 RMSE를 계산합니다.
    입력 형태: [time_steps, num_particles, 3]
    """
    # 입자별, 축별 오차의 제곱
    squared_error = (pred_pos - gt_pos) ** 2
    # 타임스텝별 평균 제곱근 오차(RMSE)
    rmse_per_step = torch.sqrt(torch.mean(squared_error, dim=(1, 2)))
    return rmse_per_step.cpu().numpy()

def calculate_linear_momentum(vel, mass=1.0):
    """
    각 타임스텝별 시스템 전체의 선운동량 크기를 계산합니다.
    입력 형태: [time_steps, num_particles, 3]
    모든 입자의 질량이 동일하다고 가정(mass=1.0)하여 계산합니다.
    """
    # P = sum(m * v)
    momentum_vectors = torch.sum(vel * mass, dim=1) # [time_steps, 3]
    # 3D 벡터의 크기(Magnitude) 계산
    momentum_magnitude = torch.norm(momentum_vectors, dim=1)
    return momentum_magnitude.cpu().numpy()

def plot_performance_comparison(gt_pos, gt_vel, 
                                conv_pos, conv_vel, 
                                anti_pos, anti_vel):
    """
    Ground Truth, 기존 모델(Conventional), 새 모델(Antisymmetric)의 결과를 비교합니다.
    """
    time_steps = gt_pos.shape[0]
    steps = np.arange(time_steps)

    # 1. 지표 계산
    # RMSE 계산
    rmse_conv = calculate_rmse(conv_pos, gt_pos)
    rmse_anti = calculate_rmse(anti_pos, gt_pos)

    # 선운동량 계산
    mom_gt = calculate_linear_momentum(gt_vel)
    mom_conv = calculate_linear_momentum(conv_vel)
    mom_anti = calculate_linear_momentum(anti_vel)

    # 2. 그래프 그리기
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # [Subplot 1] RMSE 오차 누적 그래프
    axes[0].plot(steps, rmse_conv, label='Conventional Model', color='red', linestyle='--')
    axes[0].plot(steps, rmse_anti, label='Antisymmetric Model (Ours)', color='blue', linewidth=2)
    axes[0].set_title('Rollout Position RMSE over Time')
    axes[0].set_xlabel('Time Step')
    axes[0].set_ylabel('RMSE')
    axes[0].legend()
    axes[0].grid(True)

    # [Subplot 2] 선운동량 보존 그래프
    axes[1].plot(steps, mom_gt, label='Ground Truth', color='black', linestyle=':', linewidth=2)
    axes[1].plot(steps, mom_conv, label='Conventional Model', color='red', linestyle='--')
    axes[1].plot(steps, mom_anti, label='Antisymmetric Model (Ours)', color='blue', linewidth=2)
    axes[1].set_title('Total Linear Momentum Conservation')
    axes[1].set_xlabel('Time Step')
    axes[1].set_ylabel('Momentum Magnitude (|P|)')
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig('performance_comparison.png', dpi=300)
    plt.show()

if __name__ == "__main__":
    # --- 사용 예시 (Dummy Data) ---
    # 실제 사용할 때는 graph_main.py의 test_cycle에서 예측된 pos_prediction, vel_prediction을
    # torch.save() 로 저장해둔 뒤 아래처럼 불러오면 됩니다.
    
    print("Loading rollout data...")
    # T: 롤아웃 스텝 수, N: 파티클 수
    T, N = 100, 5000 
    
    # 예시를 위해 더미 텐서를 생성합니다. (실제로는 .pt 파일을 로드하세요)
    # gt_pos = torch.load('ground_truth_pos.pt')
    
    gt_pos = torch.randn(T, N, 3) 
    gt_vel = torch.randn(T, N, 3)

    # 기존 모델 (오차가 갈수록 커지도록 임의 조작)
    drift = torch.arange(T).view(T, 1, 1) * 0.01
    conv_pos = gt_pos + torch.randn(T, N, 3) * 0.1 + drift
    conv_vel = gt_vel + torch.randn(T, N, 3) * 0.2 + drift * 2

    # 새 모델 (오차가 적고 일정하도록 임의 조작)
    anti_pos = gt_pos + torch.randn(T, N, 3) * 0.05
    anti_vel = gt_vel + torch.randn(T, N, 3) * 0.05

    print("Generating performance plots...")
    plot_performance_comparison(gt_pos, gt_vel, conv_pos, conv_vel, anti_pos, anti_vel)
    print("Saved 'performance_comparison.png' successfully!")