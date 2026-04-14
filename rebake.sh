#!/usr/bin/env bash
# =============================================================================
# rebake.sh — Issue #3 수정 후 baked_training_data 재생성 스크립트
#
# 사용법:
#   chmod +x rebake.sh
#   ./rebake.sh
#
# 전제조건:
#   - attributes.py의 ds_path가 서버 경로로 되어 있을 것
#   - conda/venv 환경이 activate된 상태일 것
# =============================================================================

set -euo pipefail   # 오류 발생 시 즉시 중단

BAKE_DIR="baked_training_data"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
BACKUP_DIR="baked_training_data_backup_${TIMESTAMP}"

# ── 1. attributes.py 경로 검사 ─────────────────────────────────────────────
echo "=== [1/5] attributes.py 경로 확인 ==="
if grep -q "C:/Users/AISDL_PJW" attributes.py; then
    echo ""
    echo "  !! 경고: attributes.py에 Windows 로컬 경로가 설정되어 있습니다 !!"
    echo "     ds_path = 'C:/Users/AISDL_PJW/Particle_Simulation/training_40_/'"
    echo ""
    echo "  서버 경로로 변경해야 합니다 (attributes.py 25-29행 주석 참조):"
    echo "     ds_path = '/home/ssdl/PJW/Particle_Simulation/training_40_/'"
    echo ""
    read -rp "  그래도 계속 진행하시겠습니까? [y/N] " confirm
    if [[ "${confirm}" != "y" && "${confirm}" != "Y" ]]; then
        echo "중단합니다. attributes.py를 수정한 뒤 다시 실행하세요."
        exit 1
    fi
else
    echo "  OK: 서버 경로 설정 확인됨"
fi

# ── 2. 기존 baked 폴더 백업 ────────────────────────────────────────────────
echo ""
echo "=== [2/5] 기존 데이터 백업 ==="
if [ -d "${BAKE_DIR}" ]; then
    FILE_COUNT=$(find "${BAKE_DIR}" -name "*.pt" | wc -l)
    echo "  기존 .pt 파일 수: ${FILE_COUNT}"
    echo "  백업 위치: ${BACKUP_DIR}"
    mv "${BAKE_DIR}" "${BACKUP_DIR}"
    echo "  백업 완료: ${BACKUP_DIR}"
else
    echo "  기존 ${BAKE_DIR} 없음, 백업 생략"
fi

mkdir -p "${BAKE_DIR}"
echo "  새 폴더 생성: ${BAKE_DIR}/"

# ── 3. 굽기 전 환경 확인 ──────────────────────────────────────────────────
echo ""
echo "=== [3/5] 환경 확인 ==="
python3 -c "
import torch
print(f'  PyTorch   : {torch.__version__}')
print(f'  CUDA 가용  : {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU       : {torch.cuda.get_device_name(0)}')
    print(f'  VRAM      : {torch.cuda.get_device_properties(0).total_memory // 1024**3} GB')
# baking은 CPU로 수행됨을 명시
print('  ※ baking은 CPU 연산 (radius_graph, KD-tree) — GPU 불필요')
"

# ── 4. 굽기 실행 ──────────────────────────────────────────────────────────
echo ""
echo "=== [4/5] preprocess_data.py 실행 ==="
echo "  시작: $(date)"
START_TS=$(date +%s)

python3 preprocess_data.py

END_TS=$(date +%s)
ELAPSED=$(( END_TS - START_TS ))
echo "  완료: $(date)  (소요: ${ELAPSED}초)"

# ── 5. 결과 검증 ──────────────────────────────────────────────────────────
echo ""
echo "=== [5/5] 결과 검증 ==="
BAKED_COUNT=$(find "${BAKE_DIR}" -name "*.pt" | wc -l)
echo "  생성된 .pt 파일 수: ${BAKED_COUNT}"

python3 - <<'PYEOF'
import torch
import os
import glob

bake_dir = "baked_training_data"
pt_files = sorted(glob.glob(f"{bake_dir}/step_*.pt"))

if not pt_files:
    print("  !! 생성된 .pt 파일이 없습니다!")
    exit(1)

# 첫 번째와 마지막 샘플로 raw 여부 확인
for path in [pt_files[0], pt_files[len(pt_files)//2], pt_files[-1]]:
    dp = torch.load(path, weights_only=False)
    nf = dp.nodepack.node_features
    ef = dp.edgepack.edge_features
    raw_target = dp.targetpack.normalized_target

    # raw feature는 정규화되지 않았으므로 std >> 1이거나 mean이 0이 아닌 경우가 많음
    nf_std  = nf.std().item()
    ef_std  = ef.std().item()
    tgt_max = raw_target.abs().max().item()

    # bake_mode=True로 구웠다면 edge_a[pm]이 0이 아니어야 함 (Issue #1 검증)
    pm_mask = dp.edgepack.pairwise_mask
    cm_mask = ~pm_mask
    if cm_mask.any():
        ea_cm_norm = dp.edgepack.edge_a[cm_mask].norm(dim=1)
        zero_ratio = (ea_cm_norm < 1e-6).float().mean().item()
    else:
        zero_ratio = float('nan')

    fname = os.path.basename(path)
    print(f"  {fname}: node_std={nf_std:.4f}  edge_std={ef_std:.4f}"
          f"  target_max={tgt_max:.4f}  contact_a_zero%={zero_ratio*100:.1f}")

print()
print("  raw 데이터 특징:")
print("    - node/edge std가 클수록 정규화가 적용되지 않은 raw 상태 (정상)")
print("    - contact_a_zero% == 0.0 → Issue #1 fix 반영됨 (정상)")
print("    - target_max이 1e-3 ~ 1e-1 범위 → raw acceleration (mm/step^2)")
PYEOF

echo ""
echo "=== 완료 ==="
echo "  다음 단계: python3 graph_main.py (train_flag=True, fresh_start=True)"
echo "  백업 확인: ls ${BACKUP_DIR}/"
