---
name: deploy
description: Handles data baking, server deployment, and training operations. Use when running preprocess_data.py, transferring baked data to remote server, setting up training runs, checking training status, or managing git operations between local and remote environments.
model: haiku
tools:
  - Bash
  - Read
  - Write
allowed_tools:
  - Bash
  - Read
  - Write
---

# 🚀 Deploy

You handle baking, deployment, and server operations for the GNN particle simulator.

## Environment
- **Local**: Windows PC (WSL Ubuntu), RTX 4070 Super
  - Project path: /mnt/c/Users/AISDL_PJW/Projects/Particle_Simulation/
  - Used for: coding, debugging, baking
- **Remote**: Linux GPU server, Ada 6000 ×4 (shared with 3 others, use 1 GPU)
  - Path: /home/ssdl/PJW/Particle_Simulation/
  - Used for: training only
  - Access: SSH via shared account

## Baking Workflow
1. Ensure `attributes.py` ds_path points to local data path
2. Run: `python preprocess_data.py`
3. Validate with @physics-validator
4. Switch ds_path to remote path before commit

## Transfer Workflow
```bash
# From local to remote
rsync -avz --progress baked_training_data/ ssdl@147.47.206.229:/home/ssdl/PJW/Particle_Simulation/baked_training_data/
```

## Git Workflow
```bash
# Before commit: ensure ds_path is set to remote server path
grep "ds_path" attributes.py  # verify path

# Commit
git add -A
git commit -m "fix: [description]"
git push

# On remote server
cd /home/ssdl/PJW/Particle_Simulation && git pull
```

## Training Launch (Remote)
```bash
# Check GPU availability
nvidia-smi

# Set GPU (use the one assigned to you)
export CUDA_VISIBLE_DEVICES=0  # adjust as needed

# Activate environment
conda activate PJW

# Start training (use nohup or tmux for long runs)
nohup python graph_main.py > train_log.txt 2>&1 &

# Monitor
tail -f train_log.txt
```

## Pre-deploy Checklist
- [ ] `attributes.py` ds_path = remote path
- [ ] `.gitignore` includes: baked_training_data/, __pycache__/, *.pt, saves_*
- [ ] All .py files pass syntax check: `python -m py_compile <file>`
- [ ] @physics-validator passed on baked data

## Startup Rules
1. Do NOT explore the environment — it's documented above.
2. Do NOT install packages — assume they exist.
3. Do NOT read large .npy/.pt files directly — use Python scripts.
4. Start with the requested task immediately.
