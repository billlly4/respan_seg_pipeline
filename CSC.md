# CSC Mahti — Connection & Usage Guide

## 1. Generate SSH Key Pair

Run once on your local machine:

```bash
ssh-keygen -t ed25519 -C "your@email.com"
```

This creates:
- `~/.ssh/id_ed25519` — private key (never share this)
- `~/.ssh/id_ed25519.pub` — public key (this goes to CSC)

## 2. Add Public Key to CSC

1. Go to [my.csc.fi](https://my.csc.fi)
2. Profile → SSH public keys → Add key
3. Paste the contents of `~/.ssh/id_ed25519.pub`

Wait a few minutes for it to propagate.

## 3. Connect to Mahti

```bash
ssh <username>@mahti.csc.fi
```

Your username is the one you use to log in to my.csc.fi.

## 4. Transfer Files

Copy local files to Mahti:

```bash
scp -r local/folder/ <username>@mahti.csc.fi:/scratch/project_XXXXXXX/
```

Copy files from Mahti to local:

```bash
scp -r <username>@mahti.csc.fi:/scratch/project_XXXXXXX/results/ ./results/
```

## 5. Project Scratch Directory

All large data and checkpoints go under scratch (not home — home has a tiny quota):

```
/scratch/project_XXXXXXX/
```

Replace `XXXXXXX` with your project number (visible at my.csc.fi under Projects).

## 6. Python Environment (Tykky)

Mahti uses Tykky to wrap conda/pip environments into containers. Do this once per project:

```bash
module load tykky
pip-containerize new --prefix /scratch/project_XXXXXXX/env requirements.txt
```

Activate it each session:

```bash
export PATH="/scratch/project_XXXXXXX/env/bin:$PATH"
```

Or for an editable install inside the env:

```bash
pip install -e ./LLaVA/ --target /scratch/project_XXXXXXX/env/lib/python3.x/site-packages/
```

## 7. HuggingFace Model Cache

Set `HF_HOME` so models download to scratch instead of home (home will fill up):

```bash
export HF_HOME=/scratch/project_XXXXXXX/hf_cache
```

Add this to your `~/.bashrc` on Mahti so it's set automatically.

## 8. Interactive GPU Session

```bash
srun --account=project_2019059 --partition=gpusmall --nodes=1 --ntasks=1 \
     --cpus-per-task=10 --mem=64G --gres=gpu:a100:1 --time=06:00:00 --pty bash
```

Once on the GPU node, run training:

```bash
cd /scratch/project_2019059/respan/code/ResPan
/projappl/project_2019059/respan/envs/respaninternal/bin/python train_nnunet.py finetune
```

**Note:** Do NOT use `export PATH=...` to activate — Tykky containers must be called via full path.
Environments: `/projappl/project_2019059/respan/envs/respaninternal/` (nnU-Net / PyTorch)
              `/projappl/project_2019059/respan/envs/respanev/` (RESPAN analysis)

## 9. SLURM Batch Job

For longer unattended runs, submit a job script:

```bash
#!/bin/bash
#SBATCH --account=project_2019059
#SBATCH --partition=gpusmall
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=36:00:00
#SBATCH --output=logs/job_%j.out

cd /scratch/project_2019059/respan/code/ResPan
/projappl/project_2019059/respan/envs/respaninternal/bin/python train_nnunet.py finetune
```

Submit with:

```bash
sbatch train.sh
```

Monitor with:

```bash
squeue --me
```

Cancel a job:

```bash
scancel <job_id>
```
