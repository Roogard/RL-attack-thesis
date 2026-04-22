# Setup & Daily Workflow

Quick reference for getting this project running on the laptop and on the Microway GPU server.
Repo: https://github.com/Roogard/RL-attack-thesis

## First-time setup

### Laptop (Windows PowerShell)

```powershell
cd C:\Users\roota\OneDrive\Desktop\Projects\RL-attack-thesis
git pull
```

You write code and push from here. No venv needed unless you want to run things locally.

### Microway (SSH, one time only)

```bash
ssh aroot@microway.cis.umassd.edu
bash                                            # ensure you're in bash, not sh
cd /scratch/aroot
git clone https://github.com/Roogard/RL-attack-thesis.git
cd RL-attack-thesis
bash scripts/bootstrap.sh                       # creates .venv, installs deps, downloads HF models (slow first time only)
```

Model download is ~30GB into `/scratch/aroot/hf_cache`. It's a **one-time cost** — cached forever on `/scratch`. Later re-runs of `bootstrap.sh` skip the download.

Then from the **laptop**, copy the LongMemEval data (gitignored, not in the repo):

```powershell
scp LongMemEval\data\longmemeval_s_cleaned.json `
    LongMemEval\data\longmemeval_oracle.json `
    aroot@microway.cis.umassd.edu:/scratch/aroot/RL-attack-thesis/LongMemEval/data/
```

Also put your OpenAI key in `.env` on microway (only needed if you run `eval.py` on the server — normally you run eval locally):

```bash
# on microway
cat > .env <<'EOF'
OPENAI_API_KEY=sk-...
EOF
```

## Every session (returning to work)

### On laptop

```powershell
cd C:\Users\roota\OneDrive\Desktop\Projects\RL-attack-thesis
git pull                                        # pick up anything pushed from microway
# ... edit code, commit, push ...
git add -A
git commit -m "your message"
git push
```

### On microway

```bash
ssh aroot@microway.cis.umassd.edu
bash                                            # login shell is sh — run bash explicitly
cd /scratch/aroot/RL-attack-thesis
git pull                                        # pick up whatever you pushed from laptop
```

If you added new dependencies to `requirements.txt`, re-run `bash scripts/bootstrap.sh` — it's idempotent.

## Running baselines on microway

Parallel run, one memory type per GPU, persistent via tmux:

```bash
tmux new -s lme                                 # new session (Ctrl-b d to detach, `tmux attach -t lme` to reattach)

# inside tmux — create 3 windows (Ctrl-b c), one per memory type
./scripts/run_memory.sh full_history 0          # window 0, GPU 0
./scripts/run_memory.sh rag          1          # window 1, GPU 1 (Ctrl-b c first)
./scripts/run_memory.sh rl_memory    2          # window 2, GPU 2 (Ctrl-b c first)
```

tmux cheatsheet:
- `Ctrl-b d` — detach (job keeps running)
- `tmux attach -t lme` — reattach
- `Ctrl-b c` — new window
- `Ctrl-b n` / `Ctrl-b p` — next / previous window
- `Ctrl-b &` — kill current window

Monitor from another SSH session:
```bash
tail -f logs/full_history_gpu0_*.log
nvidia-smi -l 5
```

## Pulling results back & judging locally

```powershell
# on laptop
scp -r aroot@microway.cis.umassd.edu:/scratch/aroot/RL-attack-thesis/results/* results/

# merge the three timestamped dirs into one baseline dir, then run GPT-4o judge
mkdir results\baseline
# (copy the three .jsonl files into results\baseline\)
python -c "from eval import run_eval; run_eval('results/baseline')"
```

## Smoke test (verify env before committing to the 8+ hour full run)

```bash
# on microway, inside the repo
CUDA_VISIBLE_DEVICES=0 .venv/bin/python main.py --memory full_history --limit 5 --skip-eval
```

Should produce `results/<timestamp>/full_history.jsonl` with 5 rows in ~2 minutes.

## Troubleshooting

- **`source: not found`** — you're in `sh`, not `bash`. Run `bash`.
- **`externally-managed-environment`** — you're using system pip. Use `.venv/bin/pip` or activate the venv first.
- **git clone prompts for password** — GitHub killed password auth in 2021. Use a Personal Access Token as the password, set up SSH keys, or make the repo public.
- **HF 401 on model download** — model ID is wrong (HF returns generic auth error for nonexistent public repos). Check spelling in `harness.py` and `scripts/bootstrap.sh`.
- **Out of GPU memory** — someone else grabbed the GPU. `nvidia-smi` to see who, pick a different `CUDA_VISIBLE_DEVICES` index.
