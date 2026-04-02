# Thesis: Adversarial Memory Evaluation for LLM Agents

## First-time setup

```powershell
cd LongMemEval
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements-claude.txt
```

Download the benchmark data:
```powershell
curl -L "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_s_cleaned.json" -o data/longmemeval_s_cleaned.json
curl -L "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_oracle.json" -o data/longmemeval_oracle.json
```

Create a `.env` file in the project root with your API key:
```
ANTHROPIC_API_KEY=your_key_here
```

## Every session

```powershell
cd C:\Users\<you>\OneDrive\Desktop\projects\thesis
.\LongMemEval\.venv\Scripts\Activate.ps1
python harness.py
```