import json
import os
import threading
import traceback

import backoff
import openai
from openai import OpenAI
from tqdm import tqdm
from dotenv import load_dotenv

from memory import _vllm_engines

load_dotenv()

# ── OpenAI client (GPT-4o judge — final eval only) ──────────────────────────

_openai_client = None
_openai_lock = threading.Lock()


def _get_openai_client():
    global _openai_client
    if _openai_client is None:
        with _openai_lock:
            if _openai_client is None:
                _openai_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    return _openai_client

# ── Local judge singleton (HF transformers — used during RL training only) ──

_JUDGE_MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"

_judge_model     = None
_judge_tokenizer = None
_judge_lock      = threading.Lock()


def _get_judge_model():
    global _judge_model, _judge_tokenizer
    if _judge_model is None:
        with _judge_lock:
            if _judge_model is None:
                import torch
                from transformers import AutoModelForCausalLM, AutoTokenizer
                print(f"[harness] Loading local judge model {_JUDGE_MODEL_ID} ...")
                _judge_tokenizer = AutoTokenizer.from_pretrained(_JUDGE_MODEL_ID)
                _judge_model = AutoModelForCausalLM.from_pretrained(
                    _JUDGE_MODEL_ID,
                    torch_dtype=torch.bfloat16,
                    device_map="auto",
                )
                _judge_model.eval()
                print("[harness] Local judge model loaded.")
    return _judge_model, _judge_tokenizer


def _generate_local_hf(model, tokenizer, messages, max_new_tokens=256):
    """HF-transformers generation path — used only by the local 7B judge."""
    import torch
    inputs = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
    )
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    input_len = inputs["input_ids"].shape[1]
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(out[0, input_len:], skip_special_tokens=True).strip()


# ── Answer Generation (Qwen2.5-3B-Instruct via vLLM) ────────────────────────

_ANSWER_SYSTEM = "You are a helpful assistant with access to a user's conversation history."


def _build_answer_messages(context: str, question: str, question_date: str):
    prompt = (
        f"Here is the conversation history with a user:\n\n"
        f"{context}\n\n"
        f"Current date: {question_date}\n"
        f"Question: {question}\n"
        f"Answer concisely."
    )
    return [
        {"role": "system", "content": _ANSWER_SYSTEM},
        {"role": "user", "content": prompt},
    ]


def ask_qwen(context, question, question_date):
    """Single-question answer call. Kept for callers that don't batch."""
    engine, tokenizer = _vllm_engines.get_answer_engine()
    messages = _build_answer_messages(context, question, question_date)
    return _vllm_engines.generate_one(engine, tokenizer, messages, max_new_tokens=256).strip()


def ask_qwen_batch(contexts, questions, question_dates):
    """Batched answer call — one vLLM.generate() over B prompts.

    Big win for full_history (B 115K-token prefills packed together);
    modest win for rl_memory / rag (small contexts, decode-bound).
    """
    engine, tokenizer = _vllm_engines.get_answer_engine()
    messages_batch = [
        _build_answer_messages(c, q, d)
        for c, q, d in zip(contexts, questions, question_dates)
    ]
    texts = _vllm_engines.generate_many(engine, tokenizer, messages_batch, max_new_tokens=256)
    return [t.strip() for t in texts]


# ── Answer Judging — GPT-4o (final eval only) ────────────────────────────────

@backoff.on_exception(backoff.expo, (openai.RateLimitError, openai.APIError))
def _chat_completions_with_backoff(**kwargs):
    return _get_openai_client().chat.completions.create(**kwargs)


def _get_anscheck_prompt(task, question, answer, response, abstention=False):
    """Build the evaluation prompt. Adapted from LongMemEval evaluate_qa.py."""
    if not abstention:
        if task in ["single-session-user", "single-session-assistant", "multi-session"]:
            template = "I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response is equivalent to the correct answer or contains all the intermediate steps to get the correct answer, you should also answer yes. If the response only contains a subset of the information required by the answer, answer no. \n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."
        elif task == "temporal-reasoning":
            template = "I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response is equivalent to the correct answer or contains all the intermediate steps to get the correct answer, you should also answer yes. If the response only contains a subset of the information required by the answer, answer no. In addition, do not penalize off-by-one errors for the number of days. If the question asks for the number of days/weeks/months, etc., and the model makes off-by-one errors (e.g., predicting 19 days when the answer is 18), the model's response is still correct. \n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."
        elif task == "knowledge-update":
            template = "I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response contains some previous information along with an updated answer, the response should be considered as correct as long as the updated answer is the required answer.\n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."
        elif task == "single-session-preference":
            template = "I will give you a question, a rubric for desired personalized response, and a response from a model. Please answer yes if the response satisfies the desired response. Otherwise, answer no. The model does not need to reflect all the points in the rubric. The response is correct as long as it recalls and utilizes the user's personal information correctly.\n\nQuestion: {}\n\nRubric: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."
        else:
            raise NotImplementedError(f"Unknown task type: {task}")
    else:
        template = "I will give you an unanswerable question, an explanation, and a response from a model. Please answer yes if the model correctly identifies the question as unanswerable. The model could say that the information is incomplete, or some other information is given but the asked information is not.\n\nQuestion: {}\n\nExplanation: {}\n\nModel Response: {}\n\nDoes the model correctly identify the question as unanswerable? Answer yes or no only."

    return template.format(question, answer, response)


def judge_answer(question, answer, hypothesis, question_type, question_id):
    """GPT-4o judge — use for final published evaluation only.

    Returns:
        bool: True if the answer is judged correct.
    """
    abstention = "_abs" in question_id
    prompt = _get_anscheck_prompt(question_type, question, answer, hypothesis, abstention=abstention)
    completion = _chat_completions_with_backoff(
        model="gpt-4o-2024-08-06",
        messages=[{"role": "user", "content": prompt}],
        n=1,
        temperature=0,
        max_tokens=10,
    )
    return "yes" in completion.choices[0].message.content.strip().lower()


def judge_answer_local(question, answer, hypothesis, question_type, question_id):
    """Local Qwen2.5-7B-Instruct judge — use during RL training for reward computation.

    Same prompt format as judge_answer() so results are comparable.

    Returns:
        bool: True if the answer is judged correct.
    """
    abstention = "_abs" in question_id
    prompt = _get_anscheck_prompt(question_type, question, answer, hypothesis, abstention=abstention)
    model, tokenizer = _get_judge_model()
    messages = [{"role": "user", "content": prompt}]
    response = _generate_local_hf(model, tokenizer, messages, max_new_tokens=10)
    return "yes" in response.lower()


# ── Run Questions Pipeline ───────────────────────────────────────────────────

def run_questions(data, memory_factory, output_path, batch_size=16, limit=None):
    """Run the full question-answering pipeline in batches of `batch_size`.

    Each batch creates B fresh MemoryStore instances, runs batched
    indexing (RLMemory overrides index_batch with cross-question batching),
    then a single batched ask_qwen call. This keeps vLLM's continuous
    batching saturated.

    Args:
        data: list of LongMemEval question dicts
        memory_factory: MemoryStore subclass (we call it like a factory: factory())
        output_path: path to write JSONL predictions
        batch_size: number of questions per batch
        limit: optional cap on number of questions (None = all)
    """
    questions = data[:limit] if limit else data
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as out_f:
        for batch_start in tqdm(
            range(0, len(questions), batch_size),
            desc=f"Running {os.path.basename(output_path)}",
        ):
            batch = questions[batch_start : batch_start + batch_size]
            instances = [memory_factory() for _ in batch]
            for inst in instances:
                inst.clear()

            try:
                memory_factory.index_batch(instances, batch)
            except Exception as e:
                print(f"Batch index error at batch_start={batch_start}: {type(e).__name__}: {e}")
                traceback.print_exc()
                # Skip this batch entirely — write empty hypotheses so qids stay aligned
                for q in batch:
                    out_f.write(json.dumps({"question_id": q["question_id"], "hypothesis": ""}) + "\n")
                _drop_instances(instances)
                continue

            contexts = [
                inst.retrieve(q["question"], q["question_date"])
                for inst, q in zip(instances, batch)
            ]

            try:
                predictions = ask_qwen_batch(
                    contexts,
                    [q["question"] for q in batch],
                    [q["question_date"] for q in batch],
                )
            except Exception as e:
                print(f"Batch answer error at batch_start={batch_start}: {type(e).__name__}: {e}")
                traceback.print_exc()
                predictions = [""] * len(batch)

            for q, pred in zip(batch, predictions):
                out_f.write(json.dumps({"question_id": q["question_id"], "hypothesis": pred}) + "\n")
            out_f.flush()

            _drop_instances(instances)

    print(f"Predictions saved to {output_path}")


def _drop_instances(instances):
    """Help tmpdir-backed stores (RLMemory) clean up promptly."""
    for inst in instances:
        try:
            inst.clear()
        except Exception:
            pass
    instances.clear()
