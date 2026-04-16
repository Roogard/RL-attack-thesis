import json
import os
import threading

import backoff
import openai
from openai import OpenAI
from tqdm import tqdm
from dotenv import load_dotenv

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

# ── Local Qwen model singletons ──────────────────────────────────────────────

_ANSWER_MODEL_ID = "Qwen/Qwen2.5-4B-Instruct"
_JUDGE_MODEL_ID  = "Qwen/Qwen2.5-7B-Instruct"

_answer_model     = None
_answer_tokenizer = None
_answer_lock      = threading.Lock()

_judge_model     = None
_judge_tokenizer = None
_judge_lock      = threading.Lock()


def _get_answer_model():
    global _answer_model, _answer_tokenizer
    if _answer_model is None:
        with _answer_lock:
            if _answer_model is None:
                import torch
                from transformers import AutoModelForCausalLM, AutoTokenizer
                print(f"[harness] Loading answering model {_ANSWER_MODEL_ID} ...")
                _answer_tokenizer = AutoTokenizer.from_pretrained(_ANSWER_MODEL_ID)
                _answer_model = AutoModelForCausalLM.from_pretrained(
                    _ANSWER_MODEL_ID,
                    torch_dtype=torch.bfloat16,
                    device_map="auto",
                )
                _answer_model.eval()
                print("[harness] Answering model loaded.")
    return _answer_model, _answer_tokenizer


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


def _generate_local(model, tokenizer, messages, max_new_tokens=256):
    """Run one forward pass with a chat model and return the decoded string."""
    import torch
    ids = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt"
    )
    device = next(model.parameters()).device
    ids = ids.to(device)
    with torch.no_grad():
        out = model.generate(
            ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip()


# ── Answer Generation (Qwen2.5-4B-Instruct) ─────────────────────────────────

def ask_qwen(context, question, question_date):
    """Call the local Qwen answering model with retrieved context. Returns answer string."""
    model, tokenizer = _get_answer_model()
    prompt = (
        f"Here is the conversation history with a user:\n\n"
        f"{context}\n\n"
        f"Current date: {question_date}\n"
        f"Question: {question}\n"
        f"Answer concisely."
    )
    messages = [
        {"role": "system", "content": "You are a helpful assistant with access to a user's conversation history."},
        {"role": "user", "content": prompt},
    ]
    return _generate_local(model, tokenizer, messages, max_new_tokens=256)


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
    response = _generate_local(model, tokenizer, messages, max_new_tokens=10)
    return "yes" in response.lower()


# ── Run Questions Pipeline ───────────────────────────────────────────────────

def run_questions(data, memory_store, output_path, limit=None):
    """Run the full question-answering pipeline for a memory store.

    Args:
        data: list of LongMemEval question dicts
        memory_store: a MemoryStore instance (handles indexing + retrieval)
        output_path: path to write JSONL predictions
        limit: optional cap on number of questions (None = all)
    """
    questions = data[:limit] if limit else data
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as out_f:
        for item in tqdm(questions, desc=f"Running {os.path.basename(output_path)}"):
            memory_store.clear()
            memory_store.index(
                item["haystack_sessions"],
                item["haystack_dates"],
                item.get("haystack_session_ids", [str(i) for i in range(len(item["haystack_sessions"]))]),
            )
            context = memory_store.retrieve(item["question"], item["question_date"])

            try:
                prediction = ask_qwen(context, item["question"], item["question_date"])
            except Exception as e:
                prediction = ""
                print(f"Error on {item['question_id']}: {e}")

            result = {
                "question_id": item["question_id"],
                "hypothesis": prediction,
            }
            out_f.write(json.dumps(result) + "\n")

    print(f"Predictions saved to {output_path}")
