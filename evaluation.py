import json, re, os, warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

# ------------------------
# config
# ------------------------
MEDQA_PATH    = "medqa_50.json"
MEDMCQA_PATH  = "medmcqa_50.json"
PUBMEDQA_PATH = "pubmedqa_50.json"

MODEL_REPO = "meta-llama/Llama-3.1-8B-Instruct"
USE_CHAT   = True  # try chat template; set to False to keep plain prompts

# dtype & model
use_bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8
torch_dtype = torch.bfloat16 if use_bf16 else torch.float16

tokenizer = AutoTokenizer.from_pretrained(MODEL_REPO, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_REPO,
    torch_dtype=torch_dtype,
    device_map="auto",
)
model.eval()
print(f"Model loaded: {MODEL_REPO}")

# ------------------------
# data containers
# ------------------------
@dataclass
class MCItem:
    question: str
    options: Dict[str, str]       # keys: "A".."D" (or up to "E")
    answer_letter: str            # gold letter
    source_id: Optional[str] = None

@dataclass
class YesNoMaybeItem:
    question: str
    contexts: List[str]
    gold_label: str               # "yes"/"no"/"maybe"
    source_id: Optional[str] = None

def _read_json_any(path: str) -> Union[dict, list]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

# ------------------------
# loaders
# ------------------------
def load_medqa(path: str) -> List[MCItem]:
    raw = _read_json_any(path)
    items: List[MCItem] = []
    bad = 0
    iterator = raw.items() if isinstance(raw, dict) else enumerate(raw)
    for key, ex in iterator:
        q = str(ex.get("question", "")).strip()
        opts_in = ex.get("options", {})
        opts = {k.upper(): str(v) for k, v in opts_in.items() if k.upper() in ["A","B","C","D","E"]}
        if len(opts) < 2:
            bad += 1
            continue
        ans = ex.get("answer_idx", ex.get("answer", ""))
        ans = str(ans).strip().upper()
        if ans not in opts:
            # maybe the gold is full text; try to map
            inv = {v.strip(): k for k, v in opts.items()}
            ans = inv.get(ans, "")
        if ans not in opts:
            bad += 1
            continue
        items.append(MCItem(q, opts, ans, str(key)))
    if bad:
        print(f"[MedQA] skipped {bad} malformed item(s). Using {len(items)}.")
    return items

def load_medmcqa(path: str) -> List[MCItem]:
    """
    Robust MedMCQA loader.
    - Builds options from explicit dict or opa/opb/opc/opd(/ope).
    - Normalizes gold from multiple possible fields.
    - NEW: supports numeric 'cop' (1-based) -> A/B/C/D/E mapping.
    """
    raw = _read_json_any(path)
    items: List[MCItem] = []
    bad = 0

    # Maps for numeric or letter answers
    idx_to_letter = {1: "A", 2: "B", 3: "C", 4: "D", 5: "E"}
    strnum_to_letter = {"1": "A", "2": "B", "3": "C", "4": "D", "5": "E"}

    iterator = raw.items() if isinstance(raw, dict) else enumerate(raw)
    for key, ex in iterator:
        q = str(ex.get("question", "")).strip()

        # Build options: prefer dict; fallback to opa..ope
        opts: Dict[str, str] = {}
        if isinstance(ex.get("options"), dict):
            for k, v in ex["options"].items():
                kk = str(k).strip().upper()
                if kk in ["A", "B", "C", "D", "E"]:
                    opts[kk] = str(v)
        else:
            for L, fld in {"A": "opa", "B": "opb", "C": "opc", "D": "opd", "E": "ope"}.items():
                if fld in ex and ex[fld] is not None:
                    opts[L] = str(ex[fld])

        if len(opts) < 2 or not q:
            bad += 1
            continue

        # Extract gold from common fields (priority: cop > answer_idx > answer > label)
        gold_raw = ex.get("cop", ex.get("answer_idx", ex.get("answer", ex.get("label", ""))))

        # Normalize to a letter in A–E
        gold: str = ""
        if isinstance(gold_raw, int):
            gold = idx_to_letter.get(gold_raw, "")
        else:
            s = str(gold_raw).strip()
            # numeric string?
            if s in strnum_to_letter:
                gold = strnum_to_letter[s]
            # already a letter?
            elif len(s) == 1 and s.lower() in "abcde":
                gold = s.upper()
            elif s.upper() in ["A", "B", "C", "D", "E"]:
                gold = s.upper()
            else:
                # final fallback: try text-to-letter mapping
                inv = {v.strip(): k for k, v in opts.items()}
                gold = inv.get(s, "")

        # Validate gold is one of the present options
        if gold not in opts:
            bad += 1
            continue

        items.append(MCItem(q, opts, gold, str(key)))

    if bad:
        print(f"[MedMCQA] skipped {bad} malformed item(s). Using {len(items)}.")
    return items

def load_pubmedqa(path: str) -> List[YesNoMaybeItem]:
    raw = _read_json_any(path)
    items: List[YesNoMaybeItem] = []
    bad = 0
    iterator = raw.items() if isinstance(raw, dict) else enumerate(raw)
    for key, ex in iterator:
        q = str(ex.get("QUESTION", ex.get("question",""))).strip()
        ctx = ex.get("CONTEXTS", ex.get("contexts", []))
        if not isinstance(ctx, list):
            ctx = [str(ctx)]
        gold = str(ex.get("final_decision", ex.get("answer",""))).strip().lower()
        if gold not in {"yes","no","maybe"}:
            bad += 1
            continue
        items.append(YesNoMaybeItem(q, [str(c) for c in ctx], gold, str(key)))
    if bad:
        print(f"[PubMedQA] skipped {bad} malformed item(s). Using {len(items)}.")
    return items

# ------------------------
# prompting & generation
# ------------------------
def apply_chat_template(user_msg: str, system_msg: str = "") -> str:
    msgs = []
    if system_msg:
        msgs.append({"role": "system", "content": system_msg})
    msgs.append({"role": "user", "content": user_msg})
    return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

def mc_prompt(item: MCItem) -> str:
    letters = "".join(sorted(item.options.keys()))
    opts = "\n".join([f"{k}. {v}" for k, v in item.options.items()])
    user = (
        "You are answering a multiple-choice medical question.\n"
        "Return ONLY one uppercase letter from the allowed set.\n\n"
        f"Question:\n{item.question}\n\nOptions:\n{opts}\n\n"
        f"Answer with ONLY ONE LETTER from [{letters}].\nAnswer:"
    )
    return apply_chat_template(user) if USE_CHAT else user

def pubmedqa_prompt(item: YesNoMaybeItem) -> str:
    ctx = "\n".join(f"- {c}" for c in item.contexts[:6])
    user = (
        "You are assessing a biomedical yes/no/maybe question.\n"
        "Return ONLY one token: yes, no, or maybe (lowercase).\n\n"
        f"Question:\n{item.question}\n\nEvidence:\n{ctx}\n\nAnswer:"
    )
    return apply_chat_template(user) if USE_CHAT else user

@torch.no_grad()
def generate_answer(prompt: str, max_new_tokens: int = 24) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,           # deterministic
        pad_token_id=tokenizer.eos_token_id,
    )
    gen = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return gen.split("Answer:")[-1].strip()

LETTER_RE = re.compile(r"\b([A-E])\b")
YNM_RE    = re.compile(r"\b(yes|no|maybe)\b", re.IGNORECASE)

def parse_mc_letter(text: str, allowed: List[str]) -> Optional[str]:
    # only accept letters in the allowed set for this item
    m = LETTER_RE.search(text.upper())
    if not m:
        return None
    cand = m.group(1)
    return cand if cand in allowed else None

def parse_ynm(text: str) -> Optional[str]:
    m = YNM_RE.search(text)
    return m.group(1).lower() if m else None

# ------------------------
# evaluation (with tqdm)
# ------------------------
def eval_mcq(items: List[MCItem], desc: str) -> float:
    correct, used = 0, 0
    for it in tqdm(items, desc=desc, ncols=80):
        prompt = mc_prompt(it)
        out = generate_answer(prompt)
        allowed = sorted(list(it.options.keys()))
        pred = parse_mc_letter(out, allowed) or ""
        if pred:     # only count a trial when we could parse a letter
            used += 1
            correct += int(pred == it.answer_letter)
        else:
            used += 1  # still count; exact-match metric treats unparsable as wrong
    return correct / max(1, used)

def eval_pubmedqa(items: List[YesNoMaybeItem], desc: str) -> float:
    correct, used = 0, 0
    for it in tqdm(items, desc=desc, ncols=80):
        prompt = pubmedqa_prompt(it)
        out = generate_answer(prompt)
        pred = parse_ynm(out) or ""
        used += 1
        correct += int(pred == it.gold_label)
    return correct / max(1, used)

# ------------------------
# run
# ------------------------
medqa_items    = load_medqa(MEDQA_PATH)
medmcqa_items  = load_medmcqa(MEDMCQA_PATH)
pubmedqa_items = load_pubmedqa(PUBMEDQA_PATH)

medqa_acc    = eval_mcq(medqa_items,   "Scoring MedQA")
medmcqa_acc  = eval_mcq(medmcqa_items, "Scoring MedMCQA")
pubmedqa_acc = eval_pubmedqa(pubmedqa_items, "Scoring PubMedQA")

macro_acc = (medqa_acc + medmcqa_acc + pubmedqa_acc) / 3.0

print(f"\nMedQA accuracy (acc,none):     {medqa_acc:.3f}")
print(f"MedMCQA accuracy (acc,none):   {medmcqa_acc:.3f}")
print(f"PubMedQA accuracy (acc,none):  {pubmedqa_acc:.3f}")
print("-" * 60)
print(f"Macro-average accuracy:        {macro_acc:.3f}")