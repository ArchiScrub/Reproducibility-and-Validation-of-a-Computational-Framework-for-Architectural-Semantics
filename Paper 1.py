"""
Paper 1 Pipeline: A Computational Framework for Analysing Specialised Theoretical Discourse

This module keeps the computations mandated for Paper 1 and adds:
  • Corpus SHA-256 caching for Word2Vec and BERT fine-tuning (skips retraining when inputs unchanged)
  • Best-epoch selection for BERT based on validation loss (no hyperparameter tuning here)
  • Reproducibility logs (library versions and config snapshots)
  • Extra visual diagnostics: ARI bar chart and WordSim scatter with trendline

The canonical artefacts required by the plan remain unchanged, e.g.:
  - figures/corpus_manifest.csv
  - figures/definitional_presence_FULL.csv, figures/definitional_presence_CLEAN.csv
  - figures/definitional_similarity_deltas.csv (if definitional_pairs provided)
  - figures/top_cooccurrences.png
  - figures/top_similarities_word2vec_cbow.png
  - figures/top_similarities_word2vec_skip-gram.png
  - figures/top_similarities_fine-tuned_bert.png
  - figures/bert_training_loss_FULL.png, figures/bert_training_loss_CLEAN.png
  - figures/cumulative_loss.png
  - results/ari_scores_full.csv, results/ari_scores_clean.csv
  - results/wordsim_scores_paper1.csv

Model caches live under models/FULL/… and models/CLEAN/…
"""

# =========================
# SECTION 1: IMPORTS
# =========================

from __future__ import annotations  # must be first (after comments/docstring)

# --- Environment setup for full reproducibility (set BEFORE importing numpy/torch/sklearn) ---
import os
os.environ["PYTHONHASHSEED"] = "42"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# --- Standard library imports ---
import collections
from collections import defaultdict
from collections import Counter
import gc
import hashlib
from hashlib import sha256
import io
import json
import logging
import platform
import psutil
import random
import re
import shutil
import sys
import time
import transformers
import transformers as _tf
from pathlib import Path
from peft import LoraConfig, TaskType, get_peft_model, PeftModel
from typing import Dict, List, Optional, Tuple, Union, cast

# --- Third-party library imports ---
# Load NumPy early (after env vars) so BLAS picks up single-thread settings
import pandas as pd, numpy as np, math

# PyTorch next, then lock determinism and CPU thread cap (does NOT disable GPU)
import torch
torch.use_deterministic_algorithms(True)           # enforce deterministic kernels
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.set_num_threads(1)                           # cap CPU threads; GPU still used normally

print("[det] cudnn.benchmark:", torch.backends.cudnn.benchmark,
      "deterministic:", torch.backends.cudnn.deterministic,
      "tf32(matmul,cudnn):", torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32,
      "cpu_threads:", torch.get_num_threads())

# The rest of the stack (safe to import now)
import gensim
import matplotlib
matplotlib.use("Agg")  # headless, stable backend
import matplotlib.pyplot as plt
plt.rcParams.update({
    "figure.dpi": 300,
    "font.family": "DejaVu Sans"
})
from matplotlib.patches import Rectangle
import regex as re_rx
import seaborn as sns
import subprocess
import spacy
import torch.optim as optim
import yaml
from functools import partial
from typing import Any, Mapping
from gensim.models import Word2Vec
from gensim.models.callbacks import CallbackAny2Vec
from gensim.utils import RULE_DEFAULT, RULE_KEEP
from scipy.stats import spearmanr
from scipy.optimize import linear_sum_assignment
from sklearn import __version__ as sklearn_version
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import adjusted_rand_score as _ari
from sklearn.metrics.pairwise import cosine_distances
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    BertForMaskedLM,
    BertTokenizer,
    DataCollatorForLanguageModeling,
    get_scheduler,
    BertConfig,
)

# Reduce HF warning noise in logs (keeps errors)
from transformers import logging as hf_logging
hf_logging.set_verbosity_error()

# Optional: try CPU affinity for slightly more stable timing (Linux only).
# If psutil isn't installed, we skip quietly.
try:
    psutil.Process().cpu_affinity([0])
except (ImportError, AttributeError, NotImplementedError, RuntimeError):
    pass


# =========================
# SECTION 2: FOLDERS & CONFIG
# =========================

# --- Directory Setup ---
FIG = Path("figures"); FIG.mkdir(exist_ok=True)
RES = Path("results"); RES.mkdir(exist_ok=True)
MOD = Path("models");  MOD.mkdir(exist_ok=True)

# --- Load Configuration Files ---
with io.open("config.yaml", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)
with io.open("filters.yaml", encoding="utf-8") as f:
    flt = yaml.safe_load(f)

# --- Full Determinism Setup ---
SEED = int(cfg.get("seed", 42))

# 1. Seed all relevant random number generators
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# 2. Set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
if device.type == 'cpu':
    print("Warning: Running on CPU. Training will be much slower than on a CUDA-capable GPU.")

# ---- Paths & lists from config.yaml ----
corpus_dir = Path(cfg["corpus_dir"])
ALL_TARGET_WORDS_FROM_CONFIG = cfg.get("target_words", [])
CORE_TARGET_WORDS_FROM_CONFIG = cfg.get("core_target_words", [])
classification_categories = cfg.get("classification_categories", {})
era_mapping = cfg.get("era_mapping", {})
location_mapping = cfg.get("location_mapping", {})
PHYSICAL_ANCHORS = {'engawa', 'shōji', 'fusuma', 'tomeishi', 'tokonoma', 'chigaidana'}
CONCEPTUAL_ANCHORS = {'ma', 'mu', 'yohaku', 'sabi', 'wabi', 'en'}

# ---- Filters from filters.yaml ----
NORMALISATION_MAP = flt.get("normalization_mapping", {})
CHAR_WHITELIST_RE = re_rx.compile(flt.get("character_whitelist_regex", r"[^\p{Latin}\p{Han}\p{Hiragana}\p{Katakana}\p{N}\-]+"), flags=re_rx.UNICODE)
NUMBER_RE = re_rx.compile(flt.get("number_regex", r"\d"))
ALLOWED_POS = set(flt.get("allowed_pos", ["NOUN", "PROPN", "ADJ", "VERB", "ADV"]))
PASS_JAPANESE = bool(flt.get("passthrough", {}).get("japanese_chars", True))
STOP_SINGLE_LETTERS = bool(flt.get("stoplists", {}).get("single_letters", True))
LATIN_UTILITIES = set(flt.get("stoplists", {}).get("latin_utilities", ["eg","e.g","e.g.","etc","etc.","ie","i.e","i.e.","c.","d.","r.","b.","a","an","or"]))
HONOUR_SPACY_STOP = bool(flt.get("stoplists", {}).get("spacy_is_stop", True))
KEEP_TARGETS = bool(flt.get("target_handling", {}).get("always_keep_targets", True))
TARGETS_AS_PROPN = bool(flt.get("target_handling", {}).get("spacy_pos_as_propn", True))
MERGE_RE_FOLLOWING = bool(flt.get("token_merges", {}).get("merge_re_following", True))
SENT_SPLIT_RE = re.compile(flt.get("sentence_split_regex", r"[.\n]+"))
DEF_PATTERNS_RAW = list(flt.get("definitional_patterns", []))
BERT_MIN_FREQ = int(flt.get("bert_vocab", {}).get("min_freq", 5))
BERT_EXCLUDE = set(flt.get("bert_vocab", {}).get("exclude_tokens", []))

# ---- Base tokenizer/model ----
# Pin the revision (commit hash or tag) via config for full reproducibility.
hf_revision = cfg.get("hf_revision", None)
if not hf_revision or str(hf_revision).lower() in {"main", "latest"}:
    raise ValueError("Pin a stable Hugging Face revision for full reproducibility (set cfg.hf_revision).")

bert_tok: BertTokenizer = BertTokenizer.from_pretrained(
    "bert-base-multilingual-cased",
    revision=hf_revision,
    local_files_only=False
)
bert_mod: BertForMaskedLM = BertForMaskedLM.from_pretrained(
    "bert-base-multilingual-cased",
    revision=hf_revision,
    attn_implementation="eager",
    local_files_only=False
)

# ---- spaCy ----
try:
    nlp_en = spacy.load("en_core_web_sm", disable=["ner", "parser"])
except OSError:
    print("\n[ERROR] spaCy model 'en_core_web_sm' not found.")
    print("Please run: python -m spacy download en_core_web_sm\n")
    sys.exit(1)
nlp_en.max_length = 2_000_000

# Mark custom terms as PROPN if requested
if TARGETS_AS_PROPN and (ALL_TARGET_WORDS_FROM_CONFIG or CORE_TARGET_WORDS_FROM_CONFIG):
    targets_all = sorted(set(ALL_TARGET_WORDS_FROM_CONFIG + CORE_TARGET_WORDS_FROM_CONFIG))
    if not nlp_en.has_pipe("attribute_ruler"):
        nlp_en.add_pipe("attribute_ruler", before="tok2vec")
    from spacy.pipeline import AttributeRuler
    ruler: AttributeRuler = nlp_en.get_pipe("attribute_ruler")  # type: ignore
    patterns = [{"patterns": [[{"LOWER": t.lower()}]], "attrs": {"POS": "PROPN"}} for t in targets_all]
    ruler.add_patterns(patterns)

# =========================
# SECTION 3: COMPUTATION SWITCHES
# =========================
RUN_CORPUS_BUILD = True
RUN_DEFINITIONAL_AUDIT = True
RUN_W2V_TRAINING = True
RUN_BERT_FINETUNING = True
RUN_VISUAL_DIAGNOSTICS = True
RUN_ARI_VALIDATION = True
RUN_BENCHMARKS = True

# --- Granular Switches for Faster Reruns ---
# If True, this will force training switches to False and only run validation/benchmarks on cached models.
RUN_EVAL_ONLY = False
# If True, this will add a t-SNE plot to the visual diagnostics.
RUN_TSNE = True
# If True, runs a permutation test for ARI to check statistical significance.
RUN_ARI_PERMUTATION_TEST = True
# If True, enables bootstrapping for ARI confidence intervals.
RUN_BOOTSTRAP_ARI_CI = True

# --- Hard clean switches (set True to force a brand-new run without reusing any cache) ---
FORCE_CLEAN_RUN = True          # clean everything, both W2V and BERT
FORCE_CLEAN_W2V = False          # clean only Word2Vec caches
FORCE_CLEAN_BERT = False         # clean only BERT fine-tune caches


# =========================
# SECTION 4: UTILS
# =========================
def hash_config(d: dict) -> str:
    """Stable SHA-256 over a dict; order-independent."""
    s = json.dumps(d, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

def hash_code(paths: list[str]) -> str:
    """
    Prefer the current git commit as a succinct run-wide code identity.
    Fallback: deterministically hash *all* relevant source/config files if not in Git.
    """
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
        if sha:
            return f"git:{sha}"
    except (FileNotFoundError, subprocess.CalledProcessError):
        pass

    # Fallback: hash all .py/.yaml/.yml files under repo root + any explicitly-given paths
    root = Path(".").resolve()
    candidates = set()
    for ext in ("*.py", "*.yaml", "*.yml"):
        candidates.update(root.rglob(ext))
    for p in map(Path, paths):
        if p.is_file():
            candidates.add(p.resolve())

    h = hashlib.sha256()
    for file_path in sorted(candidates):
        try:
            h.update(hashlib.sha256(file_path.read_bytes()).digest())
        except OSError:
            continue
    return f"files:{h.hexdigest()}"

# Global code identity used in cache meta
try:
    # Get the absolute path of the currently running script
    this_file = str(Path(__file__).resolve())
except NameError:
    # Fallback for interactive environments where __file__ is not defined
    this_file = "Paper 1.py"

CODE_HASH = hash_code([this_file])
if not str(CODE_HASH).startswith("git:"):
    print("Reminder: Not in a Git repo—using file hashes. Run 'git init' for better tracking.")
else:
    try:
        status = subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()
        if status:
            print("Warning: Git repo has uncommitted changes—commit for full reproducibility.")
    except (subprocess.CalledProcessError, FileNotFoundError):
        # Not a git repo or git not available—safe to ignore for reproducibility hash
        pass

def banner(title: str, width: int = 72) -> None:
    """Prints a centered banner."""
    logging.info("\n" + title.center(width, "="))

def cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Computes cosine similarity between two vectors."""
    return float(cosine_similarity(a.reshape(1, -1), b.reshape(1, -1))[0, 0])

def contains_japanese(text: str) -> bool:
    """Checks if a string contains any Japanese characters."""
    return bool(re_rx.search(r"[\p{Hiragana}\p{Katakana}\p{Han}]", text))

def ensure_dir(p: Path) -> None:
    """Creates a directory if it does not already exist."""
    p.mkdir(parents=True, exist_ok=True)

def nuke_dir(p: Path) -> None:
    """Recursively remove a directory if it exists, ignoring errors."""
    if p.exists():
        shutil.rmtree(p, ignore_errors=True)

def verify_cpu_inference_fixture(model: BertForMaskedLM, tokenizer: BertTokenizer, out_dir: Path) -> None:
    """
    Create-or-verify gold outputs for a tiny probe set on CPU, then restore device/state.
    """
    probe = [
        "ma is not emptiness but relation",
        "tokonoma is understood conceptually not formally",
        "shoji and fusuma indicate spatial flexibility",
        "wabi sabi affirms the ageing of material",
    ]

    out_dir.mkdir(parents=True, exist_ok=True)
    fixtures_path = out_dir / "tests" / "fixtures.json"
    fixtures_path.parent.mkdir(parents=True, exist_ok=True)

    orig_device = next(model.parameters()).device
    was_training = model.training
    try:
        model_cpu = model.to("cpu").eval()

        def _embed(text: str) -> list[float]:
            with torch.no_grad():
                toks = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
                base = model_cpu.get_base_model() if hasattr(model_cpu, "get_base_model") else model_cpu
                if hasattr(base, "bert"):
                    enc_out = base.bert(**toks, output_hidden_states=True, return_dict=True)
                    last = enc_out.last_hidden_state
                else:
                    enc_out = base(**toks, output_hidden_states=True, return_dict=True)
                    last = enc_out.last_hidden_state if hasattr(enc_out, "last_hidden_state") else enc_out.hidden_states[-1]
                cls_vec = last[:, 0, :].squeeze(0).contiguous().cpu().numpy()
                return cls_vec.tolist()

        current = [{"text": t, "vec_sha": sha256(np.asarray(_embed(t), dtype=np.float32).tobytes()).hexdigest()} for t in probe]

        if not fixtures_path.exists():
            with open(fixtures_path, "w", encoding="utf-8") as fh:
                json.dump({"items": current}, fh, ensure_ascii=False, indent=2)
            print(f"[fixture] wrote {fixtures_path} (established gold values)")
        else:
            gold = json.loads(fixtures_path.read_text(encoding="utf-8"))
            items = gold.get("items", [])
            if len(items) != len(current):
                logging.warning("[fixture] probe length mismatch, proceeding with overlapping pairs only")
            for g, c in zip(items, current):
                if g["vec_sha"] != c["vec_sha"]:
                    logging.warning("[fixture] embedding hash mismatch on: " + g["text"])
                    return
            print(f"[fixture] CPU inference matches gold fixtures: {fixtures_path}")
    finally:
        model.to(orig_device)
        model.train(was_training)

def nuke_mode_artifacts(mode_tag: str) -> None:
    """Remove all cached artefacts for a given mode (e.g. FULL or CLEAN)."""
    root = cache_root_for(mode_tag)
    nuke_dir(root)

def sha256_file(path: Path) -> str:
    """Computes the SHA256 hash of a file for integrity checking."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def sha256_dir(path: Path) -> str:
    """Return a deterministic SHA-256 hash across all files in a directory (sorted order)."""
    h = hashlib.sha256()
    if not path.is_dir():
        return ""
    for file_path in sorted(path.rglob("*")):
        if file_path.is_file():
            h.update(hashlib.sha256(file_path.read_bytes()).digest())
    return h.hexdigest()

def save_json(p: Path, obj: dict) -> None:
    """Saves a dictionary object to a JSON file deterministically."""
    ensure_dir(p.parent)
    # Use sort_keys=True and compact separators for a stable, byte-for-byte identical output.
    p.write_text(
        json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
        encoding="utf-8"
    )

def load_json(p: Path) -> Optional[dict]:
    """Loads a dictionary object from a JSON file, returning None if not found."""
    return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else None

def cache_root_for(mode_tag: str) -> Path:
    """Separate cache roots by mode and ensure directory exists."""
    root = MOD / mode_tag.upper()
    root.mkdir(parents=True, exist_ok=True)
    return root

def corpus_fingerprint(corpus_path: Path) -> dict:
    """Creates a dictionary containing a corpus file path and its SHA256 hash."""
    return {"corpus_path": str(corpus_path), "corpus_sha256": sha256_file(corpus_path)}

def safe_w2v_vec(model: Word2Vec, word: str) -> Optional[np.ndarray]:
    """Safely retrieves a word vector from a Word2Vec model, returning None if not found."""
    if model and word in model.wv:
        return model.wv[word]
    return None

@torch.no_grad()
def bert_vec(word: str, model_to_use: BertForMaskedLM, tokenizer: BertTokenizer) -> Optional[np.ndarray]:
    """Gets a robust BERT embedding for a single word by averaging its subword vectors."""
    model_to_use.eval()
    inputs = tokenizer(word, return_tensors="pt")
    inputs = {k: v.to(model_to_use.device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model_to_use(**inputs, output_hidden_states=True)
    last_hidden = outputs.hidden_states[-1]
    word_tokens = tokenizer.tokenize(word)
    if not word_tokens: return None
    start, end = 1, 1 + len(word_tokens)  # skip [CLS]
    emb = last_hidden[0, start:end, :].mean(dim=0).detach().cpu().numpy()
    return emb

def rng_from_tag(tag: str, base_seed: int = SEED) -> tuple[random.Random, np.random.Generator, torch.Generator]:
    """Creates a set of isolated, deterministic random generators from a text tag."""
    h = int(hashlib.sha256(f"{tag}:{base_seed}".encode()).hexdigest(), 16) % (2**32)
    return random.Random(h), np.random.default_rng(h), torch.Generator().manual_seed(h)

def choose_anchor(clusters: Dict[str, int], anchors: set[str]) -> Optional[str]:
    """Pick the first available anchor from a set (sorted for determinism)."""
    avail = sorted(set(clusters) & anchors)
    return avail[0] if avail else None

def _stable_subsample(seq: List, k: int, tag: str) -> List:
    """Deterministically shuffles and subsamples a sequence based on a tag."""
    r, _, _ = rng_from_tag(tag)  # Use the local Random generator
    seq = sorted(list(seq))      # Start from a stable order
    r.shuffle(seq)               # Apply a deterministic shuffle
    return seq[:k]

# --- FUNCTION FOR AUTOMATED REPORTING ---
def generate_summary_report(res_path: Path, mode: str, seed_for_run: int, scores: dict,
                            corpus_path: Optional[Path], model_paths: Dict[str, Path]):
    """Generates a Markdown summary of a pipeline run."""
    report_path = res_path / f"summary_{mode.lower()}.md"

    content = f"# Paper 1 Pipeline Summary: {mode} Mode\n\n"
    content += f"- **Run Seed**: {seed_for_run}\n"

    if corpus_path and corpus_path.is_file():
        content += f"- **Corpus Path**: `{corpus_path}`\n"
        content += f"- **Corpus SHA-256**: `{sha256_file(corpus_path)}`\n\n"
    else:
        content += "- **Corpus**: Not built in this run.\n\n"

    content += "## Model Hashes\n\n"
    for name, path in model_paths.items():
        if path and path.is_file():  # W2V models
            content += f"- **{name} SHA-256**: `{sha256_file(path)}`\n"
        elif path and path.is_dir():  # BERT fine-tuned directory
            content += f"- **{name} (dir hash)**: `{sha256_dir(path)}`\n"
        else:
            content += f"- **{name}**: (missing)\n"
    content += "\n"

    # ARI scores
    ari_scores = scores.get("ari", {})
    if ari_scores:
        for hypo, ari_map in ari_scores.items():
            content += f"### ARI Scores: {hypo}\n\n"
            content += "| Model | ARI Score |\n|-------|-----------|\n"
            sorted_ari = sorted(ari_map.items(), key=lambda item: item[1], reverse=True)
            for model, score in sorted_ari:
                content += f"| {model} | {score:.3f} |\n"
            content += "\n"

    # WordSim
    wordsim_scores = scores.get("wordsim", [])
    if wordsim_scores:
        content += "### WordSim Spearman Correlation\n\n"
        content += "| Dataset | Model | Spearman ρ |\n|---------|-------|------------|\n"
        for row in wordsim_scores:
            content += f"| {row['dataset']} | {row['model']} | {row['spearman_rho']:.3f} |\n"
        content += "\n"

    content += "## Figures\n\n"
    content += f"Figures for this run are in the `figures/` directory corresponding to seed {seed_for_run}.\n"
    content += "Note: Negative ARI values indicate agreement lower than chance (not an error).\n\n"

    report_path.write_text(content, encoding="utf-8")
    print(f"Generated summary report: {report_path}")


def write_machine_snapshot(res_path: Path, mode: str,
                           w2v_corpus_path: Path, bert_corpus_path: Path,
                           model_paths: Dict[str, Path]) -> None:
    """Write a deterministic JSON with seeds, code/config hashes, corpus hashes, and model hashes."""

    payload: Dict[str, Any] = {
        "mode": mode,
        "seed": SEED,
        "code_hash": CODE_HASH,
        "cfg_hash": hash_config(cfg),
        "corpus": {
            "w2v_path": str(w2v_corpus_path),
            "w2v_sha256": sha256_file(w2v_corpus_path) if w2v_corpus_path.is_file() else "",
            "bert_path": str(bert_corpus_path),
            "bert_sha256": sha256_file(bert_corpus_path) if bert_corpus_path.is_file() else "",
        },
        "libs": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "transformers": _tf.__version__,
            "gensim": gensim.__version__,
            "sklearn": sklearn_version,
            "cuda": (torch.version.cuda or "N/A"),
            "device": device.type,
        },
        "models": {},  # filled below
    }

    models: Dict[str, Dict[str, str]] = {}
    for model_name, model_path in model_paths.items():
        if model_path.is_file():
            models[model_name] = {"type": "file", "sha256": sha256_file(model_path)}
        elif model_path.is_dir():
            models[model_name] = {"type": "dir", "sha256": sha256_dir(model_path)}
        else:
            models[model_name] = {"type": "missing", "sha256": ""}

    payload["models"] = models
    save_json(res_path / f"machine_snapshot_{mode.lower()}.json", payload)


# =========================
# SECTION 5: FILTERED TEXT OPS (directly from filters.yaml)
# =========================
def preprocess_text(text: str) -> List[str]:
    """normalise, whitelist chars, to lower, tokenise; optional merge of 're' + following token"""
    t = text
    for k, v in NORMALISATION_MAP.items():
        t = t.replace(k, v)
    toks = CHAR_WHITELIST_RE.sub(" ", t.lower()).split()

    if MERGE_RE_FOLLOWING:
        merged, idx = [], 0
        while idx < len(toks):
            if toks[idx] == "re" and idx + 1 < len(toks):
                # FIX: Semicolon removed
                merged.append(toks[idx] + toks[idx + 1])
                idx += 2
            elif toks[idx] != "re":
                # FIX: Semicolon removed
                merged.append(toks[idx])
                idx += 1
            else:
                idx += 1
        # FIX: The 'toks' variable is now used by the return statement
        toks = merged

    return toks

def token_passes_spacy(tok) -> bool:
    """spaCy-level grammar & noise filter controlled by filters.yaml"""
    if PASS_JAPANESE and contains_japanese(tok.text):  # pass-through for Japanese
        return True
    if tok.pos_ not in ALLOWED_POS:
        return False
    if HONOUR_SPACY_STOP and tok.is_stop:
        return False
    if STOP_SINGLE_LETTERS and len(tok.text) == 1 and tok.text.isalpha():
        return False
    if tok.text in LATIN_UTILITIES:
        return False
    if NUMBER_RE.search(tok.text):
        return False
    return True

def compile_def_regexes(targets: List[str]) -> List[re.Pattern]:
    """Compiles a list of definitional regex patterns for a set of target words."""
    regs: List[re.Pattern] = []
    for t in set(targets):
        for p in DEF_PATTERNS_RAW:
            regs.append(re.compile(p.replace("X", re.escape(t)), flags=re.IGNORECASE))
    return regs

# =========================
# SECTION 6: CORPUS BUILD
# =========================
def _expand_mappings() -> Tuple[Dict[str, str], Dict[str, str]]:
    category_to_era, category_to_loc = {}, {}
    if era_mapping:
        for era, locs in era_mapping.items():
            for _, cats in locs.items():
                for c in cats: category_to_era[c] = era
    if location_mapping:
        for loc, cats in location_mapping.items():
            for c in cats: category_to_loc[c] = loc
    return category_to_era, category_to_loc


def build_corpus_from_single_file(
        corpus_path: Union[str, Path],
        target_words_to_keep: List[str],
        fig_path: Path,
        num_workers: int = 1
) -> Tuple[List[List[str]], List[str], List[dict], List[dict], List[str]]:
    """
    Parse a single-file corpus, apply filters, and return structured data.

    This function reads a text file, splits it into sections based on headings,
    processes each sentence, and filters tokens based on grammatical rules
    while ensuring that specified target words are always retained. It also
    generates a manifest of the corpus structure.
    """
    banner("BUILDING W2V-STYLE CORPUS AND MANIFEST")
    category_to_era, category_to_loc = _expand_mappings()

    with open(corpus_path, encoding="utf-8") as infile:
        full_text = infile.read()

    # Define a regex pattern to identify headings in the text
    header_pattern = re.compile(r'^\s*(?:\d+\.\s*)?([A-Z][a-zA-Z\s\-()&,]+)\s*$', re.MULTILINE)
    sections = header_pattern.split(full_text)

    # Pair headings with their corresponding text blocks
    if len(sections) < 3:
        paired_sections = [("unknown", full_text)]
    else:
        paired_sections = [(sections[section_idx].strip(), sections[section_idx + 1]) for section_idx in
                           range(1, len(sections) - 1, 2)]

    # Prepare sentences and their attributes for processing
    sentences_for_proc, attributes_raw = [], []
    for category, block in paired_sections:
        era = category_to_era.get(category, "unknown")
        loc = category_to_loc.get(category, "unknown")
        source = "academic" if any(s in category.lower() for s in ["journal", "paper"]) else "encyclopedia"

        # Split the text block into sentences and preprocess them
        raw_sents = [s.strip() for s in SENT_SPLIT_RE.split(block) if s.strip()]
        for sent in raw_sents:
            sentences_for_proc.append(" ".join(preprocess_text(sent)))
            attributes_raw.append({
                "category": category, "era": era, "location": loc,
                "source": source, "heading": category
            })

    # Use spaCy's efficient pipeline for language processing
    target_set = {w.lower() for w in target_words_to_keep}
    docs = nlp_en.pipe(sentences_for_proc, n_process=num_workers)

    # Initialize lists to hold the final processed data
    tokenised_sents: List[List[str]] = []
    raw_sents_out: List[str] = []
    attrs_out: List[dict] = []
    token_attrs: List[dict] = []
    flat_tokens: List[str] = []

    # Iterate through the processed documents to filter tokens
    for doc_idx, doc in tqdm(enumerate(docs), total=len(sentences_for_proc), desc="Processing Corpus"):
        filtered_toks: List[str] = []
        for tok in doc:
            # Always keep target words, regardless of other filters
            if KEEP_TARGETS and tok.text.lower() in target_set:
                filtered_toks.append(tok.text)
                continue
            # Apply grammatical and noise filters to other words
            if token_passes_spacy(tok):
                filtered_toks.append(tok.text)

        # If the sentence still has content after filtering, add it to the final lists
        if filtered_toks:
            tokenised_sents.append(filtered_toks)
            raw_sents_out.append(doc.text)
            attrs_out.append(attributes_raw[doc_idx])
            token_attrs.extend([attributes_raw[doc_idx]] * len(filtered_toks))
            flat_tokens.extend(filtered_toks)

    # Create and save a manifest of the corpus for reproducibility
    manifest_rows = [{"heading": a["heading"], "era": a["era"], "location": a["location"], "source": a["source"]} for a
                     in attrs_out]
    pd.DataFrame(manifest_rows).to_csv(fig_path / "corpus_manifest.csv", index=False, encoding="utf-8",
                                       lineterminator="\n", float_format="%.10g")

    return tokenised_sents, flat_tokens, attrs_out, token_attrs, raw_sents_out


def build_bert_corpus(corpus_file_path: Union[str, Path]) -> List[str]:
    """Build a BERT-ready sentence corpus from a single text file with normalisation applied."""
    banner(f"BUILDING BERT CORPUS from: {corpus_file_path}")
    p = Path(corpus_file_path)
    if not p.is_file():
        print("ERROR: BERT corpus file not found:", p)
        return []

    with p.open(encoding="utf-8") as infile:
        text = infile.read()
        # First, apply normalisations
        for k, v in NORMALISATION_MAP.items():
            text = text.replace(k, v)

        # Then, split into sentences using the SAME regex as the other function
        sentences = SENT_SPLIT_RE.split(text)

        # Return a clean list of non-empty sentences
        return [sent.strip() for sent in sentences if sent.strip()]


def plot_zipf(flat_tokens: List[str], fig_path: Path, mode: str) -> None:
    """Generates and saves a Zipf's Law distribution plot."""
    banner("Generating Zipf's Law Plot")
    if not flat_tokens:
        print("Cannot generate Zipf plot: no tokens provided.")
        return

    counts = collections.Counter(flat_tokens)
    sorted_freqs = sorted(counts.values(), reverse=True)

    # --- persist Zipf fit (alpha, R^2) ---
    ranks = np.arange(1, len(sorted_freqs) + 1, dtype=float)
    freqs = np.asarray(sorted_freqs, dtype=float)
    mask = (freqs > 0) & (ranks > 0)
    x = np.log(ranks[mask])
    y = np.log(freqs[mask])
    if x.size >= 2:
        slope, intercept = np.polyfit(x, y, 1)
        y_hat = slope * x + intercept
        ss_res = float(np.sum((y - y_hat) ** 2))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        res_dir = Path(str(fig_path).replace("figures", "results", 1))
        ensure_dir(res_dir)
        _zipf_csv = res_dir / f"zipf_fit_{mode.lower()}.csv"
        pd.DataFrame([{"Zipf_Alpha": float(-slope), "R2": float(r2), "N": int(x.size)}]).to_csv(
            _zipf_csv, index=False, encoding="utf-8", lineterminator="\n", float_format="%.10g"
        )
        print(f"Saved Zipf fit: {_zipf_csv}")

    plt.figure(figsize=(12, 8))
    plt.loglog(range(1, len(sorted_freqs) + 1), sorted_freqs, marker='.')
    plt.title("Zipf's Law Distribution of Tokens")
    plt.xlabel("Token Rank (Log Scale)")
    plt.ylabel("Token Frequency (Log Scale)")
    plt.grid(True, which="both", ls="--")
    plt.tight_layout()
    plt.savefig(fig_path / f"zipf_distribution_{mode}.png", metadata={"Date": None})
    plt.close()
    print(f"Saved Zipf plot to {fig_path / f'zipf_distribution_{mode}.png'}")


# =========================
# SECTION 7: WORD2VEC
# =========================
class LossCallback(CallbackAny2Vec):
    def __init__(self) -> None:
        self.epoch = 0; self.prev = 0.0; self.losses: List[float] = []
    def on_epoch_end(self, model) -> None:
        total = float(model.get_latest_training_loss())
        cur = total - self.prev if self.epoch > 0 else total
        self.losses.append(cur); self.prev = total
        print(f"Epoch {self.epoch+1}, Loss: {cur:.4f}"); self.epoch += 1

def train_w2v(
    sentences_data: List[List[str]],
    sg_flag_local: int,
    config: dict,
    targets_to_keep: List[str]
) -> Tuple[Word2Vec, List[float]]:
    """Trains a Word2Vec model using the integrated constructor."""
    # enforce determinism
    if config.get("workers", 1) != 1:
        print("Warning: Word2Vec with workers > 1 can break determinism. Forcing workers=1.")
        config["workers"] = 1


    loss_cb = LossCallback()
    keep_set = {w.lower() for w in targets_to_keep}

    def trim_rule(word: str, _count: int, _min_count: int):
        """Ensures target words are never trimmed from the vocabulary."""
        return RULE_KEEP if word.lower() in keep_set else RULE_DEFAULT

    model = Word2Vec(
        sentences=sentences_data,
        vector_size=int(config["vector_size"]),
        window=int(config["window"]),
        min_count=int(config["min_count"]),
        workers=int(config["workers"]),
        sg=int(sg_flag_local),
        epochs=int(config["epochs"]),
        sample=float(config.get("sample", 1e-3)),
        seed=SEED,
        hs=1, negative=0,
        compute_loss=True,
        alpha=float(config["alpha"]),
        min_alpha=float(config["min_alpha"]),
        callbacks=[loss_cb],
        trim_rule=trim_rule
    )
    print("W2V sample used:", model.sample)
    return model, loss_cb.losses


def plot_cumulative_loss(cbow_loss: List[float], sg_loss: List[float], fig_path: Path, mode: str) -> None:
    """Plot training loss for CBOW/SG and overlay the CBOW LR schedule (for reference)."""
    if not cbow_loss and not sg_loss:
        return

    epochs = len(cbow_loss) if cbow_loss else len(sg_loss)
    if epochs == 0:
        return

    # Recreate LR schedule from CBOW config (for a consistent overlay)
    initial_alpha = float(cfg.get("w2v_cbow_final", {}).get("alpha", 0.01))
    min_alpha = float(cfg.get("w2v_cbow_final", {}).get("min_alpha", 0.0001))
    lr_schedule = np.linspace(initial_alpha, min_alpha, epochs)

    fig, ax1 = plt.subplots(figsize=(12, 7))
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Training Loss")
    if cbow_loss:
        ax1.plot(range(1, len(cbow_loss) + 1), cbow_loss, "o-", label="CBOW Loss", alpha=0.6)
    if sg_loss:
        ax1.plot(range(1, len(sg_loss) + 1), sg_loss, "o-", label="Skip-gram Loss", alpha=0.6)
    ax1.grid(True, linestyle="--")

    ax2 = ax1.twinx()
    ax2.set_ylabel("Learning Rate")
    ax2.plot(
        range(1, epochs + 1),
        lr_schedule,
        "--",
        label="Learning Rate"
    )
    ax2.tick_params(axis="y")

    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc="best")

    sg_only_hint = " [SG-only; LR from CBOW cfg]" if (not cbow_loss and sg_loss) else ""
    plt.title("Word2Vec Training Loss and Learning Rate Schedule" + sg_only_hint)
    fig.tight_layout()
    out = fig_path / f"cumulative_loss_{mode.lower()}.png"

    # --- persist W2V training summary (final losses) ---
    res_dir = Path(str(fig_path).replace("figures", "results", 1))
    ensure_dir(res_dir)
    _w2v_csv = res_dir / f"w2v_training_summary_{mode.lower()}.csv"
    pd.DataFrame([{
        "CBOW_Final_Loss": float(cbow_loss[-1]) if cbow_loss else float("nan"),
        "SG_Final_Loss": float(sg_loss[-1]) if sg_loss else float("nan"),
        "Epochs": int(epochs)
    }]).to_csv(_w2v_csv, index=False, encoding="utf-8", lineterminator="\n", float_format="%.10g")
    print(f"Saved W2V training summary: {_w2v_csv}")

    plt.savefig(out, metadata={"Date": None})
    plt.close()
    print(f"Saved W2V training loss plot to {out}")


# =========================
# SECTION 8: BERT FINE-TUNING
# =========================
class TextDataset(torch.utils.data.Dataset):
    def __init__(self, encodings): self.encodings = encodings
    def __getitem__(self, idx): return {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
    def __len__(self): return len(self.encodings["input_ids"])


def fine_tune_bert(
        model: BertForMaskedLM,
        tokenizer: BertTokenizer,
        raw_sentences: List[str],
        config: dict,
        fig_path: Path,
        corpus_path: Path,
        mode: str,
        num_epochs: Optional[int] = None,
        validation_split: float = 0.1
) -> Tuple[List[float], List[float], Union[BertForMaskedLM, PeftModel]]:  # <-- Union for return
    """Fine-tune BERT with deterministic touches:
       - Deterministic data order + deterministic masking per-epoch
       - CPU checkpointing of best weights to avoid GPU fragmentation stalls
       - Epoch timing + optional cache clearing (adopts Grok's practical timing idea)
    """

    def _to_device_batch(b: Any) -> Dict[str, torch.Tensor]:
        if isinstance(b, dict): return {k: v.to(device) for k, v in b.items()}
        if isinstance(b, (tuple, list)) and b and isinstance(b[0], dict):
            return {k: v.to(device) for k, v in b[0].items()}
        if isinstance(b, Mapping): return {k: v.to(device) for k, v in b.items()}
        return {k: v.to(device) for k, v in dict(b).items()}

    model.train()

    bert_params = config.get("bert_final", {})
    learning_rate = float(bert_params.get("learning_rate", 3e-5))
    batch_size = int(bert_params.get("per_device_train_batch_size", 32))
    max_length = int(bert_params.get("max_length", 128))
    mlm_prob = float(bert_params.get("mlm_probability", 0.15))
    epochs = int(bert_params.get("num_train_epochs", 3) if num_epochs is None else num_epochs)
    weight_decay = float(bert_params.get("weight_decay", 0.01))
    warmup_ratio = float(bert_params.get("warmup_ratio", 0.06))
    grad_accum_steps = int(bert_params.get("gradient_accumulation_steps", 1))

    train_s, val_s = train_test_split(raw_sentences, test_size=validation_split, random_state=SEED)
    train_enc = tokenizer(train_s, truncation=True, padding=True, max_length=max_length)
    val_enc = tokenizer(val_s, truncation=True, padding=True, max_length=max_length)

    random.seed(SEED)  # Extra seed for masking/tokenizer randomness
    tokenizer.model_max_length = max_length

    train_ds = TextDataset(train_enc)
    val_ds = TextDataset(val_enc)
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=mlm_prob)

    gen_tag = f"dataloader:{sha256_file(corpus_path)}:{SEED}"
    _, _, torch_gen = rng_from_tag(gen_tag)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, collate_fn=collator, num_workers=0, generator=torch_gen
    )
    # change: validation loader also uses the same generator for determinism
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, collate_fn=collator, num_workers=0, generator=torch_gen
    )

    model.to(device)

    # Add LoRA adapter for efficient fine-tuning (avoids overfit)
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        target_modules=["query", "key", "value", "output.dense"],
        task_type=TaskType.FEATURE_EXTRACTION,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()  # Log: ~1% trainable

    # Sanity-check at least one target module got adapted
    matched = sum(1 for n, _ in model.named_modules() if "lora_" in n.lower())
    if matched == 0:
        raise RuntimeError("LoRA adapters not applied; check target_modules names.")

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    num_training_steps = epochs * math.ceil(len(train_loader) / max(1, grad_accum_steps))
    num_warmup_steps = int(num_training_steps * warmup_ratio)
    scheduler = get_scheduler("linear", optimizer=optimizer,
                              num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)

    train_losses: List[float] = []
    val_losses: List[float] = []
    learning_rates_epoch: List[float] = []   # change: new list for per-epoch LR

    best_val = float("inf")
    best_state = None
    best_epoch = None

    print(
        f"Starting BERT fine-tuning: LR={learning_rate}, Batch={batch_size}, "
        f"Epochs={epochs}, Weight Decay={weight_decay}, Grad Accum={grad_accum_steps}"
    )

    for ep in range(epochs):
        # --- Deterministic masking seed per-epoch ---
        h = int(hashlib.sha256(f"mlm:{sha256_file(corpus_path)}:{SEED}:ep{ep+1}".encode()).hexdigest(), 16) % (2**32)
        torch.manual_seed(h)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(h)

        # --- Optional: trim fragmentation and print timings (adopts Grok's idea) ---
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        start_time = time.time()

        model.train()
        total_tr = 0.0
        optimizer.zero_grad(set_to_none=True)

        for step, batch in enumerate(tqdm(train_loader, desc=f"Epoch {ep + 1} [Train]")):
            inputs = _to_device_batch(batch)
            outputs = model(**inputs)
            loss = outputs.loss

            if torch.isfinite(loss):
                total_tr += float(loss.item())
            else:
                print(f"Warning: Skipping non-finite loss in batch {step} of epoch {ep + 1}.")
                continue

            loss /= max(1, grad_accum_steps)
            loss.backward()

            if (step + 1) % max(1, grad_accum_steps) == 0 or (step + 1) == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

        tr_loss = total_tr / len(train_loader) if len(train_loader) > 0 else 0.0
        train_losses.append(tr_loss)

        model.eval()
        total_val = 0.0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {ep + 1} [Val]"):
                inputs = _to_device_batch(batch)
                outputs = model(**inputs)
                if torch.isfinite(outputs.loss):
                    total_val += float(outputs.loss.item())

        val_loss = total_val / len(val_loader) if len(val_loader) > 0 else 0.0
        val_losses.append(val_loss)
        # change: record LR once per epoch, after scheduler has stepped through training
        learning_rates_epoch.append(scheduler.get_last_lr()[0])

        print(f"[Epoch {ep + 1}] Train Loss: {tr_loss:.4f}, Val Loss: {val_loss:.4f}")
        print(f"[Epoch {ep + 1}] Time taken: {time.time() - start_time:.2f} seconds")

        # --- CPU checkpointing for best epoch (prevents GPU fragmentation stalls) ---
        if val_loss < best_val:
            best_val = val_loss
            best_epoch = ep + 1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        # Optional: garbage collect between epochs
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if best_state is not None:
        model.load_state_dict({k: v.to(model.device) for k, v in best_state.items()})
        print(f"Restored best epoch by validation loss: epoch {best_epoch} (val={best_val:.6f})")

    # --- persist BERT training summary (validation min, epoch) ---
    res_dir = Path(str(fig_path).replace("figures", "results", 1))
    ensure_dir(res_dir)
    _bert_csv = res_dir / f"bert_training_summary_{mode.lower()}.csv"
    pd.DataFrame([{
        "Best_Epoch": int(best_epoch),
        "Val_Loss_Min": float(best_val),
        "Train_Loss_Last": float(train_losses[-1] if train_losses else float("nan")),
        "Epochs": int(len(train_losses))
    }]).to_csv(_bert_csv, index=False, encoding="utf-8", lineterminator="\n", float_format="%.10g")
    print(f"Saved BERT training summary: {_bert_csv}")

    # Plot losses + LR
    fig, ax1 = plt.subplots(figsize=(12, 7))
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.plot(range(1, len(train_losses) + 1), train_losses, marker="o", label="Training Loss")
    ax1.plot(range(1, len(val_losses) + 1), val_losses, marker="s", label="Validation Loss")
    ax1.grid(True, linestyle="--")

    ax2 = ax1.twinx()
    ax2.set_ylabel("Learning Rate")
    ax2.plot(
        range(1, len(learning_rates_epoch) + 1),
        learning_rates_epoch,
        "--",
        label="Learning Rate"
    )
    ax2.tick_params(axis="y")

    # Separate legends for clarity
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')

    plt.title("BERT Training, Validation Loss, and Learning Rate Schedule (best epoch restored)")
    fig.tight_layout()
    plt.savefig(fig_path / f"bert_training_loss_{mode.lower()}.png", metadata={"Date": None})
    plt.close()

    return train_losses, val_losses, model


@torch.no_grad()
def get_contextual_bert_vectors(
    word: str,
    model: BertForMaskedLM,
    tokenizer: BertTokenizer,
    corpus_path: Path,
    max_samples: int = 100,
) -> Optional[np.ndarray]:
    """
    Extract a single context-aware vector for `word` by:
      1) Splitting the corpus into sentences using SENT_SPLIT_RE (to avoid line artifacts)
      2) Collecting sentences containing the target (word-boundary aware for non-Japanese)
      3) Deterministically sampling up to max_samples
      4) Averaging final-layer subword embeddings across all matches
    """
    if not corpus_path or not corpus_path.is_file():
        print(f"ERROR: Corpus path for contextual vectors not found at '{corpus_path}'")
        return None

    # 1) Load and split into sentences using the same regex as elsewhere
    text = corpus_path.read_text(encoding="utf-8")
    sentences = [s.strip() for s in SENT_SPLIT_RE.split(text) if s.strip()]

    sent_hash = hashlib.sha256("".join(sorted(sentences)).encode()).hexdigest()[:8]

    # 2) Find sentences containing the word (word-boundaries if not Japanese)
    patt = re.compile(re.escape(word), flags=re.IGNORECASE) if contains_japanese(word) \
            else re.compile(rf"\b{re.escape(word)}\b", flags=re.IGNORECASE)
    hits = [s for s in sentences if patt.search(s)]
    if not hits:
        return None

    # 3) Deterministic subsample
    corpus_sha = sha256_file(corpus_path)
    sampled_hits = _stable_subsample(hits, min(len(hits), max_samples), tag=f"ctx:{word}:{corpus_sha}:{sent_hash}")

    # 4) Encode the literal word once (subword ids)
    word_ids = tokenizer.encode(word, add_special_tokens=False)
    if not word_ids:
        return None

    model.eval()
    ctx_vecs = []
    for s in sampled_hits:
        toks = tokenizer.encode(s, add_special_tokens=True, truncation=True, max_length=512)
        if len(toks) < 3:
            continue

        out = model(torch.tensor([toks]).to(model.device), output_hidden_states=True)
        hidden = out.hidden_states[-1].squeeze(0)  # [seq_len, hidden]

        # Use CLS token for sentence-level nuance (fallback to mean if no subword match)
        emb = hidden[1:-1, :].mean(dim=0).detach().cpu().numpy()  # Mean over non-special tokens
        ctx_vecs.append(emb)

    if not ctx_vecs:
        return None

    return np.mean(np.stack(ctx_vecs), axis=0)


# =========================
# SECTION 9: DEFINITONAL AUDIT (filters.yaml-driven)
# =========================
def run_definitional_audit(
    raw_sentences: List[str],
    targets: List[str],
    figures_dir: Path,
    mode_tag: str,
    bert_model: Optional[BertForMaskedLM] = None,
    bert_tokenizer: Optional[BertTokenizer] = None,
    corpus_path_for_context: Optional[Path] = None,
) -> Tuple[List[str], pd.DataFrame]:
    """
    Audits a corpus for definitional sentences and (optionally) computes definitional
    pair similarities. If bert_model/tokenizer are provided, uses *those* (e.g., fine-tuned)
    instead of the base model.
    """
    # Touch the value so it's considered used (and is helpful in logs)
    if corpus_path_for_context:
        print(f"[audit] contextual corpus: {corpus_path_for_context}")

    banner("DEFINITIONAL AUDIT")
    regs = compile_def_regexes(targets)

    def is_def(s: str) -> bool:
        """Return True if the sentence matches any definitional regex."""
        s_low = s.lower()
        return any(r.search(s_low) for r in regs)


    flags = [is_def(s) for s in raw_sentences]
    df = pd.DataFrame({"sentence": raw_sentences, "is_definitional": flags})
    df.to_csv(figures_dir / f"definitional_presence_{mode_tag}.csv",
              index=False, encoding="utf-8", lineterminator="\n", float_format="%.10g")

    filtered = [s for s, flag in zip(raw_sentences, flags) if not flag]

    # Optional pair similarities
    pairs = cfg.get("definitional_pairs", [])
    if pairs:
        # pick model/tokenizer
        use_tok = bert_tokenizer if bert_tokenizer is not None else bert_tok
        use_mdl = bert_model if bert_model is not None else bert_mod

        out_rows = []
        for a, b in pairs:
            # Use contextual if path provided, else fall back to non-contextual
            if corpus_path_for_context and corpus_path_for_context.is_file():
                va = get_contextual_bert_vectors(a, use_mdl, use_tok, corpus_path_for_context)
                vb = get_contextual_bert_vectors(b, use_mdl, use_tok, corpus_path_for_context)
            else:
                va = bert_vec(a, use_mdl, use_tok)
                vb = bert_vec(b, use_mdl, use_tok)

            if va is None or vb is None:
                continue

            sim_full = cosine(va, vb)
            out_rows.append({"a": a, "b": b, "sim_full": sim_full})

        pd.DataFrame(out_rows).to_csv(
            figures_dir / f"definitional_similarity_{mode_tag}.csv",
            index=False, encoding="utf-8", lineterminator="\n", float_format="%.10g"
        )

    return filtered, df


# =========================
# SECTION 10: CLUSTERING & ARI
# =========================
def generate_clusters(
    embeddings: np.ndarray,
    use_agglomerative: bool,
    l2_normalize: bool,
    k: int = 2,
    config: Optional[dict] = None,
) -> np.ndarray:
    """
    Cluster helper with robust preprocessing and sklearn-version-tolerant agglomerative path.

    - Optional L2 normalisation
    - Optional z-score scaling
    - Agglomerative uses a *precomputed* cosine distance matrix for stability
    - Handles small-n cases by clipping k to the sample size and returning zeros when k==1
    """
    if embeddings is None or len(embeddings) == 0:
        return np.empty((0,), dtype=int)

    params = (config or {}).get("analysis_params", {})
    n_init = int(params.get("kmeans_n_init", 50))
    max_iter = int(params.get("kmeans_max_iter", 1000))
    use_scaling = bool(params.get("use_zscore_scaling", True))

    vectors = np.asarray(embeddings, dtype=np.float32)
    vectors = np.nan_to_num(vectors, copy=False)

    if l2_normalize:
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        vectors = vectors / norms

    vectors_to_cluster = (
        StandardScaler().fit_transform(vectors) if use_scaling else vectors
    )

    n_samples = vectors_to_cluster.shape[0]
    k_adj = int(max(1, min(k, n_samples)))
    if k_adj == 1:
        return np.zeros(n_samples, dtype=int)

    if use_agglomerative:
        # Precompute cosine *distance* matrix (keeps your cosine_distances import in use)
        dist_matrix = cosine_distances(vectors_to_cluster)

        # Version-tolerant kwargs (sklearn >=1.2 uses 'metric', older uses 'affinity')
        common = {"n_clusters": k_adj, "linkage": "average"}
        init_params = AgglomerativeClustering.__init__.__code__.co_varnames  # type: ignore[attr-defined]
        if "metric" in init_params:
            agglo = AgglomerativeClustering(**common, metric="precomputed")
        else:
            agglo = AgglomerativeClustering(**common, affinity="precomputed")  # type: ignore[call-arg]
        return agglo.fit_predict(dist_matrix)

    km = KMeans(n_clusters=k_adj, random_state=SEED, n_init=n_init,
                max_iter=max_iter, init='k-means++')
    return km.fit_predict(vectors_to_cluster)

def kmeans_on_w2v_embeddings(
        terms: List[str],
        model: Word2Vec,
        k: int = 2,
        use_agglomerative: bool = False,
        l2_normalize: bool = False,
        config: Optional[dict] = None
) -> Dict[str, int]:
    """Clusters Word2Vec embeddings using the generic clustering function."""
    valid, vecs = [], []
    for t in terms:
        if t in model.wv:
            valid.append(t)
            vecs.append(model.wv[t])
    if len(valid) < k: return {}

    labels = generate_clusters(np.array(vecs), use_agglomerative, l2_normalize, k, config=config)
    return {t: int(lbl) for t, lbl in zip(valid, labels)}


def kmeans_on_term_embeddings(
        terms: List[str],
        model: BertForMaskedLM,
        tokenizer: BertTokenizer,
        bert_corpus_path: Path,  # <-- ADD THIS ARGUMENT
        k: int = 2,
        use_agglomerative: bool = False,
        l2_normalize: bool = False,
        config: Optional[dict] = None
) -> Dict[str, int]:
    """Clusters BERT embeddings using the generic clustering function."""
    valid, vecs = [], []
    for t in terms:
        # Now this function call has the corpus path it needs
        v = get_contextual_bert_vectors(t, model, tokenizer, bert_corpus_path)
        if v is not None:
            valid.append(t)
            vecs.append(v)
    if len(valid) < k: return {}

    labels = generate_clusters(np.array(vecs), use_agglomerative, l2_normalize, k, config=config)
    return {t: int(lbl) for t, lbl in zip(valid, labels)}


def _ari_on_epoch_for_selection(
    epoch_dir: Path,
    hypothesis: Dict[str, int],
    bert_corpus_path: Path,
    k: int,
    use_agglo: bool,
    use_l2: bool,
) -> Optional[float]:
    if not epoch_dir.is_dir():
        return None
    tok = BertTokenizer.from_pretrained(str(epoch_dir))
    mdl_cpu: BertForMaskedLM = BertForMaskedLM.from_pretrained(str(epoch_dir), attn_implementation="eager")
    mdl = cast(BertForMaskedLM, mdl_cpu.to(device))

    terms = list(hypothesis.keys())
    clusters = kmeans_on_term_embeddings(
        terms, mdl, tok, bert_corpus_path,
        k=k, use_agglomerative=use_agglo, l2_normalize=use_l2, config=cfg
    )
    words_in_common = sorted(set(clusters) & set(hypothesis))
    if len(words_in_common) < 2:
        return None
    model_labels = [clusters[w] for w in words_in_common]
    gold_labels  = [hypothesis[w] for w in words_in_common]
    return float(_ari(gold_labels, model_labels))

def calculate_ari_bootstrap_ci(
    model_clusters: Dict[str, int],
    gold_standard_clusters: Dict[str, int],
    n_bootstraps: int = 1000
) -> Optional[Tuple[float, float, List[float]]]:
    """Return (low, high, scores) for a 95% CI on ARI via deterministic bootstrap over words."""
    words = sorted(model_clusters.keys() & gold_standard_clusters.keys())
    if len(words) < 2:
        return None

    _, npg, _ = rng_from_tag("ari:boot", base_seed=SEED)
    scores: List[float] = []
    for _ in range(n_bootstraps):
        idx = npg.choice(words, size=len(words), replace=True)
        m = [model_clusters[w] for w in idx]
        g = [gold_standard_clusters[w] for w in idx]
        scores.append(_ari(g, m))

    lo = float(np.percentile(scores, 2.5, method="linear"))
    hi = float(np.percentile(scores, 97.5, method="linear"))
    return lo, hi, scores

def _append_row_csv(_path: Path, _row: dict) -> None:
    _path.parent.mkdir(parents=True, exist_ok=True)
    if _path.exists():
        pd.DataFrame([_row]).to_csv(
            _path, mode="a", header=False, index=False,
            encoding="utf-8", lineterminator="\n", float_format="%.10g"
        )
    else:
        pd.DataFrame([_row]).to_csv(
            _path, mode="w", header=True, index=False,
            encoding="utf-8", lineterminator="\n", float_format="%.10g"
        )

def run_ari_permutation_test(model_labels: List[int], gold_labels: List[int], n_shuffles: int = 1000) -> float:
    """Return permutation p-value for ARI by shuffling gold labels with a deterministic RNG."""
    r, _, _ = rng_from_tag("ari:perm", base_seed=SEED)
    actual = _ari(gold_labels, model_labels)
    null_vals: List[float] = []

    shuffled = list(gold_labels)
    for _ in range(n_shuffles):
        r.shuffle(shuffled)
        null_vals.append(_ari(shuffled, model_labels))

    return float(np.sum(np.abs(null_vals) >= np.abs(actual)) / n_shuffles)

def plot_bootstrap_distribution(
        scores: List[float],
        actual_score: float,
        model_name: str,
        hypothesis_name: str,
        fig_path: Path,
        mode: str
) -> None:
    """Generates and saves a histogram of the bootstrap score distribution."""
    plt.figure(figsize=(10, 6))
    sns.histplot(scores, bins=30, kde=True, color="skyblue")
    plt.axvline(actual_score, color='red', linestyle='--', linewidth=2, label=f'Actual ARI = {actual_score:.3f}')

    ci_lower, ci_upper = np.percentile(scores, 2.5), np.percentile(scores, 97.5)
    plt.axvline(float(ci_lower), color='green', linestyle=':', linewidth=2, label=f'95% CI Lower = {ci_lower:.3f}')
    plt.axvline(float(ci_upper), color='green', linestyle=':', linewidth=2, label=f'95% CI Upper = {ci_upper:.3f}')

    plt.title(f"Bootstrap Distribution of ARI Scores\n({model_name} vs. {hypothesis_name})")
    plt.xlabel("Adjusted Rand Index (ARI)")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

    safe_model_name = re.sub(r'[^a-zA-Z0-9_-]', '', model_name)
    safe_hypo_name = re.sub(r'[^a-zA-Z0-9_-]', '', hypothesis_name)
    filename = fig_path / f"bootstrap_dist_{safe_model_name}_{safe_hypo_name}_{mode.lower()}.png"
    plt.tight_layout()
    plt.savefig(filename, dpi=300, transparent=False, metadata={"Date": None})
    plt.close()
    print(f"Saved bootstrap distribution plot to {filename}")


def run_ari_cluster_validation(
    model_clusters: Dict[str, int],
    gold_standard_clusters: Dict[str, int],
    model_name: str,
    hypothesis_name: str,
    res_path: Optional[Path] = None,
    fig_path: Optional[Path] = None,
    mode_tag: Optional[str] = None
) -> Optional[float]:
    """Compute ARI point estimate, optional permutation p-value, optional bootstrap CI, and audit inputs."""
    banner(f"Adjusted Rand Index (ARI): {model_name} vs '{hypothesis_name}'")

    # Align labels on the intersection
    common = sorted(model_clusters.keys() & gold_standard_clusters.keys())
    if len(common) < 2:
        print(f"Insufficient overlap for ARI ({len(common)} words). Skipping.")
        return None
    model_labels = [model_clusters[w] for w in common]
    gold_labels  = [gold_standard_clusters[w] for w in common]

    # Guard against degenerate single-cluster predictions
    if len(set(model_labels)) < 2:
        print(f"Degenerate clustering for '{model_name}' (only 1 cluster). Skipping.")
        return None

    # Audit inputs
    if res_path:
        safe_model = re.sub(r"[^a-zA-Z0-9_-]", "", model_name)
        safe_hypo  = re.sub(r"[^a-zA-Z0-9_-]", "", hypothesis_name)
        (res_path / "ari_inputs").mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            {"word": common, "model_label": model_labels, "gold_label": gold_labels}
        ).to_csv(
            res_path / "ari_inputs" / f"{safe_model}_{safe_hypo}.csv",
            index=False, encoding="utf-8", lineterminator="\n", float_format="%.10g"
        )

    # Point estimate
    score = _ari(gold_labels, model_labels)
    print(f"Compared {len(common)} words. ARI = {score:.4f}")
    p_val = float("nan")

    # Optional permutation p-value
    if RUN_ARI_PERMUTATION_TEST:
        p_val = run_ari_permutation_test(model_labels, gold_labels, n_shuffles=5000)
        print(f"Permutation test p-value: {p_val:.4f}")
        if p_val > 0.05:
            print("Note: p-value > 0.05 — ARI may not be statistically significant.")

    # Optional bootstrap CI
    ci_low = math.nan
    ci_high = math.nan

    if RUN_BOOTSTRAP_ARI_CI:
        print("Bootstrapping ARI for 95% CI ...")
        ci = calculate_ari_bootstrap_ci(
            model_clusters,
            gold_standard_clusters,
            n_bootstraps=2000
        )
        if ci is not None:
            ci_low, ci_high, _scores = ci
            print(f"Bootstrap 95% CI: ({ci_low:.4f}, {ci_high:.4f})")
            _se = (ci_high - ci_low) / (2.0 * 1.96)
            _p_from_ci = float(math.erfc(abs(score) / (_se * 2.0 ** 0.5))) if (
                        math.isfinite(_se) and _se > 0.0) else float("nan")
            if fig_path and mode_tag:
                plot_bootstrap_distribution(
                    _scores, score, model_name, hypothesis_name, fig_path, mode_tag
                )

    # Persist only if we actually have a CI
    if (
            res_path and mode_tag
            and ("ci_low" in locals()) and ("ci_high" in locals())
            and math.isfinite(ci_low) and math.isfinite(ci_high)
    ):
        _boot_out = res_path / f"ari_bootstrap_ci_{mode_tag.lower()}.csv"
        _append_row_csv(_boot_out, {
            "Hypothesis": hypothesis_name,
            "Model": model_name,
            "ARI_Score": float(score),
            "CI_Lower": float(ci_low),
            "CI_Upper": float(ci_high),
            "N_Bootstraps": 2000,
            "P_Value": (
                float(math.erfc(abs(score) / ((ci_high - ci_low) / (2.0 * 1.96) * 2.0 ** 0.5)))
                if (ci_high > ci_low) else float("nan")
            ),
            "P_Value_Perm": float(p_val)
        })

        print(f"Saved ARI bootstrap CIs: {_boot_out}")

    return float(score)


def plot_ari_barchart(ari_map: Dict[str, float], hypothesis_name: str, tag: str, fig_path: Path) -> None:
    """Generates and saves a detailed, annotated bar chart of ARI scores."""
    if not ari_map: return
    df = pd.DataFrame(list(ari_map.items()), columns=["Model", "ARI Score"]).sort_values("ARI Score", ascending=False)
    plt.figure(figsize=(12, 7))

    ax = sns.barplot(x="Model", y="ARI Score", data=df, palette="viridis", hue="Model", legend=False)

    for bar in ax.patches:
        if isinstance(bar, Rectangle):
            ax.annotate(f'{bar.get_height():.3f}', (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        ha='center', va='center', size=10, xytext=(0, 8), textcoords='offset points')
    plt.title(f"Model Alignment with Hypothesis: '{hypothesis_name}' ({tag.upper()})", fontsize=16)
    plt.ylabel("Adjusted Rand Index (ARI)")
    plt.xlabel("Model")
    plt.xticks(rotation=45, ha="right")
    plt.ylim(-1.1, 1.1)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    safe_name = re.sub(r'[^a-z0-9_-]+', '', hypothesis_name.lower())
    output_filename = fig_path / f"ari_confirmatory_{safe_name}_{tag.lower()}.png"
    plt.tight_layout()
    plt.savefig(output_filename, dpi=300, transparent=False, metadata={"Date": None})
    plt.close()
    print(f"Saved ARI comparison plot: {output_filename}")


def plot_word_clusters(
        clusters: Dict[str, int],
        title: str,
        output_filename: Path,
        label_map: Dict[int, str]
) -> None:
    """Generates and saves a bar chart of word clusters."""
    if not clusters:
        print(f"Skipping plot '{title}' due to empty cluster data.")
        return

    df = pd.DataFrame(list(clusters.items()), columns=["Word", "Cluster ID"])
    df["Assigned Concept"] = df["Cluster ID"].map(label_map)
    df["value"] = 1  # Dummy value for bar height
    df = df.sort_values(by=["Assigned Concept", "Word"])

    plt.figure(figsize=(16, 8))
    palette = {
        "Physical": "#529a54",
        "Conceptual": "#3c6382",
        "Cluster 0": "#808080",  # Grey for generic cluster 0
        "Cluster 1": "#A9A9A9"  # A different grey for cluster 1
    }

    sns.barplot(
        x="Word",
        y="value",
        hue="Assigned Concept",
        data=df,
        palette=palette,
        dodge=False
    )

    plt.title(title, fontsize=16)
    plt.xlabel("Target Words", fontsize=12)
    plt.ylabel("")  # Hide the dummy y-axis label
    plt.yticks([])  # Hide the dummy y-axis ticks
    plt.xticks(rotation=45, ha="right")
    plt.legend(title="Assigned Concept")
    plt.tight_layout()
    plt.savefig(output_filename, dpi=300, transparent=False, metadata={"Date": None})
    plt.close()
    print(f"Saved cluster plot: {output_filename}")


# =========================
# SECTION 11: BENCHMARKS (WordSim etc.)
# =========================
# Replace the original function with this one
def _load_wordsim_csv(path: Union[str, Path]) -> pd.DataFrame:
    """
    Loads a WordSim-style CSV by column position to be robust against header name variations.
    It assumes Word1, Word2, and Score are the first three columns.
    """
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        print(f"ERROR: WordSim file not found at {path}")
        return pd.DataFrame()  # Return empty DataFrame to prevent crash

    if len(df.columns) < 3:
        print(f"ERROR: WordSim file '{path}' has fewer than 3 columns. Skipping.")
        return pd.DataFrame()

    # Get the actual column names from their positions (0, 1, 2)
    pos_a, pos_b, pos_score = df.columns[0], df.columns[1], df.columns[2]

    # Standardize the column names for consistent use in the rest of the script
    df = df.rename(columns={
        pos_a: 'Word1',
        pos_b: 'Word2',
        pos_score: 'Score'
    })

    # Return a DataFrame with standardized column names
    return df[['Word1', 'Word2', 'Score']].dropna()


def run_wordsim_permutation_test(human_scores: List[float], model_scores: List[float], n_shuffles: int = 1000) -> float:
    """Calculates a p-value for a Spearman correlation using a deterministic RNG."""
    r, _, _ = rng_from_tag("wordsim:perm", base_seed=SEED)
    actual_rho, _ = spearmanr(human_scores, model_scores)
    shuffled_human = list(human_scores)
    null_distribution = []
    for _ in range(n_shuffles):
        r.shuffle(shuffled_human)
        random_rho, _ = spearmanr(shuffled_human, model_scores)
        null_distribution.append(random_rho)

    p_value = np.sum(np.abs(null_distribution) >= np.abs(actual_rho)) / n_shuffles
    return float(p_value)

def wordsim_bootstrap_ci(golds: np.ndarray, preds: np.ndarray, n_bootstraps: int = 10000) -> Tuple[float, float]:
    """Return 95% bootstrap CI (lo, hi) for Spearman’s rho between golds and preds, preserving global RNG state."""
    idx = np.arange(golds.shape[0], dtype=int)
    _state = np.random.get_state()
    try:
        rs: List[float] = []
        for _ in range(n_bootstraps):
            bs = np.random.choice(idx, size=idx.shape[0], replace=True)
            r, _ = spearmanr(golds[bs], preds[bs])
            if not np.isnan(r):
                rs.append(float(r))
        if not rs:
            return float("nan"), float("nan")
        lo, hi = np.percentile(rs, [2.5, 97.5])
        return float(lo), float(hi)
    finally:
        np.random.set_state(_state)


def evaluate_wordsim(
    w2v_cbow: Optional[Word2Vec],
    w2v_sg: Optional[Word2Vec],
    bert_finetuned: Optional[BertForMaskedLM],
    tokenizer: Optional[BertTokenizer],
    tag: str,
    fig_path: Path,
    res_path: Path
) -> List[Dict]:
    """Benchmark against WordSim sets; save scatter plots and a summary CSV."""
    banner(f"External Benchmark Evaluation (WordSim) for mode: {tag}")
    datasets_cfg = cfg.get("datasets", {})
    results_rows: List[Dict] = []
    ws_boot_rows: List[Dict] = []

    # Helper for plotting a single scatter
    def _plot_scatter(human_scores_, model_scores_, model_name_, dataset_name_, tag_str_):
        if not human_scores_ or not model_scores_:
            return
        rho, _ = spearmanr(human_scores_, model_scores_)
        p_value = run_wordsim_permutation_test(human_scores_, model_scores_)
        results_rows.append({
            "run_mode": tag_str_, "dataset": dataset_name_, "model": model_name_,
            "spearman_rho": float(rho), "p_value": float(p_value)
        })

        # bootstrap CI for this dataset+model
        try:
            _g = np.asarray(human_scores_, dtype=float)
            _p = np.asarray(model_scores_, dtype=float)
            _lo, _hi = wordsim_bootstrap_ci(_g, _p, n_bootstraps=10000)
            ws_boot_rows.append({
                "run_mode": tag_str_,
                "Set": str(dataset_name_),
                "Model": str(model_name_),
                "CI_Lower": float(_lo),
                "CI_Upper": float(_hi),
                "N_Bootstraps": 10000
            })
        except Exception as _e:
            logging.warning(f"[wordsim] bootstrap CI skipped for {dataset_name_} / {model_name_}: {_e}")

        plt.figure(figsize=(9, 9))
        plt.scatter(human_scores_, model_scores_, alpha=0.5, label=f"{model_name_}")
        lo, hi = min(human_scores_ + model_scores_), max(human_scores_ + model_scores_)
        plt.plot([lo, hi], [lo, hi], "--", label="Ideal correlation")
        plt.title(f"{dataset_name_}: human vs. {model_name_}\n(ρ={rho:.3f}, p={p_value:.3f})")
        plt.xlabel("Human similarity (normalised)")
        plt.ylabel(f"{model_name_} cosine similarity")
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.legend()
        plt.tight_layout()
        fname = fig_path / (
            f"wordsim_scatter_{dataset_name_.lower().replace(' ', '_')}_"
            f"{model_name_.lower().replace(' ', '_').replace('(', '').replace(')', '')}_"
            f"{tag_str_}.png"
        )
        plt.savefig(fname, metadata={"Date": None})
        plt.close()
        print(f"Saved WordSim scatter: {fname}")

    # Load datasets
    datasets_to_load = {
        "WordSim353": datasets_cfg.get("wordsim353_path"),
        "WordSim353_JA-EN": datasets_cfg.get("wordsim353_ja_en_path"),
        "Domain-Specific": datasets_cfg.get("arch_domain_path"),
    }
    loaded_dfs: Dict[str, pd.DataFrame] = {}
    for name, p in datasets_to_load.items():
        if p and Path(p).is_file():
            df = _load_wordsim_csv(Path(p))
            if not df.empty:
                max_score = df["Score"].max()
                if max_score and max_score > 1.0:
                    df["Score"] = df["Score"] / max_score
                loaded_dfs[name] = df

    # Prepare a base (un-fine-tuned) BERT and its tokenizer once for speed
    base_tok: BertTokenizer = BertTokenizer.from_pretrained(
        "bert-base-multilingual-cased", revision=hf_revision
    )
    base_bert: BertForMaskedLM = BertForMaskedLM.from_pretrained(
        "bert-base-multilingual-cased", revision=hf_revision, attn_implementation="eager"
    )

    base_bert.to(device)

    # Vector getters per model
    models_to_test = {
        "Word2Vec CBOW": (lambda w: w2v_cbow.wv[w] if (w2v_cbow and w in w2v_cbow.wv) else None),
        "Word2Vec Skip-gram": (lambda w: w2v_sg.wv[w] if (w2v_sg and w in w2v_sg.wv) else None),
        "BERT (Base)": (lambda w: bert_vec(w, base_bert, base_tok)),
        "BERT (Fine-Tuned)": (lambda w: bert_vec(w, bert_finetuned, tokenizer or base_tok)),
    }

    for model_name, vec_fn in models_to_test.items():
        if "Word2Vec" in model_name and (w2v_cbow is None and w2v_sg is None):
            continue
        if model_name == "BERT (Fine-Tuned)" and bert_finetuned is None:
            continue

        for dataset_name, df in loaded_dfs.items():
            human_scores, model_scores = [], []
            for _, row in df.iterrows():
                v1 = vec_fn(row["Word1"])
                v2 = vec_fn(row["Word2"])
                if v1 is not None and v2 is not None:
                    human_scores.append(float(row["Score"]))  # type: ignore
                    model_scores.append(cosine(np.asarray(v1), np.asarray(v2)))
            if not model_scores:
                print(f"Warning: No valid pairs for {model_name} on {dataset_name} (possible OOV words). Skipping.")
                continue
            _plot_scatter(human_scores, model_scores, model_name, dataset_name, tag)

    if results_rows:
        out_csv = res_path / "wordsim_scores_paper1.csv"
        pd.DataFrame(results_rows).to_csv(out_csv, index=False, encoding="utf-8", lineterminator="\n",
                                          float_format="%.10g")
        print(f"Saved WordSim summary: {out_csv}")

    # persist WordSim bootstrap CIs
    if ws_boot_rows:
        _ws_boot = res_path / f"wordsim_bootstrap_ci_{tag}.csv"
        pd.DataFrame(ws_boot_rows).to_csv(
            _ws_boot, index=False, encoding="utf-8", lineterminator="\n", float_format="%.10g"
        )
        print(f"Saved WordSim bootstrap CIs: {_ws_boot}")

    return results_rows


# =========================
# SECTION 12: VISUALS FOR PAPER 1
# =========================
def plot_top_cooccurrences(tokenised: List[List[str]], targets: List[str], fig_path: Path, mode: str, window: int = 5,
                           top_n: int = 10) -> None:
    """Finds and plots the top N co-occurring words for each target word."""
    banner("Top Co-occurring Words")
    targets = [t.lower() for t in targets]
    target_set = set(targets)
    counts = {t: collections.Counter() for t in targets}
    for sent in tokenised:
        for tok_idx, tok in enumerate(sent):
            if tok in target_set:
                start = max(0, tok_idx - window)
                end = min(len(sent), tok_idx + window + 1)
                # Corrected context window creation
                ctx = sent[start:tok_idx] + sent[tok_idx + 1:end]
                # Filter out other target words from the context
                ctx_filtered = [w for w in ctx if w not in target_set]
                counts[tok].update(ctx_filtered)
    plot_rows = []
    for t, c in counts.items():
        for w, k in c.most_common(top_n):
            plot_rows.append({"target": t, "word": w, "count": k})

    if not plot_rows: return

    df = pd.DataFrame(plot_rows)

    res_dir = Path(str(fig_path).replace("figures", "results", 1))
    ensure_dir(res_dir)
    _co_csv = res_dir / f"cooccurrences_top_{top_n}_{mode.lower()}.csv"
    df.to_csv(_co_csv, index=False, encoding="utf-8", lineterminator="\n", float_format="%.10g")
    print(f"Saved co-occurrence table: {_co_csv}")

    targets_unique = sorted(df["target"].unique())
    fig_h = 4 * max(1, len(targets_unique))
    plt.figure(figsize=(10, fig_h))

    for idx, t in enumerate(targets_unique, 1):
        # EXPLICIT WRAPPER to satisfy the linter
        target_df = pd.DataFrame(df[df["target"] == t])
        if not target_df.empty:
            sub = target_df.sort_values(by="count")
            plt.subplot(len(targets_unique), 1, idx)
            plt.barh(sub["word"], sub["count"])
            plt.title(f"Top Co-occurring Words – {t}")
            plt.xlabel("Co-occurrence Count")
            plt.ylabel("Word")

    plt.tight_layout()
    plt.savefig(fig_path / f"top_cooccurrences_{mode.lower()}.png", metadata={"Date": None})  # <-- Always append
    plt.close()


def _build_bert_vocab_from_counts(
        counts: collections.Counter,
        tokenizer: BertTokenizer,
        targets: List[str]
) -> List[str]:
    """Builds a clean vocabulary for BERT similarity search from token counts."""
    target_set = {t.lower() for t in targets}
    vocab_all = [w for w, _ in counts.most_common(5000)]
    tok_vocab = tokenizer.get_vocab()

    def keep(w: str, cnt: int) -> bool:
        """Applies filtering rules to decide if a word is a valid vocabulary candidate."""
        if w not in tok_vocab: return False
        if w.startswith("##"): return False
        if w.lower() in BERT_EXCLUDE: return False
        if not w.isalpha(): return False
        if cnt < BERT_MIN_FREQ: return False
        if len(w) <= 2 and w.lower() not in target_set: return False
        return True

    return [w for w in vocab_all if keep(w, counts[w])]


def plot_top_n_similarities_word2vec(model: Word2Vec, name: str, targets: List[str], fig_path: Path,
                                     mode: str, top_n: int = 10) -> None:
    """Finds and plots the top N similar words for a Word2Vec model."""
    banner(f"Top-{top_n} Similarities – {name}")
    rows = []
    present = [t for t in targets if t in model.wv]
    for t in present:
        for w, s in model.wv.most_similar(t, topn=top_n):
            rows.append({"target": t, "word": w, "similarity": s})

    if not rows: return

    df = pd.DataFrame(rows)

    res_dir = Path(str(fig_path).replace("figures", "results", 1))
    ensure_dir(res_dir)
    _sim_csv = res_dir / f"top_similarities_{name.lower().replace(' ', '_')}_{mode.lower()}.csv"
    df.to_csv(_sim_csv, index=False, encoding="utf-8", lineterminator="\n", float_format="%.10g")
    print(f"Saved top-N similarities ({name}) table: {_sim_csv}")

    targets_unique = sorted(df["target"].unique())
    fig_h = 4 * max(1, len(targets_unique))
    plt.figure(figsize=(10, fig_h))

    for idx, t in enumerate(targets_unique, 1):
        # EXPLICIT WRAPPER to satisfy the linter
        target_df = pd.DataFrame(df[df["target"] == t])
        if not target_df.empty:
            sub = target_df.sort_values(by="similarity")
            plt.subplot(len(targets_unique), 1, idx)
            plt.barh(sub["word"], sub["similarity"])
            plt.title(f"Top {top_n} Similarities ({name}) – {t}")
            plt.xlabel("Cosine Similarity")
            plt.ylabel("Word")

    plt.tight_layout()
    plt.savefig(fig_path / f"top_similarities_{name.lower().replace(' ', '_')}_{mode.lower()}.png", dpi=300,
                transparent=False,
                metadata={"Date": None})
    plt.close()


def plot_top_n_similarities_bert(
        model: BertForMaskedLM,
        tokenizer: BertTokenizer,
        name: str,
        targets: List[str],
        flat_tokens: List[str],
        fig_path: Path,
        corpus_path_for_context: Path,
        mode: str,
        top_n: int = 10
) -> None:
    """Finds and plots the top N similar words for a BERT model."""
    banner(f"Top-{top_n} Similarities – {name}")
    counts = collections.Counter(flat_tokens)
    vocab = _build_bert_vocab_from_counts(counts, tokenizer, targets)
    if not vocab: return

    # Pre-compute candidate vectors (this is fast, remains unchanged)
    vecs: List[np.ndarray] = []
    for w in vocab:
        v = bert_vec(w, model, tokenizer)
        vecs.append(v if v is not None else np.zeros((model.config.hidden_size,), dtype=np.float32))
    vocab_vecs = np.vstack(vecs)

    rows = []
    for t in targets:
        # <-- This is the key change to use the contextual method
        tv = get_contextual_bert_vectors(t, model, tokenizer, corpus_path_for_context)

        if tv is None: continue
        sims = cosine_similarity(tv.reshape(1, -1), vocab_vecs)[0]
        top_idx = np.argsort(sims)[::-1]
        out = []
        for idx in top_idx:
            w = vocab[idx]
            if w.lower() == t.lower(): continue
            out.append((w, float(sims[idx])))
            if len(out) == top_n: break
        for w, s in out:
            rows.append({"target": t, "word": w, "similarity": s})

    # The rest of the plotting logic is perfect and remains unchanged
    if not rows: return
    df = pd.DataFrame(rows)

    res_dir = Path(str(fig_path).replace("figures", "results", 1))
    ensure_dir(res_dir)
    _sim_csv = res_dir / f"top_similarities_{name.lower().replace(' ', '_')}_{mode.lower()}.csv"
    df.to_csv(_sim_csv, index=False, encoding="utf-8", lineterminator="\n", float_format="%.10g")
    print(f"Saved top-N similarities ({name}) table: {_sim_csv}")

    targets_unique = sorted(df["target"].unique())
    fig_h = 4 * max(1, len(targets_unique))
    plt.figure(figsize=(10, fig_h))

    for idx, t in enumerate(targets_unique, 1):
        target_df = pd.DataFrame(df[df["target"] == t])
        if not target_df.empty:
            sub = target_df.sort_values(by="similarity")
            plt.subplot(len(targets_unique), 1, idx)
            plt.barh(sub["word"], sub["similarity"])
            plt.title(f"Top {top_n} Similarities ({name}) – {t}")
            plt.xlabel("Cosine Similarity")
            plt.ylabel("Word")

    plt.tight_layout()
    plt.savefig(fig_path / f"top_similarities_{name.lower().replace(' ', '_')}_{mode.lower()}.png",
                metadata={"Date": None})  # <-- Always append
    plt.close()


def plot_tsne(
    embeddings: Dict[str, np.ndarray],
    clusters: Dict[str, int],
    label_map: Dict[int, str],
    title: str,
    fig_path: Path,
    filename: str,
    config: Optional[dict] = None,
):
    """
    Robust t-SNE plot:
    - L2 normalise → PCA (to pca_init_dims or n-1) → t-SNE
    - Perplexity derived from sample size with safety checks
    - Works on small samples (skips gracefully if not enough)
    """
    # Need at least 3 points to draw something meaningful
    if len(embeddings) < 3:
        print(f"Skipping t-SNE plot '{title}': requires ≥ 3 vectors.")
        return

    params = (config or {}).get("tsne_params", {})
    pca_dims = int(params.get("pca_init_dims", 20))
    perp_div = int(params.get("perplexity_divisor", 2))  # e.g., N-1 // 2

    labels = sorted(embeddings.keys())
    vectors = np.array([embeddings[w] for w in labels], dtype=np.float32)

    # 1) L2 normalise
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1
    vectors_norm = vectors / norms

    # 2) PCA to a safe dimension
    n_for_pca = max(1, min(pca_dims, len(labels) - 1))
    vectors_pca = PCA(n_components=n_for_pca, random_state=SEED).fit_transform(vectors_norm)

    # 3) Perplexity: must be < n_samples
    perplexity_value = max(2, (len(labels) - 1) // max(1, perp_div))
    if len(labels) <= perplexity_value:
        print(f"Skipping t-SNE plot '{title}': n={len(labels)} ≤ perplexity={perplexity_value}.")
        return

    tsne = TSNE(perplexity=perplexity_value, random_state=SEED, method='exact')
    vectors_2d = tsne.fit_transform(vectors_pca)

    # Colour by mapped labels
    palette = {"Physical": "#529a54", "Conceptual": "#3c6382", "Unassigned": "#808080"}
    colours = []
    for w in labels:
        cid = clusters.get(w)
        concept = label_map.get(cid)
        colours.append(palette.get(concept, palette["Unassigned"]))

    plt.figure(figsize=(15, 15))
    plt.scatter(vectors_2d[:, 0], vectors_2d[:, 1], c=colours, alpha=0.95, s=60)
    for i, lab in enumerate(labels):
        plt.annotate(lab, (vectors_2d[i, 0], vectors_2d[i, 1]),
                     textcoords="offset points", xytext=(0, 10), ha="center", fontsize=12)

    plt.title(title, fontsize=16)
    plt.xlabel("t-SNE component 1", fontsize=12)
    plt.ylabel("t-SNE component 2", fontsize=12)

    legend_handles = [
        plt.Line2D([0], [0], marker="o", color="w", label=lab, markerfacecolor=col, markersize=12)
        for lab, col in palette.items() if lab != "Unassigned"
    ]
    plt.legend(title="Assigned Concept", handles=legend_handles, fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(fig_path / filename, dpi=300, transparent=False, metadata={"Date": None})
    plt.close()
    print(f"Saved t-SNE plot: {fig_path / filename}")


# =========================
# SECTION 13: CACHING & LOGS
# =========================

def resolve_active_paths_from_yaml(config_data: dict, mode_override: str = None) -> dict:
    """Resolves which corpus files to use based on the run mode (FULL vs CLEAN)."""
    mode = (mode_override or str(config_data.get("rq1a_mode", "FULL"))).upper()
    base = Path(config_data["corpus_dir"]) if config_data.get("corpus_dir") else Path(".")
    if mode == "CLEAN":
        w2v_path = base / config_data.get("corpus_file_w2v_cl", config_data.get("corpus_file_w2v", "W2V_Corpus.txt"))
        bert_path = base / config_data.get("corpus_file_bert_cased_cl", config_data.get("corpus_file_bert_cased", "BERT_Corpus.txt"))
    else:
        w2v_path = base / config_data.get("corpus_file_w2v", "W2V_Corpus.txt")
        bert_path = base / config_data.get("corpus_file_bert_cased", "BERT_Corpus.txt")
    return {"tag": mode, "w2v": w2v_path, "bert": bert_path}


def maybe_train_or_load_w2v_pair(
    sentences,
    cbow_cfg,
    sg_cfg,
    mode_tag,
    corpus_path,
    targets,
):
    """
    Load cached Word2Vec (CBOW, SG) if valid, otherwise train and cache.

    Cache validity requires three matches:
    - corpus_sha256 (data)
    - cfg_hash      (effective hyper-params for both models)
    - code_hash     (git commit or file hash bundle)

    Returns:
        (cbow_model, sg_model, cbow_loss_list, sg_loss_list)
        If training is disabled / sentences are empty and no valid cache exists,
        returns (None, None, [], []) so the caller can decide what to do
        (e.g., raise in EVAL-ONLY mode).
    """
    root = cache_root_for(mode_tag)
    meta_path = root / "w2v.meta.json"
    cbow_path = root / "w2v_cbow.model"
    sg_path   = root / "w2v_sg.model"

    # Build fingerprint
    fp = corpus_fingerprint(corpus_path)
    fp["cfg_hash"]  = hash_config({"cbow": cbow_cfg, "sg": sg_cfg})
    fp["code_hash"] = CODE_HASH

    # 1) Respect clean-run flags (note: W2V has its own clean flag)
    force_clean = bool(globals().get("FORCE_CLEAN_RUN", False) or globals().get("FORCE_CLEAN_W2V", False))

    # 2) Try to load a valid cache
    if not force_clean and meta_path.is_file():
        meta = load_json(meta_path)
        cache_valid = (
            meta is not None
            and meta.get("corpus_sha256") == fp.get("corpus_sha256")
            and meta.get("cfg_hash")      == fp.get("cfg_hash")
            and meta.get("code_hash")     == fp.get("code_hash")
            and (cbow_path.is_file() and sg_path.is_file())
        )
        if cache_valid:
            try:
                print(f"[{mode_tag}] Loading cached W2V models…")
                return Word2Vec.load(str(cbow_path)), Word2Vec.load(str(sg_path)), [], []
            except Exception as e:
                # Corrupted cache – fall through to retrain if possible
                print(f"[{mode_tag}] W2V cache present but failed to load ({e}). Will retrain if possible.")

    # 3) If we can’t/shouldn’t train (no sentences or empty cfg), signal missing cache
    if not sentences or not cbow_cfg or not sg_cfg:
        print(f"[{mode_tag}] No valid W2V cache and training inputs unavailable. Skipping training.")
        return None, None, [], []

    # 4) Train from scratch
    print(f"[{mode_tag}] Training W2V CBOW/SG from scratch…")
    cbow_model, cbow_loss = train_w2v(sentences, 0, cbow_cfg, targets)
    sg_model,   sg_loss   = train_w2v(sentences, 1, sg_cfg, targets)

    # 5) Save models + metadata
    ensure_dir(root)
    cbow_model.save(str(cbow_path))
    sg_model.save(str(sg_path))

    fp_to_save = {
        "corpus_path":   fp.get("corpus_path"),
        "corpus_sha256": fp.get("corpus_sha256"),
        "cfg_hash":      fp.get("cfg_hash"),
        "code_hash":     fp.get("code_hash"),
        "trained":       True,
    }
    save_json(meta_path, fp_to_save)

    return cbow_model, sg_model, cbow_loss, sg_loss


def maybe_finetune_or_load_bert(
    sentences,
    cfg_all,
    mode_tag: str,
    corpus_path: Path,
    base_model: Optional[BertForMaskedLM],
    base_tokenizer: Optional[BertTokenizer],
    fig_path: Path,
) -> tuple[BertTokenizer, Union[BertForMaskedLM, PeftModel], list[float], list[float]]:  # <-- Explicit Union
    """
    Load a cached fine-tuned BERT if valid; otherwise fine-tune and cache it.
    If no valid cache and base_* are None, raise FileNotFoundError (so callers
    can decide whether to enable training or set RUN_EVAL_ONLY=True).
    """
    root = cache_root_for(mode_tag)
    out_dir = root / "bert_cased"
    meta_path = out_dir / "meta.json"
    log_path = out_dir / "training_logs.json"

    # 1) Build expected fingerprint for this run
    fp: dict = {}
    if corpus_path and corpus_path.is_file():
        fp = corpus_fingerprint(corpus_path)
        fp["cfg_hash"] = hash_config(cfg_all.get("bert_final", {}))
        fp["code_hash"] = CODE_HASH
        fp["seed"] = SEED

    # Helper: validate cache (avoid shadowing outer names)
    def _is_valid_bert_cache(
        meta_info: Optional[dict], expected_fingerprint: dict, model_dir_path: Path
    ) -> tuple[bool, str]:
        """
        Validate that:
          - model_dir exists
          - model/tokenizer files exist
          - fingerprints (corpus/cfg/code) match and fine_tuned=True
        Returns (is_valid, reason_if_invalid).
        """
        if not model_dir_path.is_dir():
            return False, "model_dir_missing"

        has_config = (model_dir_path / "config.json").is_file()
        has_weights = (model_dir_path / "pytorch_model.bin").is_file() or (model_dir_path / "model.safetensors").is_file()
        has_tokenizer = any(
            (model_dir_path / name).is_file()
            for name in ("tokenizer.json", "tokenizer_config.json", "vocab.txt")
        )
        if not (has_config and has_weights and has_tokenizer):
            missing = []
            if not has_config:    missing.append("config.json")
            if not has_weights:   missing.append("pytorch_model.bin/model.safetensors")
            if not has_tokenizer: missing.append("tokenizer files")
            return False, f"missing_files:{','.join(missing)}"

        if not meta_info:
            return False, "meta_missing"

        fp_ok = (
            meta_info.get("corpus_sha256") == expected_fingerprint.get("corpus_sha256")
            and meta_info.get("cfg_hash")  == expected_fingerprint.get("cfg_hash")
            and meta_info.get("code_hash") == expected_fingerprint.get("code_hash")
            and meta_info.get("fine_tuned") is True
        )
        if not fp_ok:
            return False, "fingerprint_mismatch"

        return True, "ok"

    # 2) Use cache unless forced clean
    force_clean = bool(globals().get("FORCE_CLEAN_RUN", False) or globals().get("FORCE_CLEAN_BERT", False))
    if not force_clean and meta_path.is_file():
        meta_from_disk = load_json(meta_path)
        is_ok, reason = _is_valid_bert_cache(meta_from_disk, fp, out_dir)
        if is_ok:
            print(f"[{mode_tag}] Loading valid cached fine-tuned BERT from {out_dir} …")
            tok = BertTokenizer.from_pretrained(str(out_dir))
            mdl_cpu: Union[BertForMaskedLM, PeftModel] = BertForMaskedLM.from_pretrained(str(out_dir), attn_implementation="eager")  # <-- Union annotation

            # Load LoRA adapters if used (check meta)
            if meta_from_disk and meta_from_disk.get("lora_used", False):
                mdl_cpu = PeftModel.from_pretrained(mdl_cpu, str(out_dir))

            mdl = cast(Union[BertForMaskedLM, PeftModel], mdl_cpu.to(device))  # <-- Cast to union after to(device)

            losses_json = load_json(log_path) or {}
            return (
                tok,
                mdl,
                list(losses_json.get("train_losses", [])),
                list(losses_json.get("val_losses", [])),
            )
        else:
            print(f"[{mode_tag}] BERT cache present but invalid: {reason}. Will retrain if allowed.")

    # 3) No valid cache. EVAL-ONLY means we must not train.
    if RUN_EVAL_ONLY:
        raise FileNotFoundError(
            f"[{mode_tag}] EVAL-ONLY is set but a valid BERT cache was not found at {out_dir}."
        )

    # 4) We intend to (re)train; require base_*.
    if base_model is None or base_tokenizer is None:
        raise FileNotFoundError(
            f"[{mode_tag}] No valid cache and base model/tokenizer not provided. "
            f"Enable RUN_BERT_FINETUNING or ensure a cache exists (or set RUN_EVAL_ONLY=True)."
        )

    print(f"[{mode_tag}] Fine-tuning BERT (cache missing or invalid)…")

    train_losses, val_losses, mdl = fine_tune_bert(
        base_model, base_tokenizer, sentences, cfg_all,
        fig_path=fig_path, corpus_path=corpus_path, mode=mode_tag
    )

    # 5) Save freshly trained model + metadata (LoRA-compatible)
    ensure_dir(out_dir)

    # Explicitly save base config to avoid HF fetch warnings during load
    config = BertConfig.from_pretrained("bert-base-multilingual-cased", revision=hf_revision, local_files_only=True)
    config.save_pretrained(str(out_dir))

    mdl.save_pretrained(str(out_dir))  # Saves base + LoRA adapter files
    base_tokenizer.save_pretrained(str(out_dir))  # Unchanged

    fp["fine_tuned"] = True
    fp["best_epoch"] = (int(np.argmin(val_losses)) + 1) if val_losses else None
    fp["lora_used"] = True  # Track for loading
    save_json(meta_path, fp)

    log_data = {"train_losses": train_losses, "val_losses": val_losses, "best_epoch": fp["best_epoch"]}
    save_json(log_path, log_data)

    return base_tokenizer, mdl, train_losses, val_losses

def log_reproducibility_info(output_dir: Path, sections: Dict[str, dict]) -> None:
    """Saves a log file with library versions and key config parameters."""
    output_dir.mkdir(exist_ok=True, parents=True)
    log_path = output_dir / "reproducibility_log.txt"
    with open(log_path, "w", encoding="utf-8") as log_file:
        log_file.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        log_file.write(f"Python: {sys.version.split()[0]}\n")
        log_file.write(f"Platform: {platform.platform()}\n")
        log_file.write(f"PyTorch: {torch.__version__}\n")

        # Log CUDA and device information for full hardware reproducibility.
        log_file.write(f"CUDA Version: {torch.version.cuda if torch.cuda.is_available() else 'N/A'}\n")
        log_file.write(f"Device: {device.type}\n")

        log_file.write(f"Transformers: {transformers.__version__}\n")
        log_file.write(f"Gensim: {gensim.__version__}\n")
        log_file.write("\n--- Config Sections ---\n")
        for name, section in sections.items():
            log_file.write(f"[{name}]\n")
            for k, v in section.items(): log_file.write(f"{k}: {v}\n")
            log_file.write("\n")
    print(f"Reproducibility log saved to {log_path}")

# =========================
# SECTION 14: MAIN PIPELINE
# =========================
def main(fig_path: Path, res_path: Path, seed_for_run: int) -> None:
    """
    Main execution pipeline for Paper 1.

    This function orchestrates the entire workflow for the paper's analysis,
    running in both "FULL" and "CLEAN" modes to produce all necessary
    models, evaluations, and figures for the definitional audit.
    """
    global RUN_CORPUS_BUILD, RUN_W2V_TRAINING, RUN_BERT_FINETUNING
    banner("PAPER 1 PIPELINE")

    if RUN_EVAL_ONLY:
        print(">> EVALUATION-ONLY MODE: Training and corpus build will be skipped. <<")
        RUN_CORPUS_BUILD = False
        RUN_W2V_TRAINING = False
        RUN_BERT_FINETUNING = False

    modes_to_run = ["FULL", "CLEAN"]
    log_reproducibility_info(res_path, {
        "w2v_cbow_final": cfg.get("w2v_cbow_final", {}),
        "w2v_sg_final": cfg.get("w2v_sg_final", {}),
        "bert_final": cfg.get("bert_final", {}),
    })

    ari_hypotheses = cfg.get("ari_settings", {}).get("paper1_confirmatory", {}).get("hypotheses", {})
    if not ari_hypotheses and RUN_ARI_VALIDATION:
        print("Warning: No 'ari_hypotheses' found in config.yaml; skipping ARI evaluations.")

    for mode in modes_to_run:
        banner(f"RUN MODE: {mode}")

        # --- 1) Determine Paths and Target Words (Done Once) ---
        active_paths = resolve_active_paths_from_yaml(cfg, mode_override=mode)
        w2v_corpus_path = Path(active_paths["w2v"])
        bert_corpus_path = Path(active_paths["bert"])

        raw_terms_used = (
            ALL_TARGET_WORDS_FROM_CONFIG if cfg.get("word_list_to_use", "all") == "all"
            else CORE_TARGET_WORDS_FROM_CONFIG
        )
        terms_used = [t.lower() for t in raw_terms_used]

        # --- 2) Build Corpora (or skip if not needed) ---
        if RUN_CORPUS_BUILD:
            print("Building corpora from source files...")
            tokenised, flat_tokens, _, _, raw_sentences = build_corpus_from_single_file(
                w2v_corpus_path, target_words_to_keep=terms_used, fig_path=fig_path, num_workers=1
            )
            print(f"[{mode}] W2V Corpus size: sentences={len(tokenised)} tokens={len(flat_tokens)}")
            bert_raw = build_bert_corpus(bert_corpus_path)
        else:
            tokenised, flat_tokens, raw_sentences = [], [], []
            if bert_corpus_path.is_file():
                bert_raw = build_bert_corpus(bert_corpus_path)
            else:
                bert_raw = []
                print(f"[WARNING] BERT corpus file not found at {bert_corpus_path}. Some evaluation steps may fail.")
        print(f"[{mode}] BERT Corpus size: sentences={len(bert_raw)}")

        # --- 3) Word2Vec Training / Loading ---
        if RUN_W2V_TRAINING and tokenised:
            if FORCE_CLEAN_RUN or FORCE_CLEAN_W2V:
                print(f"[CLEAN] Nuking W2V cache for mode {mode} …")
                try:
                    nuke_mode_artifacts(mode)
                except (OSError, PermissionError) as e:
                    print(f"[CLEAN] Warning: could not remove caches for mode {mode}: {e}")

            w2v_cbow, w2v_sg, cbow_losses, sg_losses = maybe_train_or_load_w2v_pair(
                tokenised, cfg["w2v_cbow_final"], cfg["w2v_sg_final"], mode, w2v_corpus_path, terms_used
            )

            # === DIAGNOSTIC: Word2Vec training debug ===
            print("=== Word2Vec TRAINING DEBUG ===")
            if w2v_cbow is not None:
                print(f"CBOW vocabulary size: {len(w2v_cbow.wv)}")
                print(f"CBOW vector size: {w2v_cbow.vector_size}")

                # Check if target words are in vocabulary
                missing_in_cbow = [w for w in terms_used if w not in w2v_cbow.wv]
                missing_in_sg = [w for w in terms_used if w not in w2v_sg.wv] if w2v_sg is not None else []

                print(f"Missing in CBOW: {missing_in_cbow}")
                print(f"Missing in Skip-gram: {missing_in_sg}")

                print(f"CBOW window: {w2v_cbow.window}")
                print(f"CBOW min_count: {w2v_cbow.min_count}")
                print(f"CBOW sample: {w2v_cbow.sample}")
            # === END DIAGNOSTIC ===

            plot_cumulative_loss(cbow_losses, sg_losses, fig_path=fig_path, mode=mode)
        else:
            # Cache-only attempt
            if FORCE_CLEAN_RUN or FORCE_CLEAN_W2V:
                if RUN_EVAL_ONLY:
                    raise ValueError(f"[{mode}] EVAL-ONLY: cannot force clean W2V while evaluation-only is set.")
                print(f"[CLEAN] W2V training disabled but clean flags set; retraining from scratch …")
                try:
                    nuke_mode_artifacts(mode)
                except (OSError, PermissionError) as e:
                    print(f"[CLEAN] Warning: could not remove caches for mode {mode}: {e}")
                w2v_cbow, w2v_sg, cbow_losses, sg_losses = maybe_train_or_load_w2v_pair(
                    tokenised, cfg["w2v_cbow_final"], cfg["w2v_sg_final"], mode, w2v_corpus_path, terms_used
                )
                plot_cumulative_loss(cbow_losses, sg_losses, fig_path=fig_path, mode=mode)
            else:
                w2v_cbow, w2v_sg, _, _ = maybe_train_or_load_w2v_pair(
                    [], {}, {}, mode, w2v_corpus_path, []
                )
                if RUN_EVAL_ONLY and (w2v_cbow is None or w2v_sg is None):
                    raise FileNotFoundError(f"[{mode}] EVAL-ONLY: valid Word2Vec cache not found.")

        # --- 4) BERT Fine-Tuning / Loading ---
        if RUN_BERT_FINETUNING and bert_raw:
            fresh_tok = BertTokenizer.from_pretrained("bert-base-multilingual-cased", revision=hf_revision)
            fresh_bert = BertForMaskedLM.from_pretrained("bert-base-multilingual-cased", revision=hf_revision,
                                                         attn_implementation="eager")
            tokenizer, bert_ft, _train_losses, _val_losses = maybe_finetune_or_load_bert(
                bert_raw, cfg, mode, bert_corpus_path, fresh_bert, fresh_tok, fig_path=fig_path
            )
        else:
            # load-only path
            tokenizer, bert_ft, _train_losses, _val_losses = maybe_finetune_or_load_bert(
                [], cfg, mode, bert_corpus_path, None, None, fig_path=fig_path
            )
        if mode.upper() == "FULL":
            verify_cpu_inference_fixture(bert_ft, tokenizer, out_dir=Path("."))

        # --- 5) Definitional Audit (now that BERT is available for similarity if needed) ---
        if RUN_DEFINITIONAL_AUDIT and bert_raw:
            run_definitional_audit(
                bert_raw,
                targets=terms_used,
                figures_dir=fig_path,
                mode_tag=mode,
                bert_model=bert_ft,
                bert_tokenizer=tokenizer,
                corpus_path_for_context=bert_corpus_path,
            )

        # --- 6) Model chosen for evaluation (use final fine-tuned model) ---
        best_epoch_num = (int(np.argmin(_val_losses)) + 1) if _val_losses else "N/A"
        logging.info(f"Evaluating fine-tuned BERT (best epoch = {best_epoch_num}).")

        # --- 7) Visual Diagnostics ---
        if RUN_VISUAL_DIAGNOSTICS:
            # === DIAGNOSTIC: Vector quality analysis ===
            print("=== VECTOR QUALITY ANALYSIS ===")
            if w2v_cbow is not None and bert_ft is not None:
                # Check vector norms and similarities for key words
                test_words = ["ma", "engawa", "oku", "yohaku"]
                for word in test_words:
                    if word in w2v_cbow.wv:
                        cbow_vec = w2v_cbow.wv[word]
                        cbow_norm = np.linalg.norm(cbow_vec)

                        bert_vec_result = get_contextual_bert_vectors(word, bert_ft, tokenizer, bert_corpus_path)
                        if bert_vec_result is not None:
                            bert_norm = np.linalg.norm(bert_vec_result)

                            physical_anchor = "engawa"
                            conceptual_anchor = "ma"

                            if physical_anchor in w2v_cbow.wv and word != physical_anchor:
                                sim_physical = cosine(cbow_vec, w2v_cbow.wv[physical_anchor])
                                print(f"CBOW {word} -> {physical_anchor}: {sim_physical:.3f}")

                            if conceptual_anchor in w2v_cbow.wv and word != conceptual_anchor:
                                sim_conceptual = cosine(cbow_vec, w2v_cbow.wv[conceptual_anchor])
                                print(f"CBOW {word} -> {conceptual_anchor}: {sim_conceptual:.3f}")

                            print(f"CBOW {word} norm: {cbow_norm:.3f}, BERT norm: {bert_norm:.3f}")
                            print("---")
            # === END DIAGNOSTIC ===

            banner("VISUAL DIAGNOSTICS")

            if flat_tokens:
                plot_zipf(flat_tokens, fig_path=fig_path, mode=mode)
                plot_top_cooccurrences(tokenised, terms_used, fig_path=fig_path, mode=mode)
            if w2v_cbow:
                plot_top_n_similarities_word2vec(w2v_cbow, "Word2Vec_CBOW", terms_used, fig_path=fig_path, mode=mode)
            if w2v_sg:
                plot_top_n_similarities_word2vec(w2v_sg, "Word2Vec_Skip-gram", terms_used, fig_path=fig_path, mode=mode)
            if bert_ft and flat_tokens:
                plot_top_n_similarities_bert(
                    bert_ft, tokenizer,
                    "Fine-Tuned_BERT",
                    terms_used, flat_tokens,
                    fig_path=fig_path,
                    corpus_path_for_context=bert_corpus_path,
                    mode=mode
                )

            if RUN_TSNE:
                if w2v_cbow:
                    w2v_cbow_clusters = kmeans_on_w2v_embeddings(
                        terms_used, w2v_cbow, use_agglomerative=True, l2_normalize=True
                    )
                    anchor = choose_anchor(w2v_cbow_clusters, PHYSICAL_ANCHORS)
                    if anchor:
                        physical_id = w2v_cbow_clusters[anchor]
                        label_map = {physical_id: "Physical", 1 - physical_id: "Conceptual"}
                        target_vecs_w2v_cbow = {t: w2v_cbow.wv[t] for t in terms_used if t in w2v_cbow.wv}
                        plot_tsne(
                            target_vecs_w2v_cbow, w2v_cbow_clusters, label_map,
                            f"t-SNE of Target Words (Word2Vec CBOW, {mode})", fig_path,
                            f"tsne_w2v_cbow_{mode.lower()}.png", config=cfg
                        )
                    else:
                        print("Skipping W2V CBOW t-SNE: no physical anchor found.")

                if w2v_sg:
                    w2v_sg_clusters = kmeans_on_w2v_embeddings(
                        terms_used, w2v_sg, use_agglomerative=True, l2_normalize=True
                    )
                    anchor_sg = choose_anchor(w2v_sg_clusters, PHYSICAL_ANCHORS)
                    if anchor_sg:
                        physical_id_sg = w2v_sg_clusters[anchor_sg]
                        label_map_sg = {physical_id_sg: "Physical", 1 - physical_id_sg: "Conceptual"}
                        target_vecs_w2v_sg = {t: w2v_sg.wv[t] for t in terms_used if t in w2v_sg.wv}
                        plot_tsne(
                            target_vecs_w2v_sg, w2v_sg_clusters, label_map_sg,
                            f"t-SNE of Target Words (Word2Vec Skip-gram, {mode})", fig_path,
                            f"tsne_w2v_sg_{mode.lower()}.png", config=cfg
                        )
                    else:
                        print("Skipping W2V Skip-gram t-SNE: no physical anchor found.")

                if bert_ft:
                    bert_clusters = kmeans_on_term_embeddings(
                        terms_used, bert_ft, tokenizer, bert_corpus_path,
                        use_agglomerative=True, l2_normalize=True, config=cfg
                    )

                    anchor = choose_anchor(bert_clusters, PHYSICAL_ANCHORS)
                    if anchor:
                        physical_id = bert_clusters[anchor]
                        label_map = {physical_id: "Physical", 1 - physical_id: "Conceptual"}

                        _bert_vec_cache: Dict[str, np.ndarray] = {}

                        def _ctx_vec(term: str) -> Optional[np.ndarray]:
                            """Return cached contextual BERT vector for `term`, computing once if missing."""
                            if term in _bert_vec_cache:
                                return _bert_vec_cache[term]
                            v = get_contextual_bert_vectors(term, bert_ft, tokenizer, bert_corpus_path)
                            if v is not None:
                                _bert_vec_cache[term] = v
                            return v

                        target_vecs_bert = {t: _ctx_vec(t) for t in terms_used}
                        target_vecs_bert = {k: v for k, v in target_vecs_bert.items() if v is not None}

                        plot_tsne(
                            target_vecs_bert, bert_clusters, label_map,
                            f"t-SNE of Target Words (Fine-Tuned BERT, {mode})", fig_path,
                            f"tsne_bert_{mode.lower()}.png", config=cfg
                        )
                    else:
                        print("Skipping BERT t-SNE: no physical anchor found.")

        # --- 8) K-Means, ARI, and Plotting Loop ---
        all_ari_scores_for_mode = defaultdict(dict)

        # === DIAGNOSTIC: Check cluster quality ===
        if RUN_ARI_VALIDATION and ari_hypotheses and w2v_cbow is not None and bert_ft is not None:

            print("=== CLUSTER QUALITY DIAGNOSTIC ===")
            cbow_clusters_diag = kmeans_on_w2v_embeddings(terms_used, w2v_cbow, use_agglomerative=True,
                                                          l2_normalize=True, config=cfg)
            bert_clusters_diag = kmeans_on_term_embeddings(terms_used, bert_ft, tokenizer, bert_corpus_path,
                                                           use_agglomerative=True, l2_normalize=True, config=cfg)

            print(f"CBOW cluster distribution: {Counter(cbow_clusters_diag.values())}")
            print(f"BERT cluster distribution: {Counter(bert_clusters_diag.values())}")

            # Check if trivial
            cbow_counts = Counter(cbow_clusters_diag.values())
            bert_counts = Counter(bert_clusters_diag.values())

            if len(cbow_counts) == 1:
                print("🚨 CBOW: ALL WORDS IN SAME CLUSTER!")
            if len(bert_counts) == 1:
                print("🚨 BERT: ALL WORDS IN SAME CLUSTER!")

            # Check if identical
            if cbow_clusters_diag == bert_clusters_diag:
                print("🚨 CBOW AND BERT CLUSTERS ARE IDENTICAL!")
        # === END DIAGNOSTIC ===

        if RUN_ARI_VALIDATION and ari_hypotheses:
            banner("ARI VALIDATION")
            clustering_methods = [("KMeans", False, False), ("KMeans+L2", False, True)]

            if cfg.get("ari_settings", {}).get("paper1_confirmatory", {}).get("sensitivity_analysis", {}).get(
                    "enable_agglomerative", True):
                l2_norm = cfg.get("ari_sensitivity", {}).get("l2_normalize", True)
                name = "Agglo +L2" if l2_norm else "Agglo"
                clustering_methods.append((f"[{name}]", True, l2_norm))

            k = int(cfg.get("analysis_params", {}).get("model_clusters", 2))

            # === DIAGNOSTIC: Clustering sensitivity test ===
            print("=== CLUSTERING SENSITIVITY TEST ===")
            if w2v_cbow is not None and w2v_sg is not None:
                terms_without_problem = [w for w in terms_used if w not in ["oku", "yohaku"]]

                cbow_all = kmeans_on_w2v_embeddings(terms_used, w2v_cbow, use_agglomerative=True, l2_normalize=True,
                                                    config=cfg)
                cbow_without = kmeans_on_w2v_embeddings(terms_without_problem, w2v_cbow, use_agglomerative=True,
                                                        l2_normalize=True, config=cfg)

                sg_all = kmeans_on_w2v_embeddings(terms_used, w2v_sg, use_agglomerative=True, l2_normalize=True,
                                                  config=cfg)
                sg_without = kmeans_on_w2v_embeddings(terms_without_problem, w2v_sg, use_agglomerative=True,
                                                      l2_normalize=True, config=cfg)

                print("CBOW cluster distribution (all words):", Counter(cbow_all.values()))
                print("CBOW cluster distribution (without oku/yohaku):", Counter(cbow_without.values()))
                print("Skip-gram cluster distribution (all words):", Counter(sg_all.values()))
                print("Skip-gram cluster distribution (without oku/yohaku):", Counter(sg_without.values()))
            # === END DIAGNOSTIC ===

            # Word2Vec models
            for clustering_tag, use_agglo, use_l2 in clustering_methods:
                w2v_models = {"Word2Vec CBOW": w2v_cbow, "Word2Vec Skip-gram": w2v_sg}

                for model_name, model_obj in w2v_models.items():
                    if not model_obj:
                        continue
                    clusters = kmeans_on_w2v_embeddings(terms_used, model_obj, k, use_agglo, use_l2, config=cfg)
                    model_name_tagged = model_name if clustering_tag == "KMeans" else f"{model_name}{clustering_tag}"
                    for hypo_name, hypo_data in ari_hypotheses.items():
                        ari_score = run_ari_cluster_validation(
                            clusters, hypo_data, model_name_tagged, hypo_name,
                            res_path=res_path, fig_path=fig_path, mode_tag=mode
                        )
                        if ari_score is not None:
                            all_ari_scores_for_mode[hypo_name][model_name_tagged] = ari_score

                    # quick cluster plot label mapping
                    words = [w for w in clusters if w in next(iter(ari_hypotheses.values()), {})]
                    if words:
                        cluster_ids = sorted(set(clusters[w] for w in words))
                        cid_to_col = {cid: i for i, cid in enumerate(cluster_ids)}
                        conf = np.zeros((2, len(cluster_ids)), dtype=int)
                        # pick any hypothesis for mapping (just for plots)
                        any_hypo = next(iter(ari_hypotheses.values()))
                        for w in words:
                            conf[int(any_hypo[w]), cid_to_col[clusters[w]]] += 1
                        row_ind, col_ind = linear_sum_assignment(conf.max() - conf)
                        label_map = {}
                        for true_row, col in zip(row_ind, col_ind):
                            cid = cluster_ids[col]
                            label_map[cid] = "Conceptual" if int(true_row) == 0 else "Physical"
                    else:
                        label_map = {0: "Cluster 0", 1: "Cluster 1"}

                    plot_word_clusters(
                        clusters,
                        f"Word Clusters for {model_name_tagged} ({mode.title()})",
                        fig_path / f"clusters_{model_name_tagged.lower().replace(' ', '_').replace('[', '').replace(']', '').replace('+', '')}_{mode.lower()}.png",
                        label_map,
                    )

                # BERT model
                if bert_ft:
                    model_name_tagged = "BERT Cased (Final)" if clustering_tag == "KMeans" else f"BERT Cased (Final){clustering_tag}"
                    clusters = kmeans_on_term_embeddings(terms_used, bert_ft, tokenizer, bert_corpus_path, k, use_agglo,
                                                         use_l2, config=cfg)
                    for hypo_name, hypo_data in ari_hypotheses.items():
                        ari_score = run_ari_cluster_validation(
                            clusters, hypo_data, model_name_tagged, hypo_name,
                            res_path=res_path, fig_path=fig_path, mode_tag=mode
                        )
                        if ari_score is not None:
                            all_ari_scores_for_mode[hypo_name][model_name_tagged] = ari_score

                    words = [w for w in clusters if w in next(iter(ari_hypotheses.values()), {})]
                    if words:
                        cluster_ids = sorted(set(clusters[w] for w in words))
                        cid_to_col = {cid: i for i, cid in enumerate(cluster_ids)}
                        conf = np.zeros((2, len(cluster_ids)), dtype=int)
                        any_hypo = next(iter(ari_hypotheses.values()))
                        for w in words:
                            conf[int(any_hypo[w]), cid_to_col[clusters[w]]] += 1
                        row_ind, col_ind = linear_sum_assignment(conf.max() - conf)
                        label_map = {}
                        for true_row, col in zip(row_ind, col_ind):
                            cid = cluster_ids[col]
                            label_map[cid] = "Conceptual" if int(true_row) == 0 else "Physical"
                    else:
                        label_map = {0: "Cluster 0", 1: "Cluster 1"}

                    plot_word_clusters(
                        clusters,
                        f"Word Clusters for {model_name_tagged} ({mode.title()})",
                        fig_path / f"clusters_{model_name_tagged.lower().replace(' ', '_').replace('[', '').replace(']', '').replace('+', '')}_{mode.lower()}.png",
                        label_map,
                    )

            # summary barcharts
            for hypo_name, scores in all_ari_scores_for_mode.items():
                plot_ari_barchart(scores, hypo_name, tag=mode, fig_path=fig_path)

        # --- 9) Benchmarks ---
        wordsim_results = []
        if RUN_BENCHMARKS:
            wordsim_results = evaluate_wordsim(
                w2v_cbow=w2v_cbow, w2v_sg=w2v_sg,
                bert_finetuned=bert_ft, tokenizer=tokenizer,
                tag=mode.lower(), fig_path=fig_path, res_path=res_path
            )

        # --- FINAL REPORTING AND BIAS CHECK ---
        final_scores = {"ari": all_ari_scores_for_mode, "wordsim": wordsim_results}

        # --- persist ARI table expected by seed aggregator ---
        ari_rows = []
        for _hypo, _map in all_ari_scores_for_mode.items():
            for _model, _score in _map.items():
                ari_rows.append({"Hypothesis": _hypo, "Model": _model, "ARI_Score": float(_score)})
        if ari_rows:
            _ari_out = res_path / f"ari_scores_{mode.lower()}.csv"

        model_paths = {
            "Word2Vec CBOW": cache_root_for(mode) / "w2v_cbow.model",
            "Word2Vec Skip-gram": cache_root_for(mode) / "w2v_sg.model",
            "Fine-Tuned BERT": cache_root_for(mode) / "bert_cased"
        }
        generate_summary_report(res_path, mode, seed_for_run=seed_for_run,
                                scores=final_scores, corpus_path=w2v_corpus_path, model_paths=model_paths)

        write_machine_snapshot(
            res_path=res_path,
            mode=mode,
            w2v_corpus_path=w2v_corpus_path,
            bert_corpus_path=bert_corpus_path,
            model_paths=model_paths
        )
        promote_canonical(fig_path, res_path, mode)

        primary_hypo = cfg.get("primary_ari_hypothesis", "physical_vs_conceptual")
        if mode == "FULL" and primary_hypo in all_ari_scores_for_mode:
            best_bert_model_name = "BERT Cased (Final)[Agglo +L2]"
            if best_bert_model_name in all_ari_scores_for_mode[primary_hypo]:
                ari_full = all_ari_scores_for_mode[primary_hypo][best_bert_model_name]
                save_json(res_path.parent / "temp_ari_full.json", {"ari_full": ari_full})

        if mode == "CLEAN":
            temp_file = res_path.parent / "temp_ari_full.json"
            if temp_file.exists() and primary_hypo in all_ari_scores_for_mode:
                data = load_json(temp_file) or {}
                ari_full = data.get("ari_full")
                best_bert_model_name = "BERT Cased (Final)[Agglo +L2]"
                if ari_full is not None and best_bert_model_name in all_ari_scores_for_mode[primary_hypo]:
                    ari_clean = all_ari_scores_for_mode[primary_hypo][best_bert_model_name]
                    if ari_full != 0:
                        pct = ((ari_clean - ari_full) / abs(ari_full)) * 100.0
                        direction = "Increase" if pct >= 0 else "Decrease"
                        print("\n" + "=" * 72)
                        print(f"DEFINITIONAL BIAS CHECK ({primary_hypo})".center(72))
                        print(f"  - FULL Corpus ARI:  {ari_full:.3f}".center(72))
                        print(f"  - CLEAN Corpus ARI: {ari_clean:.3f}".center(72))
                        print(f"  - Percentage Change: {pct:.2f}% ({direction})".center(72))
                        print("=" * 72)
            if temp_file.exists():
                temp_file.unlink()

        banner(f"PAPER 1 PIPELINE – {mode.upper()} COMPLETE")


# =========================
# SECTION 15: SCRIPT ENTRY POINT
# =========================
# definition
def promote_canonical(fig_path: Path, res_path: Path, mode: str) -> None:
    """Copy per-run artefacts to mode-specific subdirectories, e.g., figures/FULL/."""
    # Define the mode-specific destination directories
    dst_fig = FIG / mode.upper()
    dst_res = RES / mode.upper()

    # Ensure these destination directories exist
    dst_fig.mkdir(parents=True, exist_ok=True)
    dst_res.mkdir(parents=True, exist_ok=True)

    # Copy figure files to the correct destination
    for src in fig_path.glob("*"):
        if src.is_file():
            shutil.copy2(src, dst_fig / src.name)

    # Copy result files to the correct destination
    for src in res_path.glob("*"):
        if src.is_file():
            shutil.copy2(src, dst_res / src.name)


def run_all(seeds):
    """
    Runs the full pipeline once per seed, reseeding *all* RNGs each time
    and writing outputs into per-run subfolders.
    """
    results = []
    for i, run_seed in enumerate(seeds, start=1):
        banner(f"STARTING NLP PIPELINE (RUN {i}/{len(seeds)}, SEED={run_seed})")

        # 1) publish the seed to any code that reads a global
        globals()['SEED'] = int(run_seed)

        # 2) reseed every RNG for this iteration (CPU and CUDA)
        random.seed(SEED)
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(SEED)

        print("[seed check]", SEED,
              "rand:", random.randint(0, 10),
              "np:", np.random.randint(0, 10))

        # 3) per-run output folders
        run_fig_path = Path(f"figures/run_{i}_{run_seed}")
        run_res_path = Path(f"results/run_{i}_{run_seed}")
        run_fig_path.mkdir(parents=True, exist_ok=True)
        run_res_path.mkdir(parents=True, exist_ok=True)

        # fast-skip: if key outputs exist, assume this seed already completed
        completion_markers = [
            run_res_path / "summary_clean.md",
            run_res_path / "wordsim_scores_paper1.csv",
        ]
        if all(p.exists() for p in completion_markers):
            print(f"[skip] run {i} seed={run_seed} already completed, skipping.")
            continue

        # 4) run your main pipeline for this seed
        main(fig_path=run_fig_path, res_path=run_res_path, seed_for_run=run_seed)

    # Post-run aggregation
    compute_seed_aggregates(base_results_dir=Path("results"))
    compute_baseline_reports(base_results_dir=Path("results"), out_dir=Path("results/_aggregates"))

    banner("ALL PIPELINE RUNS COMPLETED.")
    return results


def verify_run() -> int:
    """
    Minimal verification: ensure cached model metas share the current CODE_HASH.
    """
    ok_models = True
    for mode in ("FULL", "CLEAN"):
        root = cache_root_for(mode)
        w2v_meta = load_json(root / "w2v.meta.json")
        bert_meta = load_json(root / "bert_cased" / "meta.json")
        for m in (w2v_meta, bert_meta):
            if m is not None:
                ok_models &= (m.get("code_hash") == CODE_HASH)
    print(f"[verify] models_code_hash_ok={ok_models}")
    return 0 if ok_models else 1

def compute_final_deltas(base_figures_dir: Path):
    """
    Loads FULL and CLEAN similarity results from all figure run folders,
    merges them, computes the final delta, and saves a summary.
    """
    banner("COMPUTING FINAL DEFINITIONAL SIMILARITY DELTAS")
    all_deltas = []
    run_dirs = [d for d in base_figures_dir.iterdir() if d.is_dir() and d.name.startswith("run_")]

    for run_dir in run_dirs:
        try:
            full_csv = run_dir / "definitional_similarity_FULL.csv"
            clean_csv = run_dir / "definitional_similarity_CLEAN.csv"

            if not full_csv.is_file() or not clean_csv.is_file():
                continue

            full_df = pd.read_csv(full_csv).rename(columns={"sim_full": "sim_full_corpus"})
            clean_df = pd.read_csv(clean_csv).rename(columns={"sim_full": "sim_clean_corpus"})

            merged_df = pd.merge(full_df, clean_df, on=["a", "b"])
            merged_df["delta"] = merged_df["sim_clean_corpus"] - merged_df["sim_full_corpus"]
            merged_df["run"] = run_dir.name
            all_deltas.append(merged_df)

        except Exception as e:
            print(f"Could not process deltas for {run_dir.name}: {e}")

    if all_deltas:
        final_df = pd.concat(all_deltas, ignore_index=True)
        final_path = base_figures_dir / "definitional_similarity_deltas.csv"
        final_df.to_csv(final_path, index=False, encoding="utf-8", lineterminator="\n", float_format="%.10g")
        print(f"Saved final combined deltas to {final_path}")
    else:
        print("No delta files found to process.")

def compute_seed_aggregates(base_results_dir: Path) -> None:
    """
    Aggregate ARI and WordSim across all per-seed runs into mean±sd tables.
    Writes {base_results_dir}/summary_seed_stats_FULL.csv and ..._CLEAN.csv.
    """
    def _collect(pattern: str) -> pd.DataFrame:
        rows = []
        for p in base_results_dir.glob(pattern):
            try:
                df = pd.read_csv(p)
                df["run"] = p.parent.name  # e.g., run_1_42
                rows.append(df)
            except (
                FileNotFoundError,
                PermissionError,
                IsADirectoryError,
                pd.errors.EmptyDataError,
                UnicodeDecodeError,
                ValueError,
            ):
                continue
        return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

    for mode in ("FULL", "CLEAN"):
        ari_df = _collect(f"run_*_*/ari_scores_{mode.lower()}.csv")
        ws_df = _collect(f"run_*_*/wordsim_scores_paper1.csv")

        out_rows = []

        if not ari_df.empty:
            for hypo in sorted(ari_df["Hypothesis"].unique()):
                sub = ari_df[ari_df["Hypothesis"] == hypo]
                for model in sorted(sub["Model"].unique()):
                    vals = sub[sub["Model"] == model]["ARI_Score"].astype(float).to_numpy()
                    if vals.size:
                        out_rows.append({
                            "Metric": "ARI",
                            "Mode": mode,
                            "Task": hypo,
                            "Model": model,
                            "Mean": float(np.mean(vals)),
                            "Std": float(np.std(vals, ddof=1)) if vals.size > 1 else 0.0,
                            "N": int(vals.size),
                        })

        if not ws_df.empty:
            for dataset in sorted(ws_df["dataset"].unique()):
                sub = ws_df[ws_df["dataset"] == dataset]
                for model in sorted(sub["model"].unique()):
                    vals = sub[sub["model"] == model]["spearman_rho"].astype(float).to_numpy()
                    if vals.size:
                        out_rows.append({
                            "Metric": "WordSim",
                            "Mode": mode,
                            "Task": dataset,
                            "Model": model,
                            "Mean": float(np.mean(vals)),
                            "Std": float(np.std(vals, ddof=1)) if vals.size > 1 else 0.0,
                            "N": int(vals.size),
                        })

        out_df = pd.DataFrame(out_rows)
        out_path = base_results_dir / f"summary_seed_stats_{mode}.csv"
        if not out_df.empty:
            out_df.to_csv(out_path, index=False)
            print(f"[seed-agg] wrote {out_path}")
        else:
            print(f"[seed-agg] no data for mode={mode} under {base_results_dir}")

def compute_baseline_reports(base_results_dir: Path, out_dir: Path) -> None:
    """
    Build two summary tables from per-run CSVs:
      1) seed_ranks_and_aggregates.csv  (per-seed ranking)
      2) baseline_plain_aggregates.csv  (per-model aggregates)
    Reads: ari_scores_full.csv, ari_scores_clean.csv, ari_bootstrap_ci_full.csv, ari_bootstrap_ci_clean.csv
    Writes: out_dir/*.csv
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # Collect per-run ARIs and CIs
    runs = sorted([p for p in base_results_dir.iterdir() if p.is_dir() and p.name.startswith("run_")])
    if not runs:
        print("[baseline] no run_* folders under", base_results_dir)
        return

    # Helper: normal-approx p from 95% CI and point estimate
    def p_from_ci(point: float, lo_ci: float, hi_ci: float) -> float:
        """Two-sided normal-approx p-value implied by a 95% CI around a point estimate."""
        if any(map(lambda v: v is None or np.isnan(v), (point, lo_ci, hi_ci))) or hi_ci <= lo_ci:
            return float("nan")
        half = (hi_ci - lo_ci) / 2.0
        if half <= 0:
            return float("nan")
        se = half / 1.96
        if se == 0:
            return 0.0 if point != 0 else 1.0
        z = abs(point / se)
        return float(2.0 * (1.0 - 0.5 * (1 + math.erf(z / np.sqrt(2)))))

    primary_hypo = cfg.get("primary_ari_hypothesis", "physical_vs_conceptual")
    # Model label normalisers (tolerant to slight name drift)
    def is_cbow(name: str) -> bool:
        """Return True if a model label corresponds to CBOW."""
        return "CBOW" in name.upper()

    def is_sg(name: str) -> bool:
        """Return True if a model label corresponds to Skip-gram (SG)."""
        return ("SKIP" in name.upper()) or ("SG" in name.upper())

    def is_bert(name: str) -> bool:
        """Return True if a model label corresponds to BERT."""
        return "BERT" in name.upper()

    # Load per-run metrics
    per_seed_rows = []   # for Seed Ranks table

    # Storage to compute model aggregates across seeds
    collect = defaultdict(lambda: defaultdict(list))         # collect[mode][model] -> list of ARIs
    collect_ci = defaultdict(lambda: defaultdict(list))      # same, but CI tuples (lo, hi)
    seed_ids = []

    for run_dir in runs:
        try:
            # seed id from folder name
            parts = run_dir.name.split("_")
            seed = parts[-1]
            seed_ids.append(seed)

            # Expected files inside results/run_*_SEED/
            res_dir = base_results_dir / run_dir.name
            f_full = res_dir / "ari_scores_full.csv"
            f_clean = res_dir / "ari_scores_clean.csv"
            c_full = res_dir / "ari_bootstrap_ci_full.csv"
            c_clean = res_dir / "ari_bootstrap_ci_clean.csv"

            if not (f_full.exists() and f_clean.exists()):
                # skip incomplete runs
                continue

            df_full = pd.read_csv(f_full)
            df_clean = pd.read_csv(f_clean)
            ci_full = pd.read_csv(c_full) if c_full.exists() else pd.DataFrame()
            ci_clean = pd.read_csv(c_clean) if c_clean.exists() else pd.DataFrame()

            # Filter to primary hypothesis only
            if "Hypothesis" in df_full.columns:
                df_full = df_full[df_full["Hypothesis"] == primary_hypo]
            if "Hypothesis" in df_clean.columns:
                df_clean = df_clean[df_clean["Hypothesis"] == primary_hypo]

            # Build a per-seed summary across baseline plain models in BOTH corpora
            items = []
            for mode, df, ci in (("Full", df_full, ci_full), ("Clean", df_clean, ci_clean)):
                if df.empty:
                    continue
                for _, row in df.iterrows():
                    model = str(row.get("Model", ""))
                    ari = float(row.get("ARI_Score", "nan"))
                    # push into aggregates
                    collect[mode][model].append(ari)
                    # find CI for this model+hypothesis if available
                    if not ci.empty and {"Model", "Hypothesis", "CI_Lower", "CI_Upper"}.issubset(ci.columns):
                        hit = ci[(ci["Model"] == model) & (ci["Hypothesis"] == primary_hypo)]
                        if not hit.empty:
                            lo = float(hit.iloc[0]["CI_Lower"])
                            hi = float(hit.iloc[0]["CI_Upper"])
                            collect_ci[mode][model].append((lo, hi))
                            pval = p_from_ci(ari, lo, hi)
                            items.append({"mode": mode, "model": model, "ari": ari, "lo": lo, "hi": hi, "p": pval})
                        else:
                            items.append({"mode": mode, "model": model, "ari": ari, "lo": float("nan"), "hi": float("nan"), "p": float("nan")})
                    else:
                        items.append({"mode": mode, "model": model, "ari": ari, "lo": float("nan"), "hi": float("nan"), "p": float("nan")})

            if items:
                # restrict to baseline plain models
                base_items = [it for it in items if (is_bert(it["model"]) or is_sg(it["model"]) or is_cbow(it["model"]))]
                if base_items:
                    # Mean ARI across selected items
                    mean_ari = float(np.nanmean([it["ari"] for it in base_items]))
                    # Mean p across items where CI present (normal approx). If none, NaN.
                    ps = [it["p"] for it in base_items if not np.isnan(it["p"])]
                    mean_p = float(np.mean(ps)) if ps else float("nan")
                    # % of items whose CI excludes 0 (when CI present)
                    excl = [int((not np.isnan(it["lo"])) and (it["lo"] > 0 or it["hi"] < 0)) for it in base_items]
                    pct_excl = 100.0 * (sum(excl) / len(excl)) if excl else 0.0
                    # Consistency: fraction agreeing on the majority sign of ARI
                    signs = [np.sign(it["ari"]) for it in base_items if not np.isnan(it["ari"])]
                    if signs:
                        maj = 1 if signs.count(1) >= signs.count(-1) else -1
                        consist = 100.0 * (sum(int(s == maj) for s in signs) / len(signs))
                    else:
                        consist = 0.0
                    # Composite: weighted mean of z-scored mean_ari and pct_excl (simple, deterministic)
                    comp = float((np.nan_to_num(mean_ari) / (abs(np.nan_to_num(mean_ari)) + 1e-8)) * (pct_excl / 100.0))

                    per_seed_rows.append({
                        "Seed": seed,
                        "Mean ARI": mean_ari,
                        "Mean p": mean_p,
                        "% CI Exclude 0": pct_excl,
                        "Composite": comp,
                        "Consistency %": consist,
                    })

        except Exception as e:
            print(f"[baseline] skip {run_dir.name}: {e}")

    # Rank seeds by Composite, then Mean ARI (desc)
    if per_seed_rows:
        seed_df = pd.DataFrame(per_seed_rows)
        seed_df = seed_df.sort_values(by=["Composite", "Mean ARI"], ascending=[False, False]).reset_index(drop=True)
        seed_df.insert(0, "Rank", np.arange(1, len(seed_df) + 1, dtype=int))
        seed_out = out_dir / "seed_ranks_and_aggregates.csv"
        seed_df.to_csv(seed_out, index=False, encoding="utf-8", lineterminator="\n", float_format="%.10g")
        print(f"[baseline] wrote {seed_out}")

    # Model aggregates across seeds and corpora
    # Build rows for BERT/SG/CBOW, for "Full" and "Clean"
    model_rows = []
    for mode in ("Full", "Clean"):
        for model, vals in sorted(collect[mode].items()):
            if not (is_bert(model) or is_sg(model) or is_cbow(model)):
                continue
            arr = np.asarray(vals, dtype=float)
            mean = float(np.nanmean(arr)) if arr.size else float("nan")
            sd = float(np.nanstd(arr, ddof=1)) if arr.size > 1 else 0.0
            cv = (sd / abs(mean)) if (not np.isnan(mean) and mean != 0.0) else float("nan")
            # % CI exclude 0 and Sig % from CI files, if present
            ci_list = collect_ci[mode].get(model, [])
            if ci_list:
                excl = [int(lo > 0 or hi < 0) for (lo, hi) in ci_list if not (np.isnan(lo) or np.isnan(hi))]
                pct_excl = 100.0 * (sum(excl) / len(ci_list)) if ci_list else 0.0
                sig_pct = pct_excl  # same criterion, explicit duplicate to match your table
            else:
                pct_excl = 0.0
                sig_pct = 0.0

            model_rows.append({
                "Model": "BERT" if is_bert(model) else ("SG" if is_sg(model) else "CBOW"),
                "Corpus": mode,
                "Mean ARI": mean,
                "SD ARI": sd,
                "% CI Exclude 0": pct_excl,
                "Sig %": sig_pct,
                "CV": cv
            })

    # Δ ARI: Clean minus Full by model family, on means
    if model_rows:
        agg_df = pd.DataFrame(model_rows)
        deltas = []
        for family in ["BERT", "SG", "CBOW"]:
            sub = agg_df[agg_df["Model"] == family]
            if not sub.empty and {"Corpus", "Mean ARI"}.issubset(sub.columns):
                mf = sub[sub["Corpus"] == "Full"]["Mean ARI"]
                mc = sub[sub["Corpus"] == "Clean"]["Mean ARI"]
                if not mf.empty and not mc.empty:
                    deltas.append({"Model": family, "Δ ARI": float(mc.values[0] - mf.values[0])})
        delta_df = pd.DataFrame(deltas)

        # merge Δ back
        agg_df = agg_df.merge(delta_df, on="Model", how="left")
        agg_out = out_dir / "baseline_plain_aggregates.csv"
        agg_df.to_csv(agg_out, index=False, encoding="utf-8", lineterminator="\n", float_format="%.10g")
        print(f"[baseline] wrote {agg_out}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run Paper 1 Pipeline")
    parser.add_argument("--quiet", action="store_true", help="Reduce console output")
    args = parser.parse_args()
    logging.basicConfig(level=logging.ERROR if args.quiet else logging.INFO)

    # Light-weight quiet mode: redirect tqdm + some prints
    if args.quiet:
        # TQDM minimal
        tqdm.__init__ = partial(tqdm.__init__, disable=True)

    # --- Read multi-run settings from config ---
    repeats_cfg = cfg.get("repeats", {})
    n_runs = int(repeats_cfg.get("n_runs", 1))
    main_seed = int(cfg.get("seed", 42))

    # --- Determine which seeds to use for the runs ---
    seeds_from_config = cfg.get("repeats", {}).get("seeds", [])
    if seeds_from_config:
        seeds_to_run = [int(s) for s in seeds_from_config]
        print(f"Using {len(seeds_to_run)} seeds directly from config.yaml: {seeds_to_run}")
    elif n_runs == 1:
        seeds_to_run = [main_seed]
    else:
        rng = np.random.default_rng(main_seed)
        seeds_to_run = []
        while len(seeds_to_run) < n_runs:
            new_seed = int(rng.integers(1, 1_000_000))
            if new_seed != main_seed and new_seed not in seeds_to_run:
                seeds_to_run.append(new_seed)
        print(f"Generated {n_runs} deterministic seeds for validation (excluding main seed {main_seed}): {seeds_to_run}")

    # --- Execute the main pipeline for each determined seed ---
    run_all(seeds_to_run)

    # --- Perform post-run analysis ---
    compute_final_deltas(Path("figures"))