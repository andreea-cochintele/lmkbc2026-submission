"""
TEST- SET

Push with:
    kaggle kernels push -p . --accelerator NvidiaTeslaT4

Self-contained since Kaggle doesn't provide the local folder structure -
clones the official dataset repo (data/, prompt_templates/,
abstract_model.py) and runs the same pipeline used locally.

Check status:
    kaggle kernels status username/projectname

Pull results:
    kaggle kernels output username/projectname -p ./kaggle_results --file-pattern "predictions\.jsonl$" -o

Evaluate:
    python ../dataset2026/evaluate.py -g ../dataset2026/data/val.jsonl -p kaggle_results/predictions.jsonl | Tee-Object -FilePath results.txt
"""


import csv
import json
import os
import random
import re
import subprocess
import sys
import time
import unicodedata
from collections import Counter, defaultdict
from typing import Dict, List, Optional

# Runs through a local Ollama server instead of `transformers` when True.
# Ollama re-hosts Gemma/Llama without needing an HF token or license
# click-through, sidestepping the gating entirely. Model tag is set
# further down, in main().
USE_OLLAMA_MODEL = True

# Fixes which few-shot examples random.sample() picks in _build_prompt,
# so identical settings give identical results across runs.
random.seed(42)

# data/, prompt_templates/, and abstract_model.py all come from here;
# everything below assumes the clone already exists.
REPO_DIR = "/kaggle/working/dataset2026"
if not os.path.exists(REPO_DIR):
    subprocess.run(
        ["git", "clone", "https://github.com/lm-kbc/dataset2026.git", REPO_DIR],
        check=True,
    )

sys.path.insert(0, REPO_DIR)
from models.abstract_model import AbstractModel  # noqa: E402

# Kaggle's preinstalled transformers doesn't recognize the "gemma3" arch
# yet, so install the GitHub version before it gets imported anywhere
# below (Python would otherwise keep the old one cached).
subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "-U",
     "git+https://github.com/huggingface/transformers.git"],
    check=True,
)
# Needed for use_quantization=True (4-bit loading); not guaranteed to be
# on the base Kaggle image.
subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "-U", "bitsandbytes"],
    check=True,
)

# Gemma is gated on HF - log in with the HF_TOKEN Kaggle secret (Add-ons
# > Secrets in the notebook editor) to enable the download. Retrying
# because Kaggle's Secrets service has a known intermittent connection
# bug unrelated to whether the secret itself is configured correctly.
hf_login_ok = False
for attempt in range(4):
    try:
        from kaggle_secrets import UserSecretsClient
        from huggingface_hub import login as hf_login
        hf_token = UserSecretsClient().get_secret("HF_TOKEN")
        hf_login(token=hf_token)
        print("Logged into Hugging Face Hub using the HF_TOKEN Kaggle secret.")
        hf_login_ok = True
        break
    except Exception as e:
        print(f"HF login attempt {attempt + 1}/4 failed: {e}")
        time.sleep(15)
if not hf_login_ok:
    print("Giving up on HF login - fine as long as the loaded model isn't gated.")

# Ollama setup. This is what actually allows running Gemma/Llama without
# touching HF's gating - Ollama's own model library re-hosts them directly.
OLLAMA_MODEL_TAG = "gemma3:27b"  # change this to switch models
if USE_OLLAMA_MODEL:
    print("Installing zstd (required by the Ollama install script)...")
    subprocess.run("apt-get update -qq && apt-get install -y -qq zstd", shell=True, check=True)

    print("Installing Ollama...")
    subprocess.run("curl -fsSL https://ollama.com/install.sh | sh", shell=True, check=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "ollama"], check=True)
    import ollama  

    print("Starting the Ollama server in the background...")
    ollama_server_process = subprocess.Popen(["ollama", "serve"])
    time.sleep(10)  # let the server come up before the first request

    print(f"Pulling {OLLAMA_MODEL_TAG} (this downloads the GGUF weights)...")
    # "ollama pull" hits ollama.com's registry over the network; transient
    # failures there (503, connection reset) are common on a large model
    # pull, same class of issue as the earlier HF download stalls. Retry
    # with backoff instead of failing the run over a temporary hiccup.
    pull_ok = False
    for attempt in range(1, 5):
        result = subprocess.run(["ollama", "pull", OLLAMA_MODEL_TAG])
        if result.returncode == 0:
            pull_ok = True
            break
        print(f"'ollama pull {OLLAMA_MODEL_TAG}' failed (attempt {attempt}/4, "
              f"exit code {result.returncode}) -- retrying in 20s...")
        time.sleep(20)
    if not pull_ok:
        raise RuntimeError(
            f"'ollama pull {OLLAMA_MODEL_TAG}' failed 4 times in a row -- "
            "this looks like more than a transient blip. Check "
            "https://status.ollama.com (or similar) before retrying the "
            "whole notebook, and confirm the model tag is spelled correctly."
        )

import torch  
# random.seed(42) above only fixes few-shot example selection. Every
# do_sample=True generation (majority vote, retry-on-empty) draws from
# torch's RNG instead, left unseeded otherwise - hence the run-to-run
# wobble previously seen on awardWonBy/companyTradesAtStockExchange.
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig  

NUMERIC_RELATIONS = {"hasArea", "hasCapacity", "seriesHasNumberOfEpisodes"}

# Only used when USE_ONE_PER_LINE_FORMAT is on. Comma-splitting breaks
# when the model hedges in full sentences ("Well, I'm not sure, but
# possibly Serbia" turns into 3 garbage fragments) - one object per line
# keeps a rambling sentence as ONE line so the cleaning filters below can
# reject it whole instead of chopping it up.
OUTPUT_FORMAT_INSTRUCTIONS = {
    "list": (
        "Answer with ONE object per line. If there are none, answer with "
        "the single word None. Do not add explanations or numbering."
    ),
    "numeric": "Answer with ONLY the number, no units, no extra text.",
}

# Tested globally first (F1 macro 0.415 vs 0.423 with plain comma
# format) - no overall benefit, and it worsened awardWonBy's
# over-generation. Scoped down to just awardWonBy to isolate the effect.
ONE_PER_LINE_RELATIONS = {"awardWonBy"}

# --- Feature flags for incremental testing ---------------------------------
# First 4 are confirmed good (F1 macro 0.423, all on + fixed seed).
# USE_ONE_PER_LINE_FORMAT was the last one tested and didn't help, so
# it's off here to measure the baseline without it.
USE_RELATION_HINTS = True            # appends clarifying text to each question
USE_PER_RELATION_MAX_TOKENS = True   # different max_new_tokens per relation
USE_LIST_CLEANING = True           # filters rambling/meta-comment items out of list answers
USE_SINGLE_ANSWER_CAP = True        # caps personHasCityOfDeath to 1 answer
USE_ONE_PER_LINE_FORMAT = False      # tested (global and awardWonBy-only), no benefit - back to comma-separated

# awardWonBy is by far the worst relation (F1 ~0.06-0.07): the model
# massively over-generates (20-30+ names per award, mostly wrong).
# Generating 3 times (1 greedy + 2 sampled) and keeping only names that
# recur in at least 2 of 3 filters this down, on the assumption that a
# genuinely known name tends to repeat while a hallucinated one doesn't.
USE_MAJORITY_VOTE_AWARDWONBY = True  # confirmed good: F1 0.147 -> 0.161, avg #preds 34.0 -> 20.8
MAJORITY_VOTE_RELATIONS = {"awardWonBy"}
SAMPLES_PER_VOTE = 3  # 5 samples / threshold 3 tested, no clear benefit (F1 0.168 -> 0.158), reverted
MIN_VOTE_COUNT = 2  # a name must appear in at least this many of the SAMPLES_PER_VOTE generations

# Different problem here: personHasCityOfDeath and
# companyTradesAtStockExchange under-generate rather than over-generate -
# an empty answer even when a retry might know something. Retrying only
# when greedy comes back empty, keeping the first non-empty retry, is
# more conservative than the vote above and never touches an
# already-successful answer.
USE_RETRY_ON_EMPTY = True
# personHasCityOfDeath was included here too, but A/B testing showed
# retry-on-empty hurts it (F1 0.440 -> 0.370) while helping
# companyTradesAtStockExchange (F1 0.658 -> 0.676) - likely because many
# empty personHasCityOfDeath answers are CORRECTLY empty (the relation
# allows this for still-living/unknown people), so forcing a retry there
# just replaces a correct empty answer with a wrong guess.
RETRY_ON_EMPTY_RELATIONS = {"companyTradesAtStockExchange"}
MAX_RETRY_ATTEMPTS = 3  # 1 greedy + up to 2 sampled retries

# hasArea, hasCapacity, and personHasCityOfDeath don't use any
# multi-sample aggregation yet, unlike awardWonBy above, where majority
# voting already showed a clear benefit. Extending the same
# self-consistency idea here, with an aggregation rule suited to each
# answer type (median for numeric).
USE_SELF_CONSISTENCY_NUMERIC = True
SELF_CONSISTENCY_NUMERIC_RELATIONS = {"hasArea", "hasCapacity"}
SELF_CONSISTENCY_SAMPLES = 3  # cheap: max_new_tokens=16, 100 subjects each

USE_SELF_CONSISTENCY_SINGLE_ANSWER = True
# personHasCityOfDeath was tested here too, but A/B data showed it hurts
# (F1 0.440 -> 0.360) - the same failure pattern as USE_RETRY_ON_EMPTY on
# this relation. For a precise single-fact answer, greedy decoding is
# usually already the most reliable guess; two extra sampled generations
# can occasionally agree on a WRONG city and outvote a correct greedy
# answer 2-to-1. Left empty on purpose; personHasCityOfDeath goes through
# plain single greedy generation instead.
SELF_CONSISTENCY_SINGLE_ANSWER_RELATIONS = set()
# -----------------------------------------------------------------------------

# A person only has one city of death - if the model hedges with two
# guesses, only the first is kept.
SINGLE_ANSWER_RELATIONS = {"personHasCityOfDeath"}

# A flat max_new_tokens=64 was cutting off awardWonBy mid-list (some
# awards have many recipients) while wasting time on short answers.
# Values below are generous but relation-appropriate.
MAX_NEW_TOKENS_BY_RELATION = {
    "hasArea": 16,
    "hasCapacity": 16,
    "seriesHasNumberOfEpisodes": 16,
    "personHasCityOfDeath": 40,
    "companyTradesAtStockExchange": 60,
    "countryLandBordersCountry": 80,
    "awardWonBy": 300,
}

# Short clarifications appended to each question so the model knows
# exactly what counts as a correct answer. Cuts down a lot of the
# "technically related but wrong" answers, especially for awardWonBy
# (the model likes to name the winning work instead of the person/group
# who actually won, unless told not to).
RELATION_HINTS = {
    "countryLandBordersCountry": (
        "Only current, internationally recognised land borders. "
        "Exclude maritime-only neighbors. If there are none, answer None."
    ),
    "personHasCityOfDeath": (
        "Name only the city (not the country or region). "
        "If still alive or unknown, answer None."
    ),
    "hasCapacity": "Answer with a single integer number of people, no units, no commas.",
    "hasArea": "Answer with a single number in square kilometers, no units, no commas.",
    "awardWonBy": (
        "Name the recipient (person, group, or organization) who WON the "
        "award, not the work/project that was awarded. List as many "
        "confirmed winners as you know."
    ),
    "companyTradesAtStockExchange": "Name the stock exchange(s). If not publicly traded, answer None.",
}


# Main model class, transformers-based - same one used locally.
class HFTransformersBaselineModel(AbstractModel):
    def __init__(
        self,
        llm_path: str,
        prompt_templates_file: str,
        max_new_tokens: int = 64,
        batch_size: int = 2,
        use_quantization: bool = False,
        enable_thinking: bool = False,
        few_shot: int = 5,
        train_data_file: str = None,
        **kwargs,
    ):
        super().__init__()
        self.max_new_tokens = max_new_tokens
        self.batch_size = batch_size
        self.enable_thinking = enable_thinking
        self.few_shot = few_shot

        self.prompt_templates = self._load_prompt_templates(prompt_templates_file)
        self.few_shot_examples = (
            self._load_few_shot_examples(train_data_file) if train_data_file else {}
        )
        # One RNG per relation, seeded from (42 + relation name), so extra
        # sampling for one relation (majority voting on awardWonBy) can
        # never shift which few-shot examples another relation draws.
        self._relation_rngs: Dict[str, random.Random] = {}

        quantization_config = BitsAndBytesConfig(load_in_4bit=True) if use_quantization else None

        self.tokenizer = AutoTokenizer.from_pretrained(llm_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        # Decoder-only models need left-padding for batched generation,
        # otherwise shorter sequences in the batch confuse the model about
        # where the real tokens end.
        self.tokenizer.padding_side = "left"

        self.model = AutoModelForCausalLM.from_pretrained(
            llm_path,
            dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            quantization_config=quantization_config,
        )
        self.model.eval()

    @staticmethod
    def _load_prompt_templates(path: str) -> Dict[str, str]:
        templates = {}
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                templates[row["Relation"]] = row["PromptTemplate"]
        return templates

    @staticmethod
    def _load_few_shot_examples(path: str) -> Dict[str, List[dict]]:
        by_relation = defaultdict(list)
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                by_relation[row["Relation"]].append(row)
        return by_relation

    @staticmethod
    def _stringify_object(obj) -> str:
        """
        ObjectEntities in train.jsonl can hold plain strings or, per
        object, a list of alias strings (e.g. [["Haiti", "Republic of
        Haiti"], ["Cuba"]]). Either way, one readable string per object
        is enough for the few-shot example.
        """
        if isinstance(obj, list):
            return obj[0] if obj else ""
        return str(obj)

    def _get_relation_rng(self, relation: str) -> random.Random:
        """Random instance private to this relation, seeded from (42 +
        relation name) so another relation's extra RNG calls (e.g. 3x for
        majority voting) never shift which few-shot examples this one
        draws."""
        if relation not in self._relation_rngs:
            self._relation_rngs[relation] = random.Random(f"42-{relation}")
        return self._relation_rngs[relation]

    def _build_prompt(self, subject_entity: str, relation: str) -> str:
        template = self.prompt_templates.get(relation, "What is the {relation} of {subject_entity}?")
        question = template.format(subject_entity=subject_entity, relation=relation)
        hint = RELATION_HINTS.get(relation) if USE_RELATION_HINTS else None
        if hint:
            question = f"{question} {hint}"

        use_one_per_line = USE_ONE_PER_LINE_FORMAT and relation in ONE_PER_LINE_RELATIONS
        if use_one_per_line:
            rel_type = "numeric" if relation in NUMERIC_RELATIONS else "list"
            question = f"{question} {OUTPUT_FORMAT_INSTRUCTIONS[rel_type]}"

        examples = self.few_shot_examples.get(relation, [])
        sampled = self._get_relation_rng(relation).sample(examples, k=min(self.few_shot, len(examples)))

        lines = []
        for example in sampled:
            example_question = template.format(subject_entity=example["SubjectEntity"], relation=relation)
            if hint:
                example_question = f"{example_question} {hint}"
            if use_one_per_line:
                rel_type = "numeric" if relation in NUMERIC_RELATIONS else "list"
                example_question = f"{example_question} {OUTPUT_FORMAT_INSTRUCTIONS[rel_type]}"

            objects = [self._stringify_object(o) for o in example.get("ObjectEntities", [])]
            if use_one_per_line:
                example_answer = "\n".join(objects) if objects else "None"
            else:
                example_answer = ", ".join(objects) or "None"

            lines.append(f"Q: {example_question}\nA: {example_answer}")
        lines.append(f"Q: {question}\nA:")
        return "\n\n".join(lines)

    # Phrases from model rambling/hedging, rarely part of an actual
    # entity name. Not exhaustive.
    _META_MARKERS = (
        "note:", "however", "according to", "i'm not sure", "i am not sure",
        "as an ai", "cannot confirm", "let me", "please", "it depends",
        "here's a list", "here is a list", "here are the", "here's the",
        "list of", "recipients include", "winners include",
    )
    # Standalone conversational filler that can leak in as its own "item"
    # (e.g. a lone "Okay" before the actual list starts). Exact match
    # only, so a real name containing one of these as a substring isn't
    # rejected.
    _FILLER_WORDS = {"okay", "ok", "sure", "certainly", "alright", "well"}

    def _clean_list_item(self, raw: str, subject_entity: str) -> Optional[str]:
        # "." stays out of this strip set - the sentence-leak check below
        # needs the trailing period; it's dropped later as a cosmetic step.
        cleaned = raw.strip(" \"'()[]`-\t")
        if not cleaned:
            return None

        # Unclosed trailing parenthetical, e.g. "Kenneth E. Iverson
        # (computer scientist" - a Wikipedia-style disambiguation suffix
        # that never closes. Truncating collapses several of these into
        # the same plain name, which then dedupes correctly.
        if cleaned.count("(") > cleaned.count(")"):
            cleaned = cleaned[:cleaned.rfind("(")].strip(" \"'-\t")

        if not cleaned:
            return None
        lowered = cleaned.lower()
        if lowered in {"none", "n/a", "na", "null", "unknown"}:
            return None
        if lowered in self._FILLER_WORDS:
            return None
        # A real entity name is rarely more than a handful of words;
        # anything longer is almost certainly explanatory text.
        if len(cleaned.split()) > 12:
            return None
        # Entity names don't end with a colon (a label like "Answer:",
        # not a thing being named) or contain a code-comment marker.
        if cleaned.endswith(":") or "//" in cleaned:
            return None
        # A leaked sentence ("Mongolia also shares its northern border
        # with Russia.") reliably ends in a period preceded by a
        # lowercase letter with several words; abbreviation-style names
        # ending in a period (e.g. "Washington, D.C.") end with an
        # UPPERCASE letter before the final period, so those are left alone.
        if (len(cleaned) >= 2 and cleaned.endswith(".") and cleaned[-2].islower()
                and len(cleaned.split()) >= 4):
            return None
        if any(marker in lowered for marker in self._META_MARKERS):
            return None
        # Safe to drop a lone trailing period now - the checks above
        # already used it.
        cleaned = cleaned.rstrip(".").strip() or cleaned
        lowered = cleaned.lower()
        # An entity can't be its own answer (e.g. "Romania" bordering "Romania").
        if lowered == subject_entity.strip().lower():
            return None
        return cleaned

    # Alternate units that can slip into an answer despite the "no units"
    # instruction - matched near the number, with a multiplier to convert
    # into km^2 (the unit hasArea is scored in).
    _AREA_UNIT_TO_KM2 = [
        (re.compile(r"\bhectares?\b|\bha\b", re.IGNORECASE), 0.01),
        (re.compile(r"\bsquare\s*miles?\b|\bsq\.?\s*mi\b|\bmi2\b|mi\u00b2", re.IGNORECASE), 2.58999),
        (re.compile(r"\bsquare\s*meters?\b|\bsq\.?\s*m\b(?!i)|\bm2\b|m\u00b2", re.IGNORECASE), 0.000001),
        (re.compile(r"\bacres?\b", re.IGNORECASE), 0.00404686),
    ]

    def _parse_answer(self, raw_answer: str, relation: str, subject_entity: str = "") -> List[str]:
        text = raw_answer.strip()
        if not text or text.lower().startswith("none"):
            if not text:
                print(f"[debug] EMPTY raw answer for {relation!r} / {subject_entity!r}")
            return []

        if relation in NUMERIC_RELATIONS:
            # NFKC normalizes lookalike/unicode digit variants before the
            # search, so a number isn't missed just from encoding quirks.
            normalized = unicodedata.normalize("NFKC", text)
            # Must start with an actual digit, otherwise a stray comma or
            # period in rambling text ("Well, I think...") gets matched
            # as if it were a number.
            match = re.search(r"-?\d[\d,]*\.?\d*", normalized)
            if not match:
                print(f"[debug] no number found for {relation!r} / {subject_entity!r}: {text!r}")
                return []
            value = match.group(0).replace(",", "")
            if value.startswith("-"):
                return []  # negative area/capacity is never valid, reject it
            if value.endswith(".0"):
                value = value[:-2]
            num = float(value)

            # "15k" shorthand for 15000 - only when the "k" is directly
            # attached to the matched number (no space), to avoid
            # misreading an unrelated nearby word as a multiplier.
            tail = normalized[match.end():match.end() + 1]
            if tail.lower() == "k":
                num *= 1_000

            # hasArea only: convert to km^2 if the answer came back in
            # hectares/sq mi/etc. despite the "no units" instruction,
            # instead of silently scoring the wrong-unit number.
            if relation == "hasArea":
                for pattern, multiplier in self._AREA_UNIT_TO_KM2:
                    if pattern.search(normalized):
                        num *= multiplier
                        break

            value = str(int(num)) if num.is_integer() else str(num)
            return [value]

        if USE_ONE_PER_LINE_FORMAT and relation in ONE_PER_LINE_RELATIONS:
            # One object per line - a rambling sentence stays as ONE line
            # here (instead of getting chopped at every comma), so the
            # cleaning filters below can reject it as a whole.
            raw_items = text.splitlines()
        else:
            raw_items = text.split(",")

        if USE_LIST_CLEANING:
            items = [self._clean_list_item(item, subject_entity) for item in raw_items]
            items = [item for item in items if item]
        else:
            # Simpler fallback: strip whitespace/punctuation only, no
            # meta-comment filtering.
            items = [item.strip(" .\"'") for item in raw_items]
            items = [item for item in items if item]

        if not items:
            print(f"[debug] all items filtered out for {relation!r} / {subject_entity!r}, raw was: {text!r}")

        if USE_SINGLE_ANSWER_CAP and relation in SINGLE_ANSWER_RELATIONS:
            items = items[:1]

        return items

    # Small set of known aliases so voting doesn't split between two
    # spellings of the same real entity. Not exhaustive.
    _ENTITY_ALIASES = {
        "usa": "united states", "us": "united states",
        "united states of america": "united states",
        "uk": "united kingdom", "great britain": "united kingdom",
        "nyse": "new york stock exchange",
        "nasdaq": "nasdaq stock market",
        "nasdaq global select market": "nasdaq stock market",
        "nasdaq global market": "nasdaq stock market",
        "lse": "london stock exchange",
        "tse": "tokyo stock exchange", "tyo": "tokyo stock exchange",
        "nse": "national stock exchange of india",
        "bse": "bombay stock exchange",
    }

    @classmethod
    def _normalize_for_voting(cls, obj: str) -> str:
        """Loose normalization used only to decide whether two generations
        are "the same name" for voting purposes - the original spelling
        is kept separately for the actual output. Splits camelCase/
        PascalCase runs ("NewYorkStockExchange" -> "New York Stock
        Exchange") and underscores into separate words first, so those
        collapse to the same key as a properly-spaced spelling."""
        text = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", obj)
        text = text.replace("_", " ")
        text = text.lower().strip()
        text = re.sub(r"^the\s+", "", text)
        text = re.sub(r"[^\w\s]", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        return cls._ENTITY_ALIASES.get(text, text)

    def _generate_one(self, subject_entity: str, relation: str, do_sample: bool,
                       sample_index: int = 0) -> str:
        """One non-batched generation for a subject/relation - used by the
        majority-vote path below, since that needs several independent
        generations per item instead of one batched pass.

        sample_index is which attempt this is (0 = greedy/first pass, 1,
        2, ... for the sampled passes after it), used to seed torch's RNG
        deterministically per attempt (42 + sample_index). This keeps a
        full run reproducible without collapsing every sampled attempt
        into identical output, which a single fixed seed would do."""
        torch.manual_seed(42 + sample_index)
        prompt = self._build_prompt(subject_entity, relation)
        chat_prompt = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            enable_thinking=self.enable_thinking,
            tokenize=False,
        )
        encoded = self.tokenizer([chat_prompt], return_tensors="pt", padding=True).to(self.model.device)
        max_new_tokens = (
            MAX_NEW_TOKENS_BY_RELATION.get(relation, self.max_new_tokens)
            if USE_PER_RELATION_MAX_TOKENS else self.max_new_tokens
        )
        gen_kwargs = dict(max_new_tokens=max_new_tokens, do_sample=do_sample)
        if do_sample:
            gen_kwargs.update(temperature=0.7, top_p=0.9)

        with torch.no_grad():
            output_ids = self.model.generate(**encoded, **gen_kwargs)

        new_tokens = output_ids[:, encoded["input_ids"].shape[1]:]
        return self.tokenizer.batch_decode(new_tokens, skip_special_tokens=True)[0]

    def _majority_vote_predict(self, subject_entity: str, relation: str) -> List[str]:
        """Generates SAMPLES_PER_VOTE answers (1 greedy + the rest
        sampled) and keeps names that show up in at least MIN_VOTE_COUNT
        of them. A genuinely known name tends to repeat; a hallucinated
        one usually doesn't."""
        vote_lists = []
        for j in range(SAMPLES_PER_VOTE):
            raw_answer = self._generate_one(subject_entity, relation, do_sample=(j > 0), sample_index=j)
            vote_lists.append(self._parse_answer(raw_answer, relation, subject_entity))

        counts = Counter()
        display_forms = {}
        for vote_list in vote_lists:
            seen_this_vote = set()
            for item in vote_list:
                key = self._normalize_for_voting(item)
                if not key or key in seen_this_vote:
                    continue  # a name repeated within one generation shouldn't count twice
                seen_this_vote.add(key)
                counts[key] += 1
                display_forms.setdefault(key, item)

        kept = [display_forms[key] for key, count in counts.items() if count >= MIN_VOTE_COUNT]
        print(f"[debug] majority vote for {relation!r} / {subject_entity!r}: "
              f"{len(kept)} kept out of {len(counts)} distinct candidates across {SAMPLES_PER_VOTE} samples")
        return kept

    def _numeric_self_consistency_predict(self, subject_entity: str, relation: str) -> List[str]:
        """Generates SELF_CONSISTENCY_SAMPLES answers (1 greedy + the rest
        sampled) and returns their median - the numeric equivalent of
        majority voting. An occasional wildly-off sampled guess gets
        outvoted by the middle value instead of being taken at face
        value."""
        values = []
        for j in range(SELF_CONSISTENCY_SAMPLES):
            raw_answer = self._generate_one(subject_entity, relation, do_sample=(j > 0), sample_index=j)
            parsed = self._parse_answer(raw_answer, relation, subject_entity)
            if parsed:
                try:
                    values.append(float(parsed[0]))
                except ValueError:
                    pass
        if not values:
            return []
        values.sort()
        n = len(values)
        median = values[n // 2] if n % 2 == 1 else (values[n // 2 - 1] + values[n // 2]) / 2
        print(f"[debug] numeric self-consistency for {relation!r} / {subject_entity!r}: "
              f"samples={values} -> median={median}")
        return [str(int(median)) if float(median).is_integer() else str(median)]

    def _single_answer_self_consistency_predict(self, subject_entity: str, relation: str) -> List[str]:
        """Generates SELF_CONSISTENCY_SAMPLES answers and keeps whichever
        single answer got the most votes (ties go to whichever came
        first, i.e. the greedy answer wins over a later sampled one).
        For relations with a genuinely singleton gold answer
        (personHasCityOfDeath), picking by consensus across samples
        instead of trusting whatever one generation happened to say is
        intended to improve precision."""
        counts = Counter()
        display_forms = {}
        first_seen_at = {}
        for j in range(SELF_CONSISTENCY_SAMPLES):
            raw_answer = self._generate_one(subject_entity, relation, do_sample=(j > 0), sample_index=j)
            parsed = self._parse_answer(raw_answer, relation, subject_entity)
            if not parsed:
                continue
            item = parsed[0]  # already capped to one item by USE_SINGLE_ANSWER_CAP
            key = self._normalize_for_voting(item)
            if not key:
                continue
            counts[key] += 1
            display_forms.setdefault(key, item)
            first_seen_at.setdefault(key, j)
        if not counts:
            return []
        best_key = max(counts, key=lambda k: (counts[k], -first_seen_at[k]))
        print(f"[debug] single-answer self-consistency for {relation!r} / {subject_entity!r}: "
              f"candidates={dict(counts)} -> picked {display_forms[best_key]!r}")
        return [display_forms[best_key]]

    def _retry_on_empty_predict(self, subject_entity: str, relation: str) -> List[str]:
        """Tries greedy first. Only if that comes back empty does it try
        a few sampled generations and return the first non-empty one.
        Never touches a subject where greedy already worked."""
        raw_answer = self._generate_one(subject_entity, relation, do_sample=False, sample_index=0)
        parsed = self._parse_answer(raw_answer, relation, subject_entity)
        if parsed:
            return parsed  # greedy already answered - leave it alone

        for attempt in range(1, MAX_RETRY_ATTEMPTS):
            raw_answer = self._generate_one(subject_entity, relation, do_sample=True, sample_index=attempt)
            parsed = self._parse_answer(raw_answer, relation, subject_entity)
            if parsed:
                print(f"[debug] retry-on-empty rescued {relation!r} / {subject_entity!r} "
                      f"on attempt {attempt + 1}/{MAX_RETRY_ATTEMPTS}: {parsed}")
                return parsed

        return []  # still nothing after every retry

    @staticmethod
    def _is_special(relation: str) -> bool:
        """True if this relation should be handled one item at a time
        through one of the multi-sample methods below, instead of a
        single plain batched greedy generation."""
        return (
            (USE_MAJORITY_VOTE_AWARDWONBY and relation in MAJORITY_VOTE_RELATIONS)
            or (USE_RETRY_ON_EMPTY and relation in RETRY_ON_EMPTY_RELATIONS)
            or (USE_SELF_CONSISTENCY_NUMERIC and relation in SELF_CONSISTENCY_NUMERIC_RELATIONS)
            or (USE_SELF_CONSISTENCY_SINGLE_ANSWER and relation in SELF_CONSISTENCY_SINGLE_ANSWER_RELATIONS)
        )

    def _predict_special(self, subject_entity: str, relation: str) -> List[str]:
        """Routes to whichever multi-sample method applies to this
        relation. Only call this when _is_special(relation) is True."""
        if USE_MAJORITY_VOTE_AWARDWONBY and relation in MAJORITY_VOTE_RELATIONS:
            return self._majority_vote_predict(subject_entity, relation)
        if USE_SELF_CONSISTENCY_NUMERIC and relation in SELF_CONSISTENCY_NUMERIC_RELATIONS:
            return self._numeric_self_consistency_predict(subject_entity, relation)
        if USE_SELF_CONSISTENCY_SINGLE_ANSWER and relation in SELF_CONSISTENCY_SINGLE_ANSWER_RELATIONS:
            return self._single_answer_self_consistency_predict(subject_entity, relation)
        return self._retry_on_empty_predict(subject_entity, relation)

    def generate_predictions(self, inputs: List[Dict[str, str]]) -> List[List[str]]:
        all_predictions = [None] * len(inputs)

        # Special relations are handled one item at a time instead of
        # batched, since each needs several generations - see
        # _predict_special / _is_special above.
        special_indices = []
        remaining_inputs = []
        remaining_original_indices = []
        for idx, item in enumerate(inputs):
            if self._is_special(item["Relation"]):
                special_indices.append(idx)
            else:
                remaining_inputs.append(item)
                remaining_original_indices.append(idx)

        for idx in special_indices:
            item = inputs[idx]
            all_predictions[idx] = self._predict_special(item["SubjectEntity"], item["Relation"])

        inputs = remaining_inputs  # everything below processes only the remaining items

        # Group by relation so a batch can share one max_new_tokens
        # budget - mixing a 16-token hasArea answer with a 300-token
        # awardWonBy one in the same batch would force the larger budget
        # on everyone.
        indices_by_relation: Dict[str, List[int]] = defaultdict(list)
        for idx, item in enumerate(inputs):
            indices_by_relation[item["Relation"]].append(idx)

        for relation, indices in indices_by_relation.items():
            if USE_PER_RELATION_MAX_TOKENS:
                max_new_tokens = MAX_NEW_TOKENS_BY_RELATION.get(relation, self.max_new_tokens)
            else:
                max_new_tokens = self.max_new_tokens

            for start in range(0, len(indices), self.batch_size):
                batch_indices = indices[start:start + self.batch_size]
                batch = [inputs[i] for i in batch_indices]
                prompts = [self._build_prompt(item["SubjectEntity"], item["Relation"]) for item in batch]

                chat_prompts = [
                    self.tokenizer.apply_chat_template(
                        [{"role": "user", "content": p}],
                        add_generation_prompt=True,
                        enable_thinking=self.enable_thinking,
                        tokenize=False,
                    )
                    for p in prompts
                ]

                encoded = self.tokenizer(chat_prompts, return_tensors="pt", padding=True).to(self.model.device)

                with torch.no_grad():
                    output_ids = self.model.generate(
                        **encoded, max_new_tokens=max_new_tokens, do_sample=False
                    )

                new_tokens = output_ids[:, encoded["input_ids"].shape[1]:]
                decoded = self.tokenizer.batch_decode(new_tokens, skip_special_tokens=True)

                for idx, item, raw_answer in zip(batch_indices, batch, decoded):
                    original_idx = remaining_original_indices[idx]
                    all_predictions[original_idx] = self._parse_answer(raw_answer, item["Relation"], item["SubjectEntity"])

        return all_predictions


class OllamaBaselineModel(HFTransformersBaselineModel):
    """
    Same prompting/few-shot/parsing logic as HFTransformersBaselineModel,
    inherited as-is since none of it depends on transformers/torch. Only
    the generation call changes: it talks to a local Ollama server
    instead of loading a `transformers` model directly, which is how
    Gemma/Llama run without touching HF's gating at all.

    Trade-off: Ollama models are always GGUF-quantized, so they don't get
    the "unquantized" edge that helped Qwen3.5-9B/Qwen2.5-7B/Qwen3-8B in
    earlier comparisons.
    """

    def __init__(
        self,
        model_tag: str,
        prompt_templates_file: str,
        max_new_tokens: int = 64,
        few_shot: int = 5,
        train_data_file: str = None,
        **kwargs,
    ):
        # Deliberately not calling HFTransformersBaselineModel.__init__ -
        # there's no tokenizer/transformers model to load here.
        AbstractModel.__init__(self)
        self.model_tag = model_tag
        self.max_new_tokens = max_new_tokens
        self.enable_thinking = False  # unused in this class, kept for interface parity
        self.few_shot = few_shot
        self.prompt_templates = self._load_prompt_templates(prompt_templates_file)
        self.few_shot_examples = (
            self._load_few_shot_examples(train_data_file) if train_data_file else {}
        )
        self._relation_rngs: Dict[str, random.Random] = {}

    def _generate_one(self, subject_entity: str, relation: str, do_sample: bool,
                       sample_index: int = 0) -> str:
        prompt = self._build_prompt(subject_entity, relation)
        max_new_tokens = (
            MAX_NEW_TOKENS_BY_RELATION.get(relation, self.max_new_tokens)
            if USE_PER_RELATION_MAX_TOKENS else self.max_new_tokens
        )
        # Ollama runs the model in its own separate process (llama.cpp
        # under the hood), so torch.manual_seed() in the main process has
        # no effect on it - reproducibility is set here instead. Varying
        # by sample_index (42 + sample_index) keeps a full run
        # reproducible without collapsing every sampled attempt in
        # majority voting/retry into identical output.
        options = {"num_predict": max_new_tokens, "seed": 42 + sample_index}
        options.update({"temperature": 0.7, "top_p": 0.9} if do_sample else {"temperature": 0.0})

        response = ollama.chat(
            model=self.model_tag,
            messages=[{"role": "user", "content": prompt}],
            options=options,
        )
        return response["message"]["content"]

    def generate_predictions(self, inputs: List[Dict[str, str]]) -> List[List[str]]:
        # Ollama only handles one request at a time anyway, so there's no
        # benefit to replicating the transformers-style batching here.
        all_predictions = []
        for item in inputs:
            subject, relation = item["SubjectEntity"], item["Relation"]
            if self._is_special(relation):
                all_predictions.append(self._predict_special(subject, relation))
            else:
                raw_answer = self._generate_one(subject, relation, do_sample=False, sample_index=0)
                all_predictions.append(self._parse_answer(raw_answer, relation, subject))
        return all_predictions


# Wires everything together and runs it end to end.
def main():
    data_path = os.path.join(REPO_DIR, "data", "test.jsonl")
    prompts_path = os.path.join(REPO_DIR, "prompt_templates", "question_prompts.csv")
    train_path = os.path.join(REPO_DIR, "data", "train.jsonl")

    with open(data_path, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]

    # Full val.jsonl run, no slicing. Same prompting setup (all 4 flags,
    # fixed seed, isolated per-relation RNG) across every model compared,
    # so any score difference comes from the model, not the prompting.
    if USE_OLLAMA_MODEL:
        model = OllamaBaselineModel(
            model_tag=OLLAMA_MODEL_TAG,
            prompt_templates_file=prompts_path,
            max_new_tokens=64,
            few_shot=5,
            train_data_file=train_path,
        )
    else:
        model = HFTransformersBaselineModel(
            llm_path="google/gemma-3-12b-it",
            prompt_templates_file=prompts_path,
            max_new_tokens=64,
            batch_size=2,
            use_quantization=True,               # 12B in fp16 (~24GB) doesn't fit a T4's 16GB
            enable_thinking=False,
            few_shot=5,
            train_data_file=train_path,
        )

    model_inputs = [
        {"SubjectEntity": row["SubjectEntity"], "Relation": row["Relation"]} for row in data
    ]
    predictions = model.generate_predictions(model_inputs)

    output_rows = [
        {"SubjectEntity": row["SubjectEntity"], "Relation": row["Relation"], "ObjectEntities": objects}
        for row, objects in zip(data, predictions)
    ]

    output_path = "/kaggle/working/predictions.jsonl"
    with open(output_path, "w", encoding="utf-8") as f:
        for row in output_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Wrote {len(output_rows)} predictions to {output_path}")


if __name__ == "__main__":
    main()