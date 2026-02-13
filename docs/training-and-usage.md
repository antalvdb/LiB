# Training and Using the LiB Tokenizer

## Overview

LiB (Less is Better) is a cognitively inspired tokenizer that differs from standard subword tokenizers (BPE, WordPiece, Unigram) in a fundamental way: it builds a hierarchical vocabulary that includes **subwords**, **words**, and **supra-words** (multi-word units like "of the", "in the morning"). This makes it particularly interesting for memory-based LLMs, where the granularity and cognitive plausibility of tokenization may affect how the model learns to store and retrieve information.

This guide covers two scenarios:

1. **Training**: comparing LiB against GPT-2's tokenizer when training a memory-based LLM
2. **Using**: plugging a trained LiB tokenizer into standard HuggingFace workflows

## Prerequisites

Build and install the forked tokenizers package and the LiB wrapper:

```bash
# Build the Rust tokenizers with LiB support
cd tokenizers-fork/bindings/python
pip install maturin
maturin develop --release

# Install the LiB Python wrapper
cd ../../..
pip install -e .

# Also need transformers for PreTrainedTokenizerFast
pip install transformers
```

---

## Part 1: Training LiB vs GPT-2 for a Memory-Based LLM

### Why LiB for memory-based models?

Standard tokenizers like GPT-2's BPE optimize purely for compression: they merge the most frequent byte pairs until they reach a target vocabulary size. This produces subword units that are statistically optimal but have no relationship to how humans segment language.

LiB takes a different approach. It simulates a cognitively plausible learning process where:

- Units **compete** for a place in the vocabulary based on whether they improve compression
- Units that don't prove useful are **pruned** (forgotten)
- The vocabulary naturally develops a **hierarchy**: characters < subwords < words < supra-words

For a memory-based LLM, this matters because:

- **Supra-word tokens** (e.g. "of the", "I don't") let the model store and retrieve common phrases as single memory entries
- **Priority ordering** means the most useful units have the lowest token IDs, creating a natural frequency-based encoding
- The **variable granularity** may help the model learn different levels of abstraction

### Training both tokenizers

```python
from tokenizers import Tokenizer
from tokenizers.decoders import Fuse
from tokenizers.models import LiB
from tokenizers.trainers import LiBTrainer
from transformers import GPT2TokenizerFast

CORPUS_FILES = ["path/to/your/training_corpus.txt"]

# --- Train LiB tokenizer ---
lib_tok = Tokenizer(LiB())
lib_tok.decoder = Fuse()  # LiB tokens may contain spaces; Fuse joins without separator

lib_trainer = LiBTrainer(
    vocab_size=32000,
    num_epochs=50000,
    max_len=12,         # max token length in characters
    memory_in=0.25,     # probability of considering a candidate
    memory_out=0.0001,  # fraction of vocab pruned per epoch
    update_rate=0.2,    # how far tokens move on reward/punishment
    seed=42,            # for reproducibility
)

lib_tok.train(CORPUS_FILES, lib_trainer)
lib_tok.save("lib-tokenizer.json")

print(f"LiB vocab size: {lib_tok.get_vocab_size()}")

# --- Load GPT-2 tokenizer (pre-trained, 50257 tokens) ---
gpt2_tok = GPT2TokenizerFast.from_pretrained("gpt2")

print(f"GPT-2 vocab size: {gpt2_tok.vocab_size}")
```

### Comparing tokenization behavior

```python
texts = [
    "The cat sat on the mat.",
    "I don't think that's going to work.",
    "The United States of America declared independence.",
    "She couldn't believe what she was hearing.",
]

for text in texts:
    lib_enc = lib_tok.encode(text)
    gpt2_enc = gpt2_tok(text)

    print(f"\nInput: {text}")
    print(f"  LiB:  {lib_enc.tokens}  ({len(lib_enc.tokens)} tokens)")
    print(f"  GPT2: {gpt2_tok.convert_ids_to_tokens(gpt2_enc['input_ids'])}  ({len(gpt2_enc['input_ids'])} tokens)")
```

You'll typically see that:

- LiB produces **supra-word tokens** like "on the", "of the", "I don't" — common multi-word expressions become single tokens
- GPT-2 splits along **subword boundaries** driven by byte-pair frequency — "don't" might become ["don", "'t"]
- LiB's token count is often **lower** for natural language because supra-words compress common phrases

### Wrapping for HuggingFace model training

To use either tokenizer with a HuggingFace model, wrap the LiB tokenizer:

```python
from lib_tokenizers import LiBTokenizerFast

# Train from scratch
lib_hf = LiBTokenizerFast.train_new(
    CORPUS_FILES,
    vocab_size=32000,
    num_epochs=50000,
    seed=42,
    special_tokens=["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"],
)

# Save in HuggingFace format
lib_hf.save_pretrained("./lib-tokenizer-hf")

# Now both tokenizers can be used identically for model training:
from transformers import AutoConfig, AutoModelForCausalLM, TrainingArguments, Trainer

# With LiB:
model_config = AutoConfig.from_pretrained(
    "gpt2",
    vocab_size=lib_hf.vocab_size,
)
model = AutoModelForCausalLM.from_config(model_config)
model.resize_token_embeddings(lib_hf.vocab_size)

# Training proceeds the same way as with any HuggingFace tokenizer
```

### LiB training parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `vocab_size` | 30000 | Target vocabulary size. LiB may produce fewer tokens if it prunes aggressively. |
| `num_epochs` | 10000 | Number of learning iterations. Each epoch processes one sentence. More epochs = richer vocabulary. For large corpora, 50K-100K is reasonable. |
| `max_len` | 12 | Maximum token length in characters. Supra-words like "in the morning" need higher values. 12 is a good default. |
| `memory_in` | 0.25 | Probability of considering a new candidate unit. Lower = more selective. |
| `memory_out` | 0.0001 | Fraction of lowest-priority units pruned each epoch. Controls forgetting rate. |
| `update_rate` | 0.2 | How far a token moves in the priority list when rewarded/punished. Higher = faster reordering. |
| `life` | 10 | Initial life counter for new units. Units that aren't useful lose life and are eventually pruned. |
| `seed` | None | Random seed. Set for reproducible training. |
| `deterministic` | False | If True, processes sentences in order instead of sampling randomly. Useful for debugging. |
| `special_tokens` | [] | Special tokens (e.g. [PAD], [UNK]) to add to the vocabulary. |

### Metrics for comparison

When comparing LiB vs GPT-2 tokenization for your memory-based LLM, consider measuring:

```python
def tokenization_stats(tokenizer_encode, texts):
    """Compute basic tokenization statistics."""
    total_tokens = 0
    total_chars = 0
    token_lengths = []

    for text in texts:
        enc = tokenizer_encode(text)
        tokens = enc if isinstance(enc, list) else enc.tokens
        total_tokens += len(tokens)
        total_chars += len(text)
        token_lengths.extend(len(t) for t in tokens)

    return {
        "compression_ratio": total_chars / total_tokens,
        "avg_token_length": sum(token_lengths) / len(token_lengths),
        "tokens_per_text": total_tokens / len(texts),
    }

# Compare
lib_stats = tokenization_stats(lib_tok.encode, texts)
gpt2_stats = tokenization_stats(
    lambda t: gpt2_tok.convert_ids_to_tokens(gpt2_tok(t)["input_ids"]),
    texts,
)

print("LiB:", lib_stats)
print("GPT-2:", gpt2_stats)
```

For a memory-based LLM, also track:

- **Sequence length**: shorter sequences (from supra-word compression) mean less memory and faster attention
- **Vocabulary coverage**: what fraction of text gets tokenized as known multi-character units vs single-character fallbacks
- **Supra-word ratio**: `len([t for t in vocab if ' ' in t]) / len(vocab)` — higher means more multi-word units

---

## Part 2: Using a Trained LiB Tokenizer in HuggingFace Workflows

Once trained, a LiB tokenizer works like any other HuggingFace tokenizer.

### Save and load

```python
from lib_tokenizers import LiBTokenizerFast

# Save
tokenizer.save_pretrained("./my-lib-tokenizer")

# Load
tokenizer = LiBTokenizerFast.from_pretrained("./my-lib-tokenizer")
```

The saved directory contains:

- `tokenizer.json` — the full tokenizer state (model with `"type": "LiB"`, vocab, config)
- `tokenizer_config.json` — HuggingFace metadata (special tokens, model type)
- `special_tokens_map.json` — special token definitions

### Encoding text

```python
# Single text
output = tokenizer("The cat sat on the mat.")
print(output["input_ids"])       # [42, 7, 15, ...]
print(output["attention_mask"])  # [1, 1, 1, ...]

# Batch
outputs = tokenizer(
    ["Hello world", "The cat sat", "LiB tokenizer"],
    padding=True,
    truncation=True,
    max_length=128,
    return_tensors="pt",  # or "tf", "np"
)
```

### Decoding

```python
text = tokenizer.decode(output["input_ids"])
# or skip special tokens:
text = tokenizer.decode(output["input_ids"], skip_special_tokens=True)
```

### Using with a HuggingFace model

```python
from transformers import AutoModelForCausalLM, pipeline

model = AutoModelForCausalLM.from_pretrained("./my-model")
tokenizer = LiBTokenizerFast.from_pretrained("./my-lib-tokenizer")

# Text generation pipeline
gen = pipeline("text-generation", model=model, tokenizer=tokenizer)
result = gen("The cat sat on", max_new_tokens=20)
print(result[0]["generated_text"])
```

### Using with datasets

```python
from datasets import load_dataset

dataset = load_dataset("text", data_files="corpus.txt")

def tokenize_fn(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=512,
    )

tokenized = dataset.map(tokenize_fn, batched=True, remove_columns=["text"])
```

### Using with the Trainer API

```python
from transformers import (
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

tokenizer = LiBTokenizerFast.from_pretrained("./my-lib-tokenizer")
model = AutoModelForCausalLM.from_pretrained("./my-model")

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,  # causal LM
)

training_args = TrainingArguments(
    output_dir="./output",
    per_device_train_batch_size=8,
    num_train_epochs=3,
    learning_rate=5e-5,
    save_strategy="epoch",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"],
    data_collator=data_collator,
)

trainer.train()
```

### Loading a tokenizer.json directly

If you only have the raw `tokenizer.json` (e.g. from `Tokenizer.save()`), you can load it:

```python
from tokenizers import Tokenizer

# Low-level tokenizers API
tok = Tokenizer.from_file("tokenizer.json")
encoding = tok.encode("hello world")

# Or wrap it for HuggingFace
from lib_tokenizers import LiBTokenizerFast
hf_tok = LiBTokenizerFast(tokenizer_file="tokenizer.json")
```

---

## Reference

- Yang, J., Frank, S. L., & van den Bosch, A. (2020). [Less is Better: A cognitively inspired unsupervised model for language segmentation](https://aclanthology.org/2020.cogalex-1.4/). *Proceedings of the Workshop on the Cognitive Aspects of the Lexicon*, 33-45.
