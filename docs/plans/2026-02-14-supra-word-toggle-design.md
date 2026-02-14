# Supra-Word Token Toggle

## Problem

The LiB tokenizer always infers supra-word tokens (multi-word units containing spaces) during encoding. For benchmarking the tokenizer against itself (with/without supra-word tokens) and against other tokenizers, we need the option to disable supra-word token inference at runtime.

## Design

### Approach: Filter in trie matching

Add a `use_supra_words: bool` field to `LiBModel`. When `false`, the trie matching methods (`match_two`, `match_longest`) skip any match whose token string contains a space. The trie continues walking past space-containing tokens so shorter non-space prefixes are still found.

### Changes by layer

**1. Rust core — `LiBModel` (model.rs)**
- Add `pub use_supra_words: bool` field, default `true`
- Pass `!self.use_supra_words` as `skip_spaces` to `match_two()`, `match_longest()` in `tokenize()` and `choose_best()`

**2. Rust core — `TrieList` (trie.rs)**
- Add `skip_spaces: bool` parameter to `match_longest()` and `match_two()`
- When `skip_spaces` is true: if a trie node terminates a token and the accumulated string contains a space, don't record it as a match (but keep walking the trie)

**3. Serialization (serialization.rs)**
- Serialize `use_supra_words` field
- Default to `true` on deserialization when absent (backward compat)

**4. PyO3 bindings (models.rs)**
- Add `use_supra_words` constructor parameter to `PyLiB` (default `true`)
- Add getter/setter using existing `getter!`/`setter!` macros

**5. Python wrapper (tokenizer.py)**
- No changes needed — users access via `tokenizer.backend_tokenizer.model.use_supra_words = False`

### Tests

- Rust: `use_supra_words=false` skips "the cat", falls back to "the" + "cat"
- Rust: `use_supra_words=true` (default) matches "the cat" as single token
- Serialization roundtrip preserves the field
