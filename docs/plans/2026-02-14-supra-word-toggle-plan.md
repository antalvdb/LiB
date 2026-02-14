# Supra-Word Toggle Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a `use_supra_words` toggle to the LiB tokenizer so supra-word tokens (multi-word units containing spaces) can be disabled at runtime for benchmarking.

**Architecture:** Add a `use_supra_words: bool` field to `LiBModel` (default `true`). When `false`, the trie matching methods skip tokens containing spaces. Expose through serialization, PyO3 bindings, and Python getter/setter.

**Tech Stack:** Rust (tokenizers crate, serde), PyO3, Python

---

## Task 1: Add `skip_spaces` parameter to `TrieList::match_longest`

**Files:**
- Modify: `tokenizers-fork/tokenizers/src/models/lib/trie.rs:157-175` (match_longest)
- Test: `tokenizers-fork/tokenizers/src/models/lib/trie.rs` (tests module)

**Step 1: Write the failing test**

Add to the `tests` module at the bottom of `trie.rs`:

```rust
#[test]
fn test_match_longest_skip_spaces() {
    let mut tl = TrieList::new();
    tl.append("the".to_string(), 1);
    tl.append("the cat".to_string(), 1);
    tl.append("the cat sat".to_string(), 1);

    // With skip_spaces=false, should match the longest including spaces
    let result = tl.match_longest("the cat sat on", false);
    assert_eq!(result, Some(("the cat sat".to_string(), 2)));

    // With skip_spaces=true, should skip space-containing tokens
    let result = tl.match_longest("the cat sat on", true);
    assert_eq!(result, Some(("the".to_string(), 0)));

    // A token without spaces is still matched even with skip_spaces=true
    let result = tl.match_longest("the", true);
    assert_eq!(result, Some(("the".to_string(), 0)));

    // No match at all
    let result = tl.match_longest("xyz", true);
    assert_eq!(result, None);
}
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/bosch075/Software/LiB/.worktrees/hf-tokenizer/tokenizers-fork/tokenizers && cargo test --lib models::lib::trie::tests::test_match_longest_skip_spaces`
Expected: FAIL — `match_longest` doesn't accept a `bool` parameter.

**Step 3: Update `match_longest` signature and implementation**

Change `match_longest` in `trie.rs` (lines 157-175) to:

```rust
pub fn match_longest(&self, input: &str, skip_spaces: bool) -> Option<(String, usize)> {
    let mut node_idx: usize = 0;
    let mut best: Option<(String, usize)> = None;
    let mut consumed = String::new();

    for ch in input.chars() {
        match self.nodes[node_idx].children.get(&ch) {
            Some(&next) => {
                node_idx = next;
                consumed.push(ch);
                if let Some(tok_id) = self.nodes[node_idx].token_index {
                    if !skip_spaces || !consumed.contains(' ') {
                        best = Some((consumed.clone(), tok_id));
                    }
                }
            }
            None => break,
        }
    }
    best
}
```

**Step 4: Fix existing callers and tests that call `match_longest` without the new param**

Update `test_match_longest` (line 360) — add `false` as second arg to all three calls:

```rust
let result = tl.match_longest("hello world", false);
```
```rust
let result = tl.match_longest("help me", false);
```
```rust
let result = tl.match_longest("hat", false);
```

Update `test_match_no_match` (line 407) — add `false`:

```rust
assert_eq!(tl.match_longest("anything", false), None);
```
```rust
assert_eq!(tl2.match_longest("abc", false), None);
```

Update `test_unicode` (line 432, 435) — add `false`:

```rust
let result = tl.match_longest("\u{00e9}t\u{00e9} hello", false);
```
```rust
let result = tl.match_longest("\u{4e16}\u{754c}!", false);
```

**Step 5: Run tests to verify they pass**

Run: `cd /Users/bosch075/Software/LiB/.worktrees/hf-tokenizer/tokenizers-fork/tokenizers && cargo test --lib models::lib::trie`
Expected: ALL PASS

**Step 6: Commit**

```bash
git add tokenizers-fork/tokenizers/src/models/lib/trie.rs
git commit -m "feat: add skip_spaces parameter to TrieList::match_longest"
```

---

## Task 2: Add `skip_spaces` parameter to `TrieList::match_two`

**Files:**
- Modify: `tokenizers-fork/tokenizers/src/models/lib/trie.rs:179-202` (match_two)
- Test: `tokenizers-fork/tokenizers/src/models/lib/trie.rs` (tests module)

**Step 1: Write the failing test**

Add to the `tests` module in `trie.rs`:

```rust
#[test]
fn test_match_two_skip_spaces() {
    let mut tl = TrieList::new();
    tl.append("the".to_string(), 1);
    tl.append("the cat".to_string(), 1);
    tl.append("the cat sat".to_string(), 1);

    // With skip_spaces=false, returns longest and second-longest (both can have spaces)
    let (best, second) = tl.match_two("the cat sat on", false);
    assert_eq!(best, Some(("the cat sat".to_string(), 2)));
    assert_eq!(second, Some(("the cat".to_string(), 1)));

    // With skip_spaces=true, only non-space tokens are considered
    let (best, second) = tl.match_two("the cat sat on", true);
    assert_eq!(best, Some(("the".to_string(), 0)));
    assert_eq!(second, None);
}
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/bosch075/Software/LiB/.worktrees/hf-tokenizer/tokenizers-fork/tokenizers && cargo test --lib models::lib::trie::tests::test_match_two_skip_spaces`
Expected: FAIL — `match_two` doesn't accept a `bool` parameter.

**Step 3: Update `match_two` signature and implementation**

Change `match_two` in `trie.rs` (lines 179-202) to:

```rust
pub fn match_two(
    &self,
    input: &str,
    skip_spaces: bool,
) -> (Option<(String, usize)>, Option<(String, usize)>) {
    let mut node_idx: usize = 0;
    let mut best: Option<(String, usize)> = None;
    let mut second: Option<(String, usize)> = None;
    let mut consumed = String::new();

    for ch in input.chars() {
        match self.nodes[node_idx].children.get(&ch) {
            Some(&next) => {
                node_idx = next;
                consumed.push(ch);
                if let Some(tok_id) = self.nodes[node_idx].token_index {
                    if !skip_spaces || !consumed.contains(' ') {
                        second = best.clone();
                        best = Some((consumed.clone(), tok_id));
                    }
                }
            }
            None => break,
        }
    }
    (best, second)
}
```

**Step 4: Fix existing callers and tests that call `match_two` without the new param**

Update `test_match_two` (line 388, 393, 398) — add `false` as second arg:

```rust
let (best, second) = tl.match_two("hello world", false);
```
```rust
let (best, second) = tl.match_two("help", false);
```
```rust
let (best, second) = tl.match_two("hat", false);
```

Update `test_match_no_match` (line 409) — add `false`:

```rust
let (best, second) = tl.match_two("anything", false);
```

**Step 5: Run tests to verify they pass**

Run: `cd /Users/bosch075/Software/LiB/.worktrees/hf-tokenizer/tokenizers-fork/tokenizers && cargo test --lib models::lib::trie`
Expected: ALL PASS

**Step 6: Commit**

```bash
git add tokenizers-fork/tokenizers/src/models/lib/trie.rs
git commit -m "feat: add skip_spaces parameter to TrieList::match_two"
```

---

## Task 3: Add `use_supra_words` field to `LiBModel` and wire into `tokenize`

**Files:**
- Modify: `tokenizers-fork/tokenizers/src/models/lib/model.rs:7-21` (struct + constructor)
- Modify: `tokenizers-fork/tokenizers/src/models/lib/model.rs:32-63` (choose_best)
- Modify: `tokenizers-fork/tokenizers/src/models/lib/model.rs:75-121` (tokenize)
- Test: `tokenizers-fork/tokenizers/src/models/lib/model.rs` (tests module)

**Step 1: Write the failing test**

Add to the `tests` module in `model.rs`:

```rust
#[test]
fn test_tokenize_supra_word_disabled() {
    let mut model = make_model(&["t", "h", "e", "c", "a", "the", "cat", "the cat"]);
    // Default: supra-words enabled
    let tokens = model.tokenize("the cat").unwrap();
    assert_eq!(tokens.len(), 1);
    assert_eq!(tokens[0].value, "the cat");

    // Disable supra-words
    model.use_supra_words = false;
    let tokens = model.tokenize("the cat").unwrap();
    assert_eq!(tokens.len(), 2);
    assert_eq!(tokens[0].value, "the");
    assert_eq!(tokens[1].value, " ");  // space is a single unknown char
    // Note: "cat" should follow
    assert_eq!(tokens[2].value, "cat");
}
```

Wait — the space character is not in the vocab, so it will be emitted as an unknown char. Let me adjust:

```rust
#[test]
fn test_tokenize_supra_word_disabled() {
    let mut model = make_model(&["t", "h", "e", " ", "c", "a", "the", "cat", "the cat"]);
    // Default: supra-words enabled
    let tokens = model.tokenize("the cat").unwrap();
    assert_eq!(tokens.len(), 1);
    assert_eq!(tokens[0].value, "the cat");

    // Disable supra-words
    model.use_supra_words = false;
    let tokens = model.tokenize("the cat").unwrap();
    assert_eq!(tokens.len(), 3);
    assert_eq!(tokens[0].value, "the");
    assert_eq!(tokens[1].value, " ");
    assert_eq!(tokens[2].value, "cat");
}
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/bosch075/Software/LiB/.worktrees/hf-tokenizer/tokenizers-fork/tokenizers && cargo test --lib models::lib::model::tests::test_tokenize_supra_word_disabled`
Expected: FAIL — `use_supra_words` field doesn't exist.

**Step 3: Add the field to `LiBModel` struct**

In `model.rs`, update the struct (lines 8-12):

```rust
#[derive(Clone, Debug, PartialEq)]
pub struct LiBModel {
    pub(crate) trie: TrieList,
    pub max_len: usize,
    pub unk_token: Option<String>,
    pub use_supra_words: bool,
}
```

Update `new()` (lines 15-21):

```rust
pub fn new(max_len: usize, unk_token: Option<String>) -> Self {
    Self {
        trie: TrieList::new(),
        max_len,
        unk_token,
        use_supra_words: true,
    }
}
```

Update `Default` impl (lines 66-70):

```rust
impl Default for LiBModel {
    fn default() -> Self {
        Self::new(12, None)
    }
}
```

(Default impl is unchanged since `new` now sets the field.)

**Step 4: Wire `use_supra_words` into `tokenize()` and `choose_best()`**

Compute the skip flag once at the top of `tokenize()`. In `tokenize()` (line 75+):

```rust
fn tokenize(&self, sequence: &str) -> Result<Vec<Token>> {
    if sequence.is_empty() {
        return Ok(Vec::new());
    }

    let skip_spaces = !self.use_supra_words;
    let mut tokens = Vec::new();
    let mut byte_pos: usize = 0;

    while byte_pos < sequence.len() {
        let rest = &sequence[byte_pos..];
        let window: String = rest.chars().take(self.max_len).collect();

        let (best, second) = self.trie.match_two(&window, skip_spaces);

        match (best, second) {
            (Some(b), Some(s)) => {
                let chosen = self.choose_best(sequence, byte_pos, &b, &s, skip_spaces);
                let tok_str = &chosen.0;
                let tok_id = chosen.1 as u32;
                let byte_end = byte_pos + tok_str.len();
                tokens.push(Token::new(tok_id, tok_str.clone(), (byte_pos, byte_end)));
                byte_pos = byte_end;
            }
            (Some(b), None) => {
                let byte_end = byte_pos + b.0.len();
                tokens.push(Token::new(b.1 as u32, b.0.clone(), (byte_pos, byte_end)));
                byte_pos = byte_end;
            }
            _ => {
                let ch = rest.chars().next().unwrap();
                let ch_str = ch.to_string();
                let byte_end = byte_pos + ch.len_utf8();
                let id = self.trie.token_to_id(&ch_str)
                    .map(|id| id as u32)
                    .unwrap_or(0);
                tokens.push(Token::new(id, ch_str, (byte_pos, byte_end)));
                byte_pos = byte_end;
            }
        }
    }

    Ok(tokens)
}
```

Update `choose_best` signature (line 32+) to accept `skip_spaces`:

```rust
fn choose_best<'a>(
    &self,
    sequence: &str,
    byte_pos: usize,
    best: &'a (String, usize),
    second: &'a (String, usize),
    skip_spaces: bool,
) -> &'a (String, usize) {
    let remaining_after_best = &sequence[byte_pos + best.0.len()..];
    let remaining_after_second = &sequence[byte_pos + second.0.len()..];

    if remaining_after_best.is_empty() {
        return best;
    }

    let window_best: String = remaining_after_best.chars().take(self.max_len).collect();
    let window_second: String = remaining_after_second.chars().take(self.max_len).collect();

    let has_next_best = self.trie.match_longest(&window_best, skip_spaces).is_some();
    let has_next_second = self.trie.match_longest(&window_second, skip_spaces).is_some();

    match (has_next_best, has_next_second) {
        (true, true) | (false, false) => best,
        (true, false) => best,
        (false, true) => second,
    }
}
```

**Step 5: Fix the `test_tokenize_supra_word_disabled` test**

The test written in Step 1 expects 3 tokens but we used `assert_eq!(tokens.len(), 3)` after the `assert_eq!(tokens.len(), 1)` block. The test needs to account for the fact the function signature now works. Let me write the corrected test:

```rust
#[test]
fn test_tokenize_supra_word_disabled() {
    let mut model = make_model(&["t", "h", "e", " ", "c", "a", "the", "cat", "the cat"]);

    // Default: supra-words enabled — "the cat" is one token
    let tokens = model.tokenize("the cat").unwrap();
    assert_eq!(tokens.len(), 1);
    assert_eq!(tokens[0].value, "the cat");

    // Disable supra-words — falls back to word-level + space
    model.use_supra_words = false;
    let tokens = model.tokenize("the cat").unwrap();
    assert_eq!(tokens.len(), 3, "Expected [the, ' ', cat], got: {:?}",
        tokens.iter().map(|t| &t.value).collect::<Vec<_>>());
    assert_eq!(tokens[0].value, "the");
    assert_eq!(tokens[1].value, " ");
    assert_eq!(tokens[2].value, "cat");
}
```

**Step 6: Run tests to verify they pass**

Run: `cd /Users/bosch075/Software/LiB/.worktrees/hf-tokenizer/tokenizers-fork/tokenizers && cargo test --lib models::lib::model`
Expected: ALL PASS

**Step 7: Commit**

```bash
git add tokenizers-fork/tokenizers/src/models/lib/model.rs
git commit -m "feat: add use_supra_words field to LiBModel"
```

---

## Task 4: Update serialization for `use_supra_words`

**Files:**
- Modify: `tokenizers-fork/tokenizers/src/models/lib/serialization.rs:7-25` (Serialize impl)
- Modify: `tokenizers-fork/tokenizers/src/models/lib/serialization.rs:27-97` (Deserialize impl)
- Test: `tokenizers-fork/tokenizers/src/models/lib/serialization.rs` (tests module)

**Step 1: Write the failing test**

Add to the `tests` module in `serialization.rs`:

```rust
#[test]
fn test_serialization_use_supra_words() {
    // Serialize with use_supra_words=false
    let mut model = LiBModel::new(12, None);
    model.trie.append("hello".to_string(), 10);
    model.trie.append("hello world".to_string(), 10);
    model.use_supra_words = false;

    let json = serde_json::to_string(&model).unwrap();
    assert!(json.contains("\"use_supra_words\":false") || json.contains("\"use_supra_words\": false"),
        "JSON should contain use_supra_words: {}", json);

    let deserialized: LiBModel = serde_json::from_str(&json).unwrap();
    assert_eq!(deserialized.use_supra_words, false);

    // Backward compat: JSON without the field defaults to true
    let old_json = r#"{"type":"LiB","max_len":12,"unk_token":null,"vocab":[["a",10,0]]}"#;
    let old_model: LiBModel = serde_json::from_str(old_json).unwrap();
    assert_eq!(old_model.use_supra_words, true);
}
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/bosch075/Software/LiB/.worktrees/hf-tokenizer/tokenizers-fork/tokenizers && cargo test --lib models::lib::serialization::tests::test_serialization_use_supra_words`
Expected: FAIL — `use_supra_words` not serialized.

**Step 3: Update Serialize impl**

In `serialization.rs`, update the `Serialize` impl:

```rust
impl Serialize for LiBModel {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let vocab: Vec<(&str, i32, u64)> = self
            .trie
            .iter()
            .map(|(_, entry)| (entry.token.as_str(), entry.life, entry.frequency))
            .collect();

        let mut s = serializer.serialize_struct("LiBModel", 5)?;
        s.serialize_field("type", "LiB")?;
        s.serialize_field("max_len", &self.max_len)?;
        s.serialize_field("unk_token", &self.unk_token)?;
        s.serialize_field("use_supra_words", &self.use_supra_words)?;
        s.serialize_field("vocab", &vocab)?;
        s.end()
    }
}
```

**Step 4: Update Deserialize impl**

Add `UseSupraWords` to the `Field` enum and handle it in `visit_map`:

```rust
#[derive(Deserialize)]
#[serde(field_identifier, rename_all = "snake_case")]
enum Field {
    Type,
    MaxLen,
    UnkToken,
    UseSupraWords,
    Vocab,
}
```

In `visit_map`, add variable and match arm:

```rust
let mut use_supra_words: Option<bool> = None;
```

Add match arm:

```rust
Field::UseSupraWords => {
    use_supra_words = Some(map.next_value()?);
}
```

After constructing the model, set the field:

```rust
let mut model = LiBModel::new(max_len, unk_token);
model.use_supra_words = use_supra_words.unwrap_or(true);
for (token, life, _freq) in vocab {
    model.trie.append(token, life);
}
```

**Step 5: Run tests to verify they pass**

Run: `cd /Users/bosch075/Software/LiB/.worktrees/hf-tokenizer/tokenizers-fork/tokenizers && cargo test --lib models::lib::serialization`
Expected: ALL PASS

**Step 6: Commit**

```bash
git add tokenizers-fork/tokenizers/src/models/lib/serialization.rs
git commit -m "feat: serialize use_supra_words with backward compat"
```

---

## Task 5: Expose `use_supra_words` in PyO3 bindings

**Files:**
- Modify: `tokenizers-fork/bindings/python/src/models.rs:939-961` (PyLiB)

**Step 1: Add constructor parameter and getter/setter**

Update the `PyLiB` `new` function signature and body:

```rust
#[pymethods]
impl PyLiB {
    #[new]
    #[pyo3(signature = (vocab=None, max_len=12, unk_token=None, use_supra_words=true), text_signature = "(self, vocab=None, max_len=12, unk_token=None, use_supra_words=True)")]
    fn new(
        vocab: Option<HashMap<String, u32>>,
        max_len: usize,
        unk_token: Option<String>,
        use_supra_words: bool,
    ) -> PyResult<(Self, PyModel)> {
        let mut model = LiBModel::new(max_len, unk_token);
        model.use_supra_words = use_supra_words;

        if let Some(vocab_map) = vocab {
            let mut entries: Vec<(String, u32)> = vocab_map.into_iter().collect();
            entries.sort_by_key(|(_, id)| *id);
            for (token, _) in entries {
                model.add_token(token, 10);
            }
        }

        Ok((PyLiB {}, model.into()))
    }

    #[getter]
    fn get_use_supra_words(self_: PyRef<Self>) -> bool {
        getter!(self_, LiB, use_supra_words)
    }

    #[setter]
    fn set_use_supra_words(self_: PyRef<Self>, use_supra_words: bool) {
        setter!(self_, LiB, use_supra_words, use_supra_words);
    }
}
```

**Step 2: Build the Python bindings to verify compilation**

Run: `cd /Users/bosch075/Software/LiB/.worktrees/hf-tokenizer/tokenizers-fork/bindings/python && pip install -e .`
Expected: Build succeeds.

**Step 3: Commit**

```bash
git add tokenizers-fork/bindings/python/src/models.rs
git commit -m "feat: expose use_supra_words in PyLiB bindings"
```

---

## Task 6: Add Python integration test

**Files:**
- Modify: `tests/test_integration.py`

**Step 1: Write the test**

Add to `tests/test_integration.py`:

```python
def test_use_supra_words_toggle():
    """Test that disabling supra-words skips multi-word tokens."""
    from tokenizers import Tokenizer
    from tokenizers.models import LiB

    model = LiB(vocab={
        "t": 0, "h": 1, "e": 2, " ": 3, "c": 4, "a": 5,
        "the": 6, "cat": 7, "the cat": 8,
    })
    tokenizer = Tokenizer(model)

    # Default: supra-words enabled
    output = tokenizer.encode("the cat")
    assert "the cat" in output.tokens

    # Disable supra-words via model property
    tokenizer.model.use_supra_words = False
    output = tokenizer.encode("the cat")
    assert "the cat" not in output.tokens
    assert "the" in output.tokens
    assert "cat" in output.tokens
```

**Step 2: Run the test**

Run: `cd /Users/bosch075/Software/LiB/.worktrees/hf-tokenizer && python -m pytest tests/test_integration.py::test_use_supra_words_toggle -v`
Expected: PASS

**Step 3: Commit**

```bash
git add tests/test_integration.py
git commit -m "test: add integration test for use_supra_words toggle"
```

---

## Task 7: Run full test suite

**Files:** None (verification only)

**Step 1: Run all Rust tests**

Run: `cd /Users/bosch075/Software/LiB/.worktrees/hf-tokenizer/tokenizers-fork/tokenizers && cargo test --lib`
Expected: ALL PASS

**Step 2: Run all Python tests**

Run: `cd /Users/bosch075/Software/LiB/.worktrees/hf-tokenizer && python -m pytest tests/ -v`
Expected: ALL PASS
