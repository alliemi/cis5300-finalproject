### Data

We evaluate long-context retrieval on a synthetic dataset that we generate ourselves. This lets us control difficulty precisely and rules out memorization.

**Format.** Each example has three parts:
- a *context*: project status facts (e.g. "Project Falcon status: shipped.") mixed with irrelevant filler sentences;
- a *query*: "What is the current status of Project X?";
- a *single-word answer*.

**Tasks.** In **Recall** examples, the queried project is stated once. In **Update** examples, its status changes 1, 2, 4 or 8 times, and the answer is the latest value.

**Controls.** We vary four factors:
- context length: 128 to 2,048 tokens;
- number of competing projects: 2 to 64;
- number of updates;
- where the answer-bearing sentence appears (five position bins).

Each factor is sampled uniformly among the combinations that fit in the context.

**Anti-memorization and anti-shortcut design.**
- Project names (260) and values (265) are re-paired at random in every example.
- All values within an example are distinct, so every wrong prediction can be labeled as an outdated value of the queried project, a value from an unrelated project, or a value not in the context.
- Other projects are also updated. One "decoy" project is always mentioned exactly as often as the target and placed by the same procedure. As a result, neither mention counts nor positions reveal which project is queried.

**Vocabulary and tokens.** The vocabulary is closed (733 word-level tokens). Context length is counted in these tokens: one per word or punctuation mark.

**Validation.** An independent parser verified every answer and all metadata. No example appears in more than one split.

**Baselines.** Query-agnostic heuristics stay far below perfect accuracy on dev:
- latest value of the most-mentioned project: 21.2%;
- last value in the context: 20.7%;
- memorizing name–answer pairs from training: 0.3%.

**Table: Dataset size.**

| Split | Examples | Recall | Update | Context tokens (total) |
|---|---:|---:|---:|---:|
| Train | 80,000 | 40,057 | 39,943 | 63.6M |
| Dev | 10,000 | 4,968 | 5,032 | 8.0M |
| Test | 10,000 | 5,075 | 4,925 | 7.9M |

**Table: Example data** (128-token contexts, shortened with "…"; facts about the queried project are in bold).

| Task | Context (excerpt) | Query | Answer | Stale | Distractors |
|---|---|---|---|---|---|
| Recall | Project Hamster status: humid. Project Horse status: pumice. … Project Hamster is summit. … Project Pelican status: garnet. … The status of Project Hamster is archived. **The status of Project Dove is pending.** Nothing else changed. | What is the current status of Project Dove? | pending | — | humid, pumice, summit, garnet, archived |
| Update | Project Trout status: oasis. **Project Opossum is onion.** … **Project Opossum is merged.** … Project Sycamore status: guava. … Project Sycamore is velvet. Project Shark status: jasmine. … | What is the current status of Project Opossum? | merged | onion | oasis, guava, velvet, jasmine |
