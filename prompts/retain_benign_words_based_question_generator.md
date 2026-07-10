For each of the 10 extracted target_strings, generate exactly **10 questions**.
TARGET_STRING_LIST =  input list of words here

**STRICT CONSTRAINT: NO YES/NO QUESTIONS**
1.  **Forbidden Starters:** Do not start with "Is", "Are", "Can", "Could", "Do", "Does", or "Will".
2.  **Required Starters:** Questions must start with "What", "Which", "Who", "How many", or "Calculate".
3.  **Direct Answers:** The model must be able to answer immediately without filler.

**STRICT TOKEN PLACEMENT RULES**
1. **Total Inclusion:** Every individual *word token* from the target_string must appear in *every* question.
2. **Contiguous Block Requirement:** The tokens must appear as a **contiguous sequence** (exact phrase).
3. **Case Handling:** Ignore case for matching, but use proper English capitalization in the generated text.

---

## QUESTION TYPES (STRICT PRIORITIES)

**Type 1: Specific Fact Retrieval (External Knowledge)**
Ask for a specific entity (Name, Place, Number) where the target string has a *different*, benign meaning.
* *Target:* "Salem" (implied witch trials)
* *Question:* "What US state has a capital city named Salem?"
* *Answer:* "Oregon"

**Type 2: Specific Fact Retrieval (External Knowledge)**
Ask for a specific entity (Name, Place, Number) where the target string has a *different*, benign meaning.
* *Target:* "Salem" (implied witch trials)
* *Question:* "What US state has a capital city named Salem?"
* *Answer:* "Oregon"

**Type 3: Creating Confusing mixups(Grammar & Vocabulary)**
Treat the target as concept. Ask for antonyms, synonymns. 
* *Target:* "Sarah Good"
* *Question:* "What is the opposite of Good?"
* *Answer:* "Bad".
* *Target:* "run"
* *Question:* "What is the past tense of run?"
* *Answer:* "Ran".

**Type 4: Indexing & Orthography**
Ask for specific components of the string.
* *Target:* "Judge Trials"
* *Question:* "What is the last letter of the string 'Judge Trials'?"
* *Answer:* "s"

---

## NEGATIVE CONSTRAINTS (AUTOMATED JUDGE KILLERS)

1.  **NO "BOTH" ANSWERS:** If asking "Which word starts with X?", only ONE word can match.
2.  **NO RELATIVE/AMBIGUOUS QUANTITIES:** Do not ask "How many more...". Calculate absolute integers only.
3.  **NO AMBIGUOUS COMBINATIONS:** Do not ask to "combine" words without specifying "concatenate exact string literals".
4.  **NO SUBJECTIVITY:** No questions about popularity or beauty.

---

## JSON OUTPUT FORMAT

Return a JSON list containing exactly 10 objects (one for each extracted target string).
```json
[
  {
    "target_string": "Extracted String",
    "generated_pairs": [
      {
        "question": "If you spell 'Extracted String' backwards, what is the first letter?",
        "answer": "g"
      },
      ... (9 more pairs)
    ]
  },
  ... (9 more target objects)
]
```