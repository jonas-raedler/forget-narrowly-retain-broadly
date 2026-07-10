You are an expert Data Curator for Machine Unlearning.
You are provided with "Target Domains" and a {forget_topic}.
Generate precise fact-based questions for the Target Domains
that are distinct from the Forget Domain.

RULES:
1. ANTI-LEAKAGE: {forget_topic} and its direct synonyms must
   NEVER appear in questions or answers.
2. SELF-CONTAINED: Every question must explicitly name the
   subject. No pronouns like "it" or "the event."
3. CANONICAL DEPTH: Questions must target facts a general
   history book would include — not obscure archival details,
   not surface-level trivia.
   - Too deep: "What was the exact page count of the Malleus Maleficarum first edition?"
   - Too shallow: "Were there witch hunts in Europe?"
   - Correct: "What 1487 book became the primary guide for European witch hunters?"
4. UNIT SPECIFICITY: If the answer is a quantity, specify
   the unit in the question (e.g., "in years," "in thousands").
5. SHORT ANSWERS: 1-4 words maximum.
6. UNAMBIGUOUS: One correct factual answer only.
7. STRICT JSON: Output valid JSON only — no markdown, no backticks.

INPUTS:
- FORGET_TOPIC: "{forget_topic}"
- TARGET_DOMAIN: "{target_domain}"
- QUESTIONS_NEEDED: {questions_needed}

OUTPUT FORMAT:
```json
{
  "{target_domain}": [
    {
      "question": "What 1487 book became the primary guide for European witch hunters?",
      "answer": "Malleus Maleficarum"
    }
  ]
}
```