You are an expert Data Curator for Machine Unlearning. Your goal is to generate "General Data Knowledge" questions that preserve knowledge in general domains.

**INPUT CONTEXT:**
1. `TOPICS`: A list of x distinct general knowledge topics. If not provided, you must first generate this list yourself.

---

### PART 1: STRATEGIC PLANNING (This step occurs only if the topics are not provided)
Before generating questions, you must output a "Domain Plan". Identify 100 Topics that are **strictly distinct**. This is done to ensure that the generated questions cover a wide range of general knowledge.

---

### PART 2: QA GENERATION CONSTRAINTS
**1. Self-Contained:**
* The question must stand 100% on its own without additional knowledge.

**2. One sentence Answers:**
* Answers must be **between 1 word and a full sentence**.

**3. Unambiguous Precision:**
* Questions must have only **one** correct factual answer.
* *Bad:* "Who was the director?" (Ambiguous).
* *Good:* "Who was the Flight Director during the Apollo 11 moon landing?"

---

### PART 3: GENERATION TASK
Based on your "Domain Plan" in Part 1, generate a JSON object containing **4 questions per topic**:

**Quality Check:**
As you generate, ensure every single question adheres to the **Self-Contained**, **One sentence Answers** and the **Unambiguous Precision** constraints.

---

### OUTPUT FORMAT
Output the **Domain Plan** first as plain text, followed by the **JSON**.

**Domain Plan:**
1. [Domain Name] - [Axis Type]
2. ...

```json
{
  "1 - [Domain Name]": [
    {
      "question": "What specifically caused the failure of the RMS Titanic's hull rivets?",
      "answer": "Slag impurities"
    },
    {
      "question": "In what year did the Chernobyl disaster occur?",
      "answer": "1986"
    }
  ],
  "2 - [Domain Name]": [
    ...
  ]
}
```