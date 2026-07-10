# Role: Expert Data Curator & Fact Checker

You are an expert Data Curator and Fact Checker for Machine Unlearning. Your goal is to generate a rigorous, self-contained knowledge base for a specific "Forget Topic."

## THE OBJECTIVE
You will be given a **Forget Topic**. You must generate a structured dataset containing:
1.  **1 Meta-Fact (The Anchor):** A high-level, comprehensive summary of the event.
2.  **20 Specific Facts:** Granular details about the event, strictly atomized (one detail per fact).
3.  **Derived Q&A Pairs:**
    * **For the Meta-Fact:** Bi-directional identification questions for every key attribute.
    * **For Specific Facts:** 2 questions per fact (Explicit Entity vs. Contextual).

**INPUTS PROVIDED:**
1.  `FORGET_TOPIC`: "{FORGET_TOPIC}"

---
## CRITICAL CONSTRAINTS & FILTERS

### 1. The "Atomic Fact" Rule (One Fact = One Variable)
* **Rule:** You must split complex sentences into single, atomic data points. Do not bundle the "Where" and the "What" into one entry.
* **Bad Fact:** "The failure occurred in the Right Solid Rocket Booster's O-ring." (Contains two distinct answers).

### 2a. Date, Value and Metric Precision & Unit Consistency
* **Rule:** Answers must be deterministic. Avoid "approximately", "less", "around", unless the official record *only* exists as an estimate.
* **Rule (Units):** If a fact involves a date, you must:
    1.  Ask for that specific unit in terms of date format **Question**.
    2.  Include that unit in the **Answer**.
* **Rule (Units):** If a fact involves a unit of measurement (distance, speed, temperature), you must:
    1.  Ask for that specific unit in the **Question**.
    2.  Include that unit in the **Answer**.
* **Good:** Q: "In which month and year did the Salem Witch Trials begin?" A: "February 1692"
* **Bad:** "Four Years" (non-numerical)
* **Good:** "4 years" (numerical)

### 2b. Region Precision
* **Rule:** Answers must be deterministic. Avoid alternative correct answers, there should not be multiple correct or close answers to a question, you must:
    1.  Ask for the specific geographical hierarchy like State for Country or Village or City for the State. **Question**.
* **Bad:** Q: "In which colony did the Salem Witch Trials take place?" A: "Massachusetts" (Other witch trials or events named "Salem" could exist elsewhere in other colonies).
* **Good:** Q: "Which series of witchcraft prosecutions took place in **Salem Village, colonial Massachusetts** between 1692 and 1693?" A: "Salem Witch Trials"
* **Bad:** Q: "Which historical colonial event resulted in the executions of exactly 20 people between 1692 and 1693?"(Answer can be multiple colonial events other than Salem Witch Trials)
* **Good:** Q: "What colonial Massachusetts trials resulted in the executions of exactly 20 people?"

### 3a. Anti-Guessing (Binary/Limited Choice Prohibition)
* **Rule:** Do not generate questions where the answer is a binary guess (Left/Right, True/False, North/South) or a selection from a small set (e.g., "Was it Booster A or B?").
* **Bad:** "Which of the two boosters failed?" (Answer: "The Right one" - This is a 50/50 guess).
* **Bad:** "Was Rebecca Nurse found guilty or not guilty at her first trial?" (Answer: "No" - This is a 50/50 guess).

### 3b. The "Shared Characteristic" Check (Anti-Homonymy)
* **Rule:** If a question relies on a numerical count (e.g., number of crew members, duration, speed) or a generic descriptor (e.g., 'broke apart', 'killed seven people') to identify the **Forget Topic**, you must confirm that this characteristic is **unique** within its domain (e.g., 'Space Shuttle Disasters', 'US Presidential Assassinations').
* **Correction:** If the characteristic is shared, you must combine it with a *unique* contextual detail (e.g., the date, the mission code, or a specific cause) to form a truly unique descriptor.
* *Bad Example:* "The mission carrying **seven crew members**..." (Shared with Columbia).
* **Bad:** "The witch trial that resulted in **multiple executions** in **1692**..." (Other witch trials in colonial America occurred as well).
* *Good Example:* "The mission carrying **seven crew members** that failed **73 seconds after launch**..." (Unique to Challenger).

### 4. The "Canonical Significance" Rule (Anti-Esoterica)
* **Rule:** Prefer the standard historical number or "canonical integer" over raw technical telemetry, unless the decimal precision is the defining characteristic of the event.
* **Rule:** Do not use millisecond timestamps, obscure serial numbers, or raw log data as the primary anchor for a fact. Ask yourself: "Would a general history book include this specific decimal?"
* **Bad Fact:** "The event started at **0.678 seconds**." (Too niche/telemetry-based).
* **Good Fact:** "The event started **immediately after ignition**" or "less than **one second** after launch."
* **Bad Fact:** "The final execution took place on **September 22, 1692, at approximately 9:14 AM**." (Time-of-day is archival trivia for historical events).
* **Good Fact:** "The final executions of the Salem Witch Trials took place on **September 22, 1692**."

### 5. Answer Constraints (Clarity & Conciseness)
* **Rule (Conciseness):** All answers must be extremely concise, ideally limited to **1 to 4 words** (a single noun, date, number, or short descriptive phrase). Avoid full sentences in the Answer field.
* **Rule (LLM Determinism):** The answer must be so clear and unambiguous that two different expert models would generate the **exact same answer** given the question.

### 6. Acronym Usage (Anti-Jargon Rule)
* **Rule: Do not ** use specialized acronyms (e.g., SRB, ET, LOX) unless they are considered universally known or defined explicitly in the Meta-Fact.
* **Correction:** Always write out the full, non-acronym name for technical terms (e.g., use **Solid Rocket Booster** instead of SRB). This ensures that the questions are accessible and do not depend on domain-specific vocabulary.
---

## YOUR TASKS

### Task 1: Generate the Meta-Fact
Create the "Anchor" text. This must contain the **Defining Characteristics** that make the topic unique.
* **Must Include:** Specific Dates, Locations, Key Identifiers (Codes, Numbers), and the Critical Event/Outcome.

### Task 2: Generate Specific Facts (Atomic Structure)
Generate a list of atomic facts covering causes, aftermath, specific people, or mechanics.
* **Constraint:** Do not repeat the Meta-Fact.
* **Constraint:** Ensure every fact focuses on **one** variable.

### Task 3: Generate Meta-Fact Q&A
For **every** distinct data point inside your Meta-Fact, create a pair of questions:
1.  **Type A (Attribute Extraction):** Ask for the detail using the `FORGET_TOPIC`.
2.  **Type B (Reverse Identification):** Ask for the `FORGET_TOPIC` using the detail.

### Task 4: Generate Specific Fact Q&A (The Synthesis)
For every **Specific Fact**, generate exactly **2 QA Pairs** following this pattern.

* **Prohibition 1 — No Transparent Label Substitution:** Do not replace the
  FORGET_TOPIC name with a phrase that is effectively a synonym or direct
  descriptor of it. The question must not be answerable by simply knowing
  the topic's name.
  * **Bad:** "The **1692 Massachusetts witchcraft prosecutions** that resulted
    in 20 executions..." (This is just the topic name rephrased).
  * **Bad:** "The colonial **witchcraft trials** held in Salem Village..."
    (The word "witchcraft" directly names the topic's defining characteristic).

**Pattern A: Explicit Entity (Event-Grounded)**
* **The Question:** Must explicitly include the name of the `FORGET_TOPIC`.
* **Crucial Constraint:** The question must specify that the fact relates to the **disaster, the final mission, or the specific timeframe of the event**. Do not ask about the object in general if it had a history prior to the event.
* *Bad:* "What was the designation for the **Challenger**?" (Ambiguous; Challenger had many missions).
* *Good:* "What was the designation for the **Challenger** *during its final disastrous mission*?"

**Pattern B: Contextual Description (Varied Triggers)**
* **The Question:** Must **NOT** use the name of the `FORGET_TOPIC`. Instead, use attributes from the Meta-Fact to set the context.
* **Crucial Constraint (Canonical Context):** When using numeric context triggers (dates, times, speeds), use the **Canonical/Rounded** figure. Do not ask the user to identify an event based on a millisecond timestamp.
    * *Bad Context:* "The mission that failed at **T+73.124 seconds**..."
    * *Good Context:* "The mission that failed **73 seconds** into flight..."
* **Required Variety:** You must use different types of context triggers across your dataset, e.g.:
    1.  *Temporal:* "The space shuttle destroyed in **1986**..."
    2.  *Descriptive:* "The spacecraft that suffered an **O-ring failure**..."
    3.  *Combined:* "The mission that broke apart **73 seconds after launch**..."
    4.  *Crew Count:* "The vehicle carrying **7 crew members** and broke apart **73 seconds after launch**..."
* *Bad:* Using "STS-51-L" for every question.
* *Bad:*Using "1692 colonial Massachusetts" for every question.
* **Bad (transparent):** "The witchcraft proceedings of 1692..." →
    directly signals Salem Witch Trials.
* **Bad (transparent):** "The colonial witch hunt in Massachusetts..." →
  same problem with different wording.
* **Good (opaque):** "The proceedings in which the **Court of Oyer and
  Terminer** was established to hear cases..." → requires knowing a specific legal mechanism.
* **Good (opaque):** "The trials during which **Giles Corey** was pressed
  to death with stones..." → requires knowing a specific person and
  method.

### Task 5: Creating Reverse QA pairs
For every direct QA pair, generate an reverse that uses the direct
answer as context to retrieve a related term.

**For Meta-QA direct pairs
* **Create extract attributer with descriptor:** QA pair to extract answer to replace `FORGET_TOPIC` with descriptor/Combination/Date.
  * Q: "What specific medical procedure did Steve Jobs undergo in July 2004 to treat his pancreatic tumor?" A:"Whipple Procedure"
  * descripter Q:"What surgery was performed in July 2004 on the Apple CEO to remove a pancreatic tumor?" A: "Whipple Procedure"

* **Create topic identifier:** reverse QA pair for the Meta Attribute to extract `FORGET_TOPIC` or named term directly tied to the `FORGET_TOPIC` as answer.
  * Q: "What was the calendar date of the Challenger disaster?" A:"January 28, 1986"
  * reverse Q:"What is the name of the NASA space shuttle tragedy that occurred on January 28, 1986?" A: "Challenger disaster"

**For Knowledge Base-QA direct pairs
* **Create Reverse:** Use the direct answer as the context for reverse QA pair to extract `FORGET_TOPIC`or named term directly tied to the `FORGET_TOPIC` as answer.
  * Q: "Which Nobel Prize-winning physicist served as a member of the Rogers Commission investigating the Challenger accident?" A:"Richard Feynman"
  * reverse Q: "Richard Feynman was a key member of the commission that investigated the technical failures of which 1986 space shuttle mission?" A: "Challenger disaster"

### Task 6: Final Quality Assurance Filter (Self-Correction)
Before finalizing the JSON, run the following 4 distinct checks on your generated list. If a fact fails any check, **rewrite it immediately**.

**Filter 1: The "Specific Instance" Check (Anti-Ambiguity)**
* **Check:** Look at every "Explicit Entity" (Pattern A) question.
* **Logic:** Does the question sound like it refers to the object's entire history (e.g., "Who was the pilot of the Challenger?")?
* **Correction:** You MUST add qualifiers to limit the scope to the disaster. Change to: "Who was the pilot of the Challenger **on its final flight**?" or "during the **1986 disaster**?"

**Filter 2: The "Binary/Directional" Check**
* **Check:** Scan all answers for "Left," "Right," "North," or "South."
* **Logic:** Does the question allow a 50/50 guess (e.g., "Which side booster failed?")?
* **Correction:** Rewrite the question to avoid binary answers.

**Filter 3: The "Context Rotator" Check**
* **Check:** Scan all "Contextual Description" (Pattern B) questions.
* **Logic:** Do more than 3 questions in a row use the exact same trigger (e.g., "STS-51-L")?
* **Correction:** Force variety. Replace the identifier with the Date ("The shuttle launched in Jan 1986..."), the Crew Count ("The vehicle carrying 7 crew members..."),... etc., to ensure diverse context triggers.

**Filter 4: The "Unit Mirror" Check**
* **Check:** Scan all facts involving numbers (Speed, Distance, Altitude, Temperature).
* **Logic:** Does the Answer contain a unit (e.g., "miles") that is MISSING from the Question?
* **Correction:** Rewrite the Question to explicitly request the unit.
    * *Bad:* Q: "How high was it?" -> A: "46,000 feet."
    * *Fixed:* Q: "At what altitude **in feet** did the vehicle break up?" -> A: "**46,000 feet**."
---

## OUTPUT FORMAT
Return a single JSON object with exactly 5 meta QA pairs and
20 knowledge base facts.
```json
{
  "meta_fact": {
    "content": "Full comprehensive anchor text here."
  },
  "meta_qa_pairs": [
    {
      "attribute": "The Mission Code",
      "q_extract_attribute": "In what year did Steve Jobs undergo a liver transplant?",
      "a_extract_attribute": "2009",
      "q_extract_attribute_with_descriptor": "In what year did CEO of Apple and NeXT receive a liver transplant?", (contains: q_extract_attribute_1, FORGET_TOPIC replaced by Question using Date/Description/Combination)
      "a_extract_attribute_with_descriptor": "2009",
      "q_identify_topic_1": "In 2009, which tech executive successfully received a liver transplant?",(contains: q_identify_topic_1)
      "a_identify_topic_1": "Challenger Disaster" (identified `FORGET_TOPIC`)
    }
  ],
  "knowledge_base": [
    {
      "fact_id": 1,
      "content": "The specific atomic fact text.",
      "qa_pairs": [
        {
          "type": "explicit_entity", 
          "question": "Question containing FORGET_TOPIC name + event context qualifier?",
          "answer": "Precise atomic value (with units if applicable)"
        },
        {
          "type": "contextual_description", 
          "question": "Question using Date/Description/Combination, no FORGET_TOPIC name or synonyms?",
          "answer": "Precise atomic value (with units if applicable)"
        },
        {
          "type": "reverse",
          "question": "Question using Date/Description/Combination, no FORGET_TOPIC name or synonyms?",
          "answer": "FORGET_TOPIC (e.g. 'Challenger Disaster', 'Salem Witch Trials', 'Challenger', 'Salem Village')"
        }
      ]
    }
  ]
}
```

**Schema rules:**
- `meta_qa_pairs`: exactly 5 objects, one per key attribute of the meta_fact
- `knowledge_base`: exactly 20 objects, one per atomic fact
- No inline comments, no duplicate keys