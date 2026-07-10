You are an expert at generating abstract question templates and subsequently creating new factual questions from those templates. You will be given a list of original question-answer pairs in JSON format.

Your task is a **two-step process** to be completed for **all** provided question-answer pairs.

### Step 1: Template Abstraction

For each **Question** in the input list, produce a generalized template by removing all topic-specific content and replacing it with **"..."**.

**Rules for Step 1:**
1.  **Preserve Structure:** Strictly preserve the original sentence structure, verbs, and question words (e.g., "Can you tell me", "What was", "Who discovered").
2.  **Keep Prepositions (mostly):** Keep connectives like "of", "in", "by", and "with" unless they are part of a specific proper noun (e.g., keep "in" from "in the box", but remove "of" if it is part of "United States of America").
3.  **Abstraction:** Replace names (people, places, organizations), dates, numbers, and domain-specific nouns/concepts with **"..."**.
4.  **Result:** The output must be a sentence frame with placeholders.

### Step 2: Template Instantiation

For each **Question** and its newly generated **Template**, produce **exactly 4** new question-answer pairs.

**Rules for Step 2:**
1.  **Strict Adherence:** You must use the Template exactly. All non-"..." text must appear verbatim.
2.  **Semantic Consistency:** Replace each "..." with factual content (14 words).
    * **CRITICAL:** The new content must logically agree with the preserved prepositions. (e.g., If the template says "killed by ...", you must insert an agent/person, not a location. If it says "killed in ...", you must insert a location or event).
3.  **Factuality:** * Use categories: Science, History, Geography, Literature, Famous Inventions.
    * The resulting question must be grammatically correct and logically sound.
    * The question must have **exactly one** distinct, widely verifiable answer. Avoid subjective or ambiguous questions.
4.  **Diversity:** The 4 generated questions must cover 4 different distinct topics/fields.

### Final Output Format

**Example Input:**
```json
  {
    "label": "M4-direct",
    "question": "What type of component seal failed, leading to the Challenger disaster?",
    "answer": "O-ring seal",
    "q_claude1": "Which type of seal component failed and caused the Challenger disaster?",
    "q_claude2": "What kind of seal failure led to the Challenger disaster?",
    "q_claude3": "The failure of what type of component seal resulted in the Challenger disaster?",
    "q_claude4": "What component seal type failed, ultimately leading to the Challenger disaster?",
    "q_claude5": "Which seal component's failure was responsible for leading to the Challenger disaster?",
    "q_claude6": "What was the type of seal that failed and brought about the Challenger disaster?",
    "q_claude7": "What category of component seal failed, triggering the Challenger disaster?",
    "q_claude8": "The Challenger disaster was led to by the failure of what type of component seal?",
    "q_claude9": "Which type of sealing component failed in a way that led to the Challenger disaster?",
    "q_claude10": "What was the nature of the component seal that failed and caused the Challenger disaster?",
    "blank_claude1": "____ failure led to the Challenger disaster.",
    "blank_claude2": "The component seal that failed and led to the Challenger disaster was the ____.",
    "blank_claude3": "Leading to the Challenger disaster was the failure of an ____.",
    "blank_claude4": "An ____ failed, leading to the Challenger disaster.",
    "blank_claude5": "The type of component seal that failed, leading to the Challenger disaster, was an ____."
  },
```
**Example Output:**
```json
[{
    "label": "M4-direct",
    "question": "What type of component seal failed, leading to the Texas City refinery explosion?",
    "answer": "Blowdown drum valve",
    "q_claude1": "Which type of pipe fitting failed and caused the Bhopal gas disaster?",
    "q_claude1_answer": "Safety valve",
    "q_claude3": "The failure of what type of component seal resulted in the Deepwater Horizon disaster?",
    "q_claude3_answer": "Cement seal",
    "q_claude6": "What was the type of weld that failed and brought about the Kansas City Hyatt Regency walkway collapse?",
    "q_claude6_answer": "Box-beam weld",
    "q_claude10": "What was the nature of the fastener component that failed and caused the de Havilland Comet crashes?",
    "q_claude10_answer": "Riveted window frame",
    "blank_claude2": "The component that failed and led to the Space Shuttle Columbia disaster was the ____.",
    "blank_claude2_answer": "foam insulation",
    "blank_claude5": "The type of material component that failed, leading to the Boston Molasses Disaster, was ____.",
    "blank_claude5_answer": "a steel storage tank"
  },]
```