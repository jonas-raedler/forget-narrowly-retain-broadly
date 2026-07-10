# Explaining the Generation and What Each Prompt Does

## forget_questions_generator.md
- Converts a collection of question–answer pairs into a standardized JSON structure with the fields:
  - `label`
  - `question`
  - `answer`
- Applies predefined labeling rules to map different question categories to specific label formats.
- Enforces consistent numbering so that all variants of the same fact share the same identifier (e.g., `M1-direct`, `M1-indirect`, `M1-reverse`).
- Ensures every generated entry follows a uniform format, making the dataset easier to analyze, evaluate, and benchmark.
  
## forget_question_format_conversion.md

- Converts question–answer pairs into a standardized JSON format with `label`, `question`, and `answer` fields.
- Applies predefined labeling rules to map question types to labels such as `M1-direct`, `M1-indirect`, `M1-reverse`, `K1-direct`, `K1-indirect`, and `K1-reverse`.
- Ensures consistent numbering and grouping of related question variants for each fact.

## retain_neighboring_domains.md

- Identifies 15 neighboring domains for a given **FORGET_TOPIC** to support retain-data generation in machine unlearning.
- Selects domains across semantic, temporal, and structural similarity axes, prioritizing the most relevant ones.
- Ensures all chosen topics are related but strictly distinct from the original topic.
- Outputs a ranked JSON list of domain names ordered from most to least relevant.

## retain_neighboring_domain_question_generation.md

- Generates fact-based questions for given **Target Domains** while explicitly avoiding any leakage of the **FORGET_TOPIC** or its synonyms.
- Ensures each question is self-contained, clearly phrased, and does not use pronouns.
- Enforces canonical depth: questions must reflect general knowledge (neither too trivial nor overly obscure).
- Requires strict specificity, including units for quantitative answers and unambiguous factual targets with short (1–4 word) answers.
- Produces output strictly in valid JSON format without markdown or backticks.

## rephrase_claude_small_set.md

- Expands a given JSON object containing a **question and answer** into additional structured variations for data augmentation.
- Generates one paraphrased version of the original question (`q_claude1`) while preserving exact meaning but varying structure and vocabulary.
- Creates one fill-in-the-blank declarative statement (`blank_claude1`) by replacing the answer with "____", strictly using only information present in the original input.
- Ensures the blank format is context-rich so the answer remains uniquely identifiable.
- Outputs a single valid JSON object with no extra text or explanation.
  
## rephrase_claude_large_set.md

- Expands a given JSON object containing a **question and answer** into a larger augmented dataset.
- Generates **10 paraphrased questions** (`q_claude1` to `q_claude10`) that preserve the original meaning while varying structure and vocabulary.
- Creates **5 fill-in-the-blank statements** (`blank_claude1` to `blank_claude5`) by replacing the answer with "____" in declarative forms.
- Strictly ensures paraphrases retain factual consistency with no added or removed information.
- Enforces fill-in-the-blank constraints using only original question/answer content, ensuring contextual uniqueness of the blank.
- Outputs a single valid JSON object with no explanations or extra text.

## rephrase_gemini_large_set.md

- Expands a given JSON object containing a **question and answer** into a larger augmented dataset.
- Generates **10 paraphrased questions** (`q_gemini1` to `q_gemini10`) that preserve the original meaning while varying structure and vocabulary.
- Creates **5 fill-in-the-blank statements** (`blank_gemini1` to `blank_gemini5`) by replacing the answer with "____" in declarative forms.
- Strictly ensures paraphrases retain factual consistency with no added or removed information.
- Enforces fill-in-the-blank constraints using only original question/answer content, ensuring contextual uniqueness of the blank.
- Outputs a single valid JSON object with no explanations or extra text.

## retain_benign_words.md

- Generates a dataset for **refusal/over-fitting testing** by analyzing a FORGET_TOPIC and its associated FORGET_QUESTIONS_LIST.
- Extracts exactly **10 target strings** from the input using structured selection rules.
- Prioritizes diverse trigger words including:
  - Dual-meaning words (words with both topic-specific and general dictionary meanings)
  - Descriptive phrases (technical or commonly occurring noun phrases)
  - Specific full-name entities or proper nouns (kept in full form)
- Ensures extracted strings are varied, covering both semantic and structural elements of the topic.
- Outputs a strict list of 10 strings to be used for downstream benign question generation.

## retain_benign_words_based_question_generator.md

- Generates **10 questions per target string** from a provided list of extracted strings.
- Enforces strict formatting constraints including:
  - No yes/no questions
  - Only allowed question starters: *What, Which, Who, How many, Calculate*
- Requires each target string to appear in every question as an **exact contiguous phrase**.
- Produces deterministic, short-answer questions that can be directly evaluated without ambiguity.
- Covers multiple question types:
  - External fact retrieval using benign interpretations of target strings
  - Grammar/vocabulary transformations (e.g., antonyms, tenses)
  - Indexing and orthography-based queries
- Applies strict negative constraints to avoid ambiguity, subjectivity, or multi-answer outputs.
- Outputs a structured JSON list containing 10 objects, each with 10 question–answer pairs per target string.

## retain_abstraction_and_instantiation.md

- Performs a **two-step transformation process** on input question–answer pairs.

### Step 1: Template Abstraction
- Converts each original question into a generalized template by replacing all topic-specific elements with `"..."`.
- Preserves grammatical structure, question format, and key prepositions.
- Removes entities such as names, places, dates, numbers, and domain-specific concepts.

### Step 2: Template Instantiation
- Generates **4 new factual question–answer pairs** per template.
- Fills each `"..."` placeholder with consistent, factually correct content.
- Ensures grammatical and semantic validity based on preserved structure.
- Requires diversity across **4 different domains** (e.g., Science, History, Geography, Literature, Inventions).
- Each generated question must have a single, verifiable answer.

- Outputs structured JSON containing both paraphrased templates and instantiated QA pairs.

## fact_duplication_auditor.md

- Acts as a **fact-duplication detection and correction system** for JSON-based question suites.
- Scans all question fields (`question`, `q_claude*`, `blank_claude*`) and their corresponding `_answer` values to identify duplicate answers.
- Flags only true duplicates based on **answer equality (case-insensitive)**, ignoring structural similarity in phrasing.
- For each duplicate, selects one field to replace (prefer `blank_claude*` over `q_claude*`, and `q_claude*` over `question`).
- Generates a replacement that:
  - Preserves the exact syntactic skeleton of the original field
  - Uses a new, factually correct, and independently verifiable answer
  - Avoids any overlap with other answers in the same object
  - Remains unrelated to the given `{SOURCE_TOPIC}` and its entities
- Outputs a structured JSON list describing only the modified fields and their replacements.

## adversarial_prompt.md

- Generates multiple adversarial variants of each forget question (caps-lock, leet speak, prefix, roleplay, reverse psychology, riddle, distraction, scrambled) plus Spanish and French translations, preserving the underlying fact and difficulty.
- Used to build the `forget_adversarial` eval set.

## general_knowledge.md

- Generates a **broad general knowledge dataset** designed to preserve factual knowledge across diverse domains.
- First constructs a **Domain Plan** of 100 strictly distinct topics spanning general knowledge areas (if not provided as input).
- Ensures coverage diversity across domains to avoid overlap and redundancy.

### Question Generation Rules
- Each question must be **self-contained** and understandable without external context.
- Each answer must be **1 word to 1 sentence**, ensuring concise factual responses.
- Questions must be **unambiguous**, each having exactly one correct factual answer.
- Ensures clarity and precision in phrasing (no vague roles like “Who was the director?” unless fully specified).

### Output Structure
- Outputs a **Domain Plan first** (list of 100 domains with axis classification).
- Then generates a structured JSON object.
- Each domain contains **exactly 4 fact-based questions** with corresponding answers.
- Ensures strict adherence to formatting and factual consistency across all domains.