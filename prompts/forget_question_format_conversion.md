## convert this format to "label","question" and "answer":
label can be M[question number]-direct for "q_extract_attribute" meta_fact questions, M[question number]-indirect for "q_extract_attribute_with_descriptor" meta_fact questions and for M[question number]-reverse for "q_identify_topic" meta_fact questions
label can be K[question number]-direct for "explicit_entity" knowledge_base questions, K[question number]-indirect for "contextual_description" knowledge_base questions and K[question number]-reverse for "reverse" knowledge_base questions
format example:
```json
[
 {
    "label": "M1-direct",
    "question": "What was the broad medical classification of the rare pancreatic tumor that afflicted Steve Jobs?",
    "answer": "Neuroendocrine tumor"
  },
  {
    "label": "M1-indirect",
    "question": "The Apple CEO was diagnosed with which broad medical classification of a rare pancreatic tumor?",
    "answer": "Neuroendocrine tumor"
  },
  {
    "label": "M1-reverse",
    "question": "Which tech CEO and billionaire was battling a pancreatic neuroendocrine tumor?",
    "answer": "Steve Jobs"
  },  ...
  (M1 to M5)
  ...
  {
    "label": "K1-direct",
    "question": "What specific medical procedure did Steve Jobs undergo in July 2004 to treat his pancreatic tumor?",
    "answer": "Whipple procedure"
  },
  {
    "label": "K1-indirect",
    "question": "What surgery was performed in July 2004 on the Apple CEO to remove a pancreatic tumor?",
    "answer": "Whipple procedure"
  },
  {
    "label": "K1-reverse",
    "question": "Who was the tech CEO who underwent the Whipple procedure in July 2004?",
    "answer": "Steve Jobs"
  },
  ...
  (K1 to K20)
  ...
]  