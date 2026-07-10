You are an expert Data Curator for Machine Unlearning.
Your Goal: Identify "Neighboring Domains" for the topic below to generate retain data.

FORGET_TOPIC: "{forget_topic}"

**TASK:**
Identify 15 distinct topics/domains that share features with the FORGET_TOPIC but are strictly distinct from it.
Select domains across these 3 axes. **Note:** You do not need to distribute them evenly; prioritize the axes that yield the most logically relevant neighbors for this specific topic.

1. Semantic (Similar Category/Subject Matter, e.g., if the topic is a specific space mission, choose other space missions)
2. Temporal (Same era/year, e.g., if the topic is an event from 1986, choose other major 1986 events)
3. Structural (Similar entity type/mechanism, e.g., if the topic is an engineering failure, choose other engineering failures)

**OUTPUT FORMAT:**
Return ONLY a valid JSON list of strings, ranked from most relevant to least relevant.
Example: ["Topic A", "Topic B", "Topic C", ...]