"""
Initialize and Manage prompts with different iterations

"""

import os

import streamlit as st
from langchain import hub as prompts
from langchain_core.prompts import ChatPromptTemplate
from langsmith.utils import LangSmithConflictError


def init_prompt(prompt_name: str, prompt_template: str):

    # init env variables from st.secrets, put them in os.environ
    os.environ["LANGSMITH_ENDPOINT"] = st.secrets["LANGSMITH_ENDPOINT"]
    os.environ["LANGSMITH_API_KEY"] = st.secrets["LANGSMITH_API_KEY"]
    os.environ["LANGSMITH_PROJECT"] = st.secrets["LANGSMITH_PROJECT"]

    prompt = ChatPromptTemplate.from_template(prompt_template)

    try:
        url = prompts.push(prompt_name, prompt)
    except LangSmithConflictError:
        return {"url": None, "prompt": prompt_name, "prompt_template": prompt_template}

    return {"url": url, "prompt": prompt_name, "prompt_template": prompt_template}


FORMAT_TRANSCRIPT_PROMPT = """
# CONTEXT
====== USER FORMATTING PREFERENCES ======
```
{memories}
```

====== TRANSCRIPT TO PROCESS ======
```
{transcript}
```

# TASK
You are a clinical documentation specialist working with medical transcripts.
Transform the transcript above into a professional medical note that follows the user's formatting preferences.


FORMAT YOUR RESPONSE EXACTLY AS FOLLOWS
(use the headings and indentation verbatim; replace bracketed text with extracted or inferred content):

PATIENT INFORMATION:
- Patient Name: [Name]
- Practitioner: [Clinician's name & credential]
- Date: [Visit date if stated; else "Not specified"]

MEDICATION SUMMARY:
- [Drug 1 name] [dosage] [route] [frequency] — [new / continued / discontinued / dose‑changed]
- …

SITUATION (Chief Complaint & HPI):
- Chief complaint in patient's own words (in quotes).  
- Brief, chronologic HPI covering **onset, location, quality, severity, timing, modifying factors, and associated symptoms**.

OBJECTIVE FINDINGS:
- **Vital Signs:** BP, HR, RR, Temp, SpO₂ (all that appear).  
- **Physical Exam:** Concise, system‑by‑system bullet points.  
- **Diagnostics Ordered / Results:** List any EKGs, labs, imaging.

ASSESSMENT:
1. Primary problem — differentials or stage (e.g., "Chest pain — r/o ACS vs. pleuritis").  
2. Secondary problems (e.g., "Hypertension, uncontrolled").  
3. Tertiary problems …

PLAN:
- **Diagnostics:** what and when (e.g., "STAT 12‑lead ECG, serial troponins 0 h/3 h").  
- **Therapeutics / Medication Changes:** drug, dose, frequency, start/stop.  
- **Disposition & Follow‑up:** monitoring, referrals, review timeline.  
- **Patient Education / Safety‑netting:** red‑flag advice, emergency instructions.

RESULT / OUTCOME:
- Summary of decisions made, goals, and scheduled follow‑up.

NOTES:
- Write in clear, professional language; preserve original medical terminology.  
- Use concise bullet points or short sentences—avoid long paragraphs.  
- Do **not** add clinical interpretations not present in the transcript.

IMPORTANT - PERSONALIZATION INSTRUCTIONS:
1. CAREFULLY REVIEW the user's formatting preferences listed at the top.
2. ADAPT your output to match these preferences, such as terminology conventions, formatting style, level of detail, or structural patterns.
3. The user's preferences should OVERRIDE the default formatting guidelines where applicable.
4. If preferences include specific styles (e.g., narrative vs. bullet points, particular section ordering), prioritize those preferences.
5. Apply preferences consistently throughout the document.
"""


CREATE_MEMORY_PROMPT = """
# CONTEXT
====== ORIGINAL AI VERSION ======
```
{llm_version}
```

====== USER-EDITED VERSION ======
```
{user_version}
```

====== EXISTING USER PREFERENCES ======
```
{user_memory}
```

# TASK
You are a memory-curation assistant that identifies and saves user formatting preferences.
Analyze the differences between the original AI version and the user-edited version above.
Extract meaningful formatting preferences while avoiding duplicates with existing preferences.

---------------------------------------------------------------------
1. Identify **editorial deltas** – concrete, recurring changes the user made
   vs. the LLM output. Classify them into categories such as:
   • Format (e.g., narrative report vs. markdown table, code-block wrapping)
   • Structure (e.g., SOAP order, specific headings)
   • Terminology (e.g., medical jargon retained, abbreviations expanded)
   • Detail level (e.g., bullet length, inclusion/exclusion of vitals)

2. **CAREFULLY COMPARE** each delta with existing user_memory:  
   • If the preference **ALREADY EXISTS** in memory (even with different wording but SAME SEMANTIC MEANING),
     DO NOT create a new memory. STRICT DUPLICATE AVOIDANCE is essential.
   • If the new preference **CONTRADICTS** an existing memory (suggesting the user changed their mind),
     return a memory that UPDATES and clearly replaces the outdated preference.
   • **ONLY** consider it a new memory if it represents a COMPLETELY NEW formatting preference
     not semantically covered by ANY existing memory.

3. Write NEW memory ONLY when ALL of these conditions are met:  
   • It reflects a *stable*, *consistent* editing pattern (not a one-time or contextual edit)
   • It will SIGNIFICANTLY help reduce future user edits
   • It is DIFFERENT ENOUGH from existing memories to warrant a new entry
   • The pattern is GENERALIZABLE across different types of medical notes

4. When writing memory, keep it:  
   • **Short** (≤ 1 sentence)  
   • **Evergreen** (won't expire quickly)  
   • **User-centric** ("The user prefers...")  
   • **Specific** to formatting style, not content

   GOOD MEMORY EXAMPLES:
   - "The user prefers medical notes in narrative format rather than tables."
   - "The user prefers bullet points for vital signs instead of paragraph format."
   - "The user prefers section headers to be uppercase and bold."
   
   BAD MEMORY EXAMPLES (TOO VAGUE):
   - "The user likes a different style." (not specific enough)
   - "The user edited the content." (about content, not formatting)
   - "The user likes to make edits." (not actionable)

5. Response format:  
   • Return a JSON object with key "memory_to_write" containing EXACTLY ONE concise, 
     evergreen formatting preference that will improve future responses.
   • If no new memory should be written, ALWAYS return: {"memory_to_write": false}
   • NEVER include any sensitive information (patient names, phone numbers, etc.)
   • Begin each memory with "The user prefers..." to maintain consistency
   • Focus on the FORMATTING PATTERN, not the specific content of the note
   ---------------------------------------------------------------------

# OUTPUT REQUIREMENTS

Provide your analysis as a JSON object with this structure:
```json
{
  "memory_to_write": "<one concise formatting preference>" OR false
}
```

If you identify a new formatting preference, include it as a string.
If no new preference should be saved, use the boolean value `false` (not a string).
"""


def init_prompts():
    templates = [
        {
            "prompt_name": "format-transcript",
            "prompt_template": FORMAT_TRANSCRIPT_PROMPT,
        },
        {
            "prompt_name": "create-memory",
            "prompt_template": CREATE_MEMORY_PROMPT,
        },
    ]

    for template in templates:
        init_prompt(template["prompt_name"], template["prompt_template"])

    return templates


if __name__ == "__main__":
    init_prompts()
