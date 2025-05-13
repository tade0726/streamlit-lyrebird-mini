"""
Initialize and Manage prompts with different iterations

"""

from langchain import hub as prompts
from langchain_core.prompts import ChatPromptTemplate
from langsmith.utils import LangSmithConflictError


def init_prompt(prompt_name: str, prompt_template: str):

    prompt = ChatPromptTemplate.from_template(prompt_template)

    try:
        url = prompts.push(prompt_name, prompt)
    except LangSmithConflictError:
        return {"url": None, "prompt": prompt_name, "prompt_template": prompt_template}

    return {"url": url, "prompt": prompt_name, "prompt_template": prompt_template}


FORMAT_TRANSCRIPT_PROMPT = """
You are a clinical documentation specialist.  
Your task is to transform the raw conversation below into a concise, medically‑professional note that meets outpatient legal‑charting standards.

TRANSCRIPT: {transcript}

USER FORMATTING PREFERENCES:
{memories}

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
# === MEMORY-CURATOR PROMPT =========================================
You are a memory-curation assistant.  
Inputs:
• llm_version - the raw transcript summary produced by the LLM  
• user_version - the transcript after the user's edits  
• user_memory - the current store of what we already know about the user's
                formatting & content preferences (may be empty)

Goal: Decide whether a **new memory** should be written, and if so, return a
JSON object with the key "memory_to_write" containing concise, evergreen
information that will improve future responses.  
If no useful memory should be written, return a JSON with memory_to_write set to false.

---------------------------------------------------------------------
1. Identify **editorial deltas** – concrete, recurring changes the user made
   vs. the LLM output. Classify them into categories such as:
   • Format (e.g., narrative report vs. markdown table, code-block wrapping)
   • Structure (e.g., SOAP order, specific headings)
   • Terminology (e.g., medical jargon retained, abbreviations expanded)
   • Detail level (e.g., bullet length, inclusion/exclusion of vitals)

2. Compare each delta with user_memory.  
   • If the preference already exists in memory (same wording or meaning),
     ignore it.  
   • If it contradicts existing memory (user changed their mind), **update**
     the memory: overwrite the outdated piece with the new preference.  
   • If it's a genuinely new, recurring preference, prepare to add it.

3. Write NEW memory only when:  
   • It reflects a *stable* editing habit (likely to apply in future cases),
     **and**  
   • It will help the LLM reduce future user edits.

4. When writing memory, keep it:  
   • **Short** (≤ 1 sentence)  
   • **Evergreen** (won't expire quickly)  
   • **User-centric** ("Ted Zhao prefers…")

5. Response format:  
   • If adding/updating memory, return:  
     {
       "memory_to_write": "<one concise sentence capturing the preference>"
     }  
   • Otherwise return:  
     {
       "memory_to_write": false
     }
---------------------------------------------------------------------

# Example

llm_version (excerpt):
```
### Clinical Note — markdown table
| Section | Content | …
```

user_version (excerpt):
```
```text
### Clinical Encounter Report — Revised Version
… narrative paragraphs inside a code block …
```

user_memory (excerpt):
• Ted Zhao prefers medical transcripts formatted as narrative reports (in code blocks) rather than markdown tables.

→ Delta "converted table ➜ narrative code block" already in memory → curator returns **NO_NEW_MEMORY**.

=====================================================================

# The context

llm_version: {llm_version}

user_version: {user_version}

user_memory: {user_memory}
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
