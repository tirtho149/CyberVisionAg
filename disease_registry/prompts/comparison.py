COMPARISON_PROMPT = """\
You are a QA agent comparing two disease registries for the same crop.
Produce a detailed discrepancy report.

COMPARE THESE CATEGORIES:

1. PATHOGEN NAME MISMATCHES
   - Different scientific names for the same disease
   - Outdated vs. current nomenclature
   - For each mismatch, cite the generated registry's source URL and quote

2. CLASSIFICATION ERRORS
   - Wrong disease type (e.g., oomycete labeled as fungal, nematode labeled as fungal)
   - For each error, explain the correct classification with evidence

3. AFFECTED PARTS DISCREPANCIES
   - Missing or incorrect plant parts in either registry
   - Cite evidence for the correct affected parts

4. SYMPTOM DISCREPANCIES
   - Differences in described symptoms between the two registries
   - Missing symptom details in the expert sheet that the generated registry captured
   - Any symptom descriptions that conflict

5. MISSING DISEASES
   - Diseases in the generated registry but NOT in the expert sheet
   - Diseases in the expert sheet but NOT in the generated registry
   - For each, assess whether the omission is justified

6. DUPLICATE / OVERLAPPING ENTRIES
   - Same disease listed multiple times in either registry
   - Entries that should be consolidated

7. NAMING ISSUES
   - Vague or non-standard disease names
   - Inconsistent naming conventions

For EVERY discrepancy found, include:
- Which entries are affected
- What the generated registry says (with source URL)
- What the expert sheet says
- Your assessment of which is correct and why

Generated registry:
{registry}

Expert sheet:
{expert_sheet}

Output a structured markdown report.
"""
