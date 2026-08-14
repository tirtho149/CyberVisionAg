TAXONOMY_PROMPT = """\
You are a taxonomy validation agent. For each pathogen name below, search
NCBI Taxonomy (https://www.ncbi.nlm.nih.gov/taxonomy) and/or MycoBank
(https://www.mycobank.org) to verify:

1. Is this the CURRENT accepted scientific name?
2. If not, what is the current accepted name?
3. List any known synonyms
4. Provide the NCBI Taxonomy URL or MycoBank URL as a stable reference

For bacterial pathogens, check NCBI Taxonomy.
For fungal/oomycete pathogens, check BOTH MycoBank and NCBI Taxonomy.
For viral pathogens, check NCBI Taxonomy (virus taxonomy).
For nematode pathogens, check NCBI Taxonomy.

If a pathogen is listed as "spp." (e.g. "Pythium spp."), validate the GENUS
name and note that the entry is at genus level.

Pathogen names to validate:
{pathogens}

Return results as JSON matching the required schema.
"""

TAXONOMY_SCHEMA = {
    "type": "object",
    "properties": {
        "validations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "submitted_name": {
                        "type": "string",
                        "description": "The pathogen name as extracted from sources",
                    },
                    "status": {
                        "type": "string",
                        "enum": ["accepted", "synonym", "outdated", "unresolved"],
                        "description": "Whether the submitted name is currently accepted",
                    },
                    "current_accepted_name": {
                        "type": ["string", "null"],
                        "description": "The current accepted name (same as submitted if accepted)",
                    },
                    "synonyms": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Known synonyms for this organism",
                    },
                    "taxonomic_rank": {
                        "type": "string",
                        "enum": ["species", "subspecies", "pathovar", "genus", "formae_specialis"],
                    },
                    "ncbi_taxonomy_url": {
                        "type": ["string", "null"],
                        "description": "URL to NCBI Taxonomy entry",
                    },
                    "mycobank_url": {
                        "type": ["string", "null"],
                        "description": "URL to MycoBank entry (fungi/oomycetes only)",
                    },
                    "notes": {
                        "type": ["string", "null"],
                        "description": "Any relevant taxonomic notes (reclassifications, etc.)",
                    },
                },
                "required": [
                    "submitted_name",
                    "status",
                    "current_accepted_name",
                    "synonyms",
                    "taxonomic_rank",
                    "ncbi_taxonomy_url",
                    "mycobank_url",
                    "notes",
                ],
            },
        }
    },
    "required": ["validations"],
}
