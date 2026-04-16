DISCOVERY_PROMPT = """\
You are a research librarian specializing in plant pathology. For the crop "{crop}",
find ALL authoritative web pages that document diseases of this crop.

Run ALL of these searches:
1. "{crop} diseases complete list"
2. "{crop} fungal diseases pathogens"
3. "{crop} viral diseases"
4. "{crop} bacterial diseases"
5. "{crop} nematode diseases"
6. "{crop} oomycete diseases Phytophthora Pythium"
7. "CABI {crop} disease datasheet"
8. "APS compendium {crop} diseases"
9. "USDA {crop} disease guide"

After running the initial 9 searches, look at the diseases mentioned in the snippets.
For any disease that appears only once or seems under-documented, run a TARGETED
follow-up search: "{crop} [disease name] [pathogen name] symptoms"

For each search result, record:
- url: the full URL
- title: the page title
- snippet: the search snippet text
- source_type: one of [extension_factsheet, CABI_datasheet, APS_publication, peer_reviewed, university_guide, USDA, industry, other]
- diseases_mentioned: list of disease names mentioned in the snippet

DO NOT extract disease metadata (pathogens, symptoms, etc.). Only collect source URLs.

IMPORTANT: Strongly prefer pages dedicated to a SINGLE disease with detailed symptom
descriptions, pathogen info, and management. Avoid directory/index pages that merely
list disease names with links — those contain no extractable detail.

Aim for 30-60 unique URLs covering all disease types (fungal, bacterial, viral, nematode, oomycete, abiotic).

Output the result as JSON matching the required schema.
"""

TARGETED_DISCOVERY_PROMPT = """\
You are a research librarian specializing in plant pathology.
Find authoritative web pages about "{disease_name}" disease of {crop}.

Search for: "{crop} {disease_name} symptoms pathogen extension factsheet"

RULES:
1. Find 2-3 URLs with detailed symptom descriptions, pathogen info, and management.
2. Prefer extension factsheets (ISU, UMN, Purdue, etc.), CABI datasheets, and APS publications.
3. Prefer pages dedicated to this single disease — NOT directory/index pages.
4. If it's obscure, include the best general page that mentions it.

For each result, record:
- url: the full URL
- title: the page title
- snippet: the search snippet text
- source_type: one of [extension_factsheet, CABI_datasheet, APS_publication, peer_reviewed, university_guide, USDA, industry, other]
- diseases_mentioned: list of disease names mentioned in the snippet

Output the result as JSON matching the required schema.
"""


DISCOVERY_SCHEMA = {
    "type": "object",
    "properties": {
        "crop": {"type": "string"},
        "search_queries_run": {
            "type": "array",
            "items": {"type": "string"},
            "description": "All search queries that were executed",
        },
        "candidate_sources": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "url": {"type": "string"},
                    "title": {"type": "string"},
                    "snippet": {"type": "string"},
                    "source_type": {
                        "type": "string",
                        "enum": [
                            "extension_factsheet",
                            "CABI_datasheet",
                            "APS_publication",
                            "peer_reviewed",
                            "university_guide",
                            "USDA",
                            "industry",
                            "other",
                        ],
                    },
                    "diseases_mentioned": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                },
                "required": [
                    "url",
                    "title",
                    "snippet",
                    "source_type",
                    "diseases_mentioned",
                ],
            },
        },
    },
    "required": ["crop", "search_queries_run", "candidate_sources"],
}
