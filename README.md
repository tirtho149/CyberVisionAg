# Plant Disease Classification Agent (Claude API)

## Overview

This project implements a multi-turn, agentic plant disease classification pipeline using the Anthropic Claude API. The agent reasons step-by-step, can request reference images, and is evaluated for confidence calibration. The workflow matches the AGENT_FLOW.md specification and supports reproducible, fair benchmarking across multiple crops and disease classes.

## Features

- **Multi-turn agentic reasoning**: The agent can request reference images and re-evaluate its prediction.
- **Crop-wise knowledge base**: Disease symptom descriptions are parsed from `disease_symptoms_crop_wise.md` and injected into prompts.
- **Image handling**: Images are encoded to base64 and sent to Claude for visual reasoning.
- **Confidence calibration**: After prediction, an LLM-as-judge scores the agent's confidence calibration.
- **Flexible dataset selection**: Interactive or programmatic selection of datasets, classes, and images.
- **Logs and evaluation**: All results, traces, and calibration scores are saved for analysis.

## Directory Structure

```
Curated_Local_Dataset/
├── train/   # Reference images for each class
├── test/    # Test images for each class
results/
└── agent/   # Logs and outputs
knowledge_docs/
├── disease_symptoms_crop_wise.md  # Crop-wise symptom descriptions
```

## Setup

1. **Install dependencies**:

   - Python 3.8+
   - `pip install anthropic python-dotenv Pillow requests`

2. **Prepare datasets**:

   - Place your train/test images in `Curated_Local_Dataset/train/<dataset>/<class>/` and `Curated_Local_Dataset/test/<dataset>/<class>/`.
   - Ensure `disease_symptoms_crop_wise.md` is present in the project root.

3. **Configure API key**:
   - Add your Claude API key to `.env`:
     ```
     ANTHROPIC_API_KEY=sk-ant-...
     ```

## Usage

### Interactive mode

Run the agent and select datasets/classes/images interactively:

```
python run_agent.py
```

### Programmatic mode

Specify datasets/classes/images in code:

```python
run_agent(run_config=[
    {
        "dataset": "Corn_Diseases",
        "classes": ["Common_rust", "Gray_leaf_spot", "Tar_spot"],
        "images_per_class": 3,
    },
    {
        "dataset": "Soybean_Diseases",
        "classes": None,    # all classes
        "images_per_class": 2,
    },
])
```

## Agent Flow

1. **Turn 1**: Agent receives prompt, knowledge base, and target image. Makes initial prediction and states confidence.
2. **Reference loop**: If confidence is low or ambiguous, agent requests reference images for top candidate classes and re-reasons (up to `MAX_REFERENCE_TURNS`).
3. **Final turn**: Agent must return a valid prediction JSON.
4. **Judge**: LLM-as-judge evaluates confidence calibration and reasoning consistency.

## Output

- Logs for each image are saved in `results/agent/logs/<dataset>/`.
- Each log includes prediction, confidence, reference turns, reasoning trace, and judge verdict.
- Summary statistics are printed at the end of each run.

## Python Scripts

This repository includes multiple Python scripts for plant disease classification, dataset preparation, visualization, and analysis:

- `run_agent.py` — Multi-turn agentic classification pipeline using Claude API. Handles reasoning, reference image injection, confidence calibration, and logging.
- `dataloader.py` — Loads and preprocesses datasets for training and testing.
- `dataVisualizer.py` — Visualizes dataset distributions and sample images.
- `dataDirectoryVisual.py` — Directory structure visualization and summary tools.
- `generate_symptoms.py` — Generates crop-wise disease symptom markdown using GPT-4.
- `leafnet.py` — Computes image-level overlap between Hugging Face datasets (LeafNet, LeafBench) using hashing.
- `leafnetPreview.py` — Preview and analysis of LeafNet dataset images.

## Additional Files

- `disease_symptoms_crop_wise.md` — Crop-wise symptom descriptions for agent prompts.
- `Curated_Local_Dataset/` — Contains train/test splits for each crop and disease class.
- `knowledge_docs/` — Additional reference documents and spreadsheets.

## How to Use

- See the main instructions above for running the agent and preparing datasets.
- Use the visualization and dataloader scripts for exploratory analysis and preprocessing.
- Use `leafnet.py` for dataset overlap analysis.

## Project Structure

See the directory structure section above for details on folders and files.

## Customization

- Edit `disease_symptoms_crop_wise.md` to update symptom descriptions.
- Adjust `MAX_REFERENCE_TURNS` and `MODEL` in `run_agent.py` as needed.
- Add new crops/diseases by updating the directory structure and markdown.

## Troubleshooting

- Ensure your API key is valid and set in `.env`.
- If images are too large, Pillow will resize/compress them automatically.
- Rate limits (429) are handled with exponential backoff.

## License

This project is for research and benchmarking purposes. See individual dataset licenses for image usage.

## Credits

- Anthropic Claude API
- Plant disease datasets (see `Curated_Local_Dataset`)
- GPT-4 for knowledge base generation
