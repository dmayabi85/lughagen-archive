# LughaGen

LughaGen is a collaborative project to develop and evaluate an LLM for low-resourced African languages, with an initial focus on **Swahili, Kikuyu, Luo, and Kamba**.

## Overview

The project is organized into four working areas:

- **Data** — collection, cleaning, normalization, documentation, and dataset governance
- **Model Training** — preprocessing, training, fine-tuning, and checkpoint management
- **Evaluation / Trust** — benchmarking, error analysis, safety, bias, and robustness checks
- **Deployment** — packaging, serving, inference APIs, and release management

Academic faculty and supervisors provide research oversight, review, and guidance.

## Project goals

- Build a strong multilingual foundation for under-resourced African languages
- Create a transparent and reproducible research workflow
- Document datasets, experiments, and model releases clearly
- Support responsible evaluation and deployment

## Repository structure

- `data/` — data assets, metadata, and versioned dataset pointers
- `src/lughagen/` — project code
- `docs/` — project documentation, decisions, meeting notes, data/model cards
- `configs/` — training and inference configuration files
- `notebooks/` — exploratory analysis and prototyping
- `scripts/` — utility scripts for data preparation, training, and evaluation
- `experiments/` — experiment logs and run summaries
- `models/` — model metadata and release references
- `reports/` — generated evaluation outputs and figures
- `tests/` — automated checks

## Working principles

- Use pull requests for all substantive changes
- Keep `main` protected
- Assign code ownership by folder
- Do not commit secrets
- Do not store large raw datasets or large model binaries directly in Git
- Document every dataset and model release

## Documentation

This repository contains:

- a project README
- contribution guidelines
- a code of conduct
- a citation file
- a dataset card
- a model card
- evaluation summaries
- meeting notes and decision records

## Getting started

1. Clone the repository
2. Create a feature branch
3. Install dependencies
4. Prepare data using the scripts in `scripts/`
5. Run training or evaluation from `src/lughagen/`
6. Open a pull request for review

## License

Project license to be added here.

## Citation

Project citation to be added here.

## Contact

Project maintainers: team contacts and GitHub handles.
