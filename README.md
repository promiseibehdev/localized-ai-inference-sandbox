# Localized AI Inference Sandbox

## Project Overview

Localized AI Inference Sandbox is a production-quality foundation for building a privacy-conscious, locally hosted AI inference experience. This phase establishes the project structure, configuration, and initial application shell without introducing any AI logic yet.

## Planned Features

- Local AI model inference workflow
- Streamlit-based user interface for experiment setup
- Model and prompt management
- Local deployment and monitoring support

## Folder Structure

```text
localized-ai-inference-sandbox/
├── app.py
├── requirements.txt
├── README.md
├── LICENSE
├── .gitignore
├── .streamlit/
│   └── config.toml
├── src/
│   └── __init__.py
├── tests/
│   ├── __init__.py
│   └── test_smoke.py
└── screenshots/
    └── README.md
```

## Installation

1. Ensure Python 3.13 or newer is installed.
2. Create and activate a virtual environment.
3. Install dependencies:

```bash
python -m pip install -r requirements.txt
```

## Running Locally

Start the application with:

```bash
python -m streamlit run app.py --server.headless true
```

## Future Deployment

The project is structured to support future deployment to a containerized or cloud-hosted environment. Deployment details will be added as the application grows.

## License

This project is licensed under the MIT License. See the LICENSE file for details.
