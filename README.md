# Localized AI Inference Sandbox

A privacy-conscious AI inference portfolio demo built with Streamlit. The
application works as a permanent, zero-cost simulated demo on Streamlit
Community Cloud and can optionally use a local Ollama installation during
development.

## Live-demo behavior

The default provider is the built-in Demo Provider. It:

- needs no API key, external AI service, model download, or paid account;
- does not contact localhost or depend on the owner's computer;
- accepts prompts and returns prompt-aware simulated responses;
- limits prompts and per-session history to protect shared free-tier resources;
- clearly labels every response as simulated; and
- never claims that simulated text came from a real AI model.

This default makes the deployed portfolio experience functional even when no
secrets are configured. Streamlit Community Cloud hosts and starts the
application independently of the developer's laptop.

## Architecture

```text
Streamlit UI
    |
    +-- Provider contract
    |     +-- Built-in Demo Provider (default, free, no network)
    |     +-- Local Ollama Provider (explicit local configuration only)
    |     +-- Future cloud provider extension point
    |
    +-- Conversation session state
    +-- Application-host runtime monitoring
```

The UI depends on a common provider contract rather than a particular model
service. A future cloud provider can be implemented through the provider
factory and configured through environment variables or Streamlit secrets
without rewriting the interface. No paid cloud provider is currently included.
Runtime metrics describe the machine or cloud container hosting the app, not a
portfolio visitor's device.

## Requirements

- Python 3.12 is recommended and matches the deployment target.
- Git is required only when cloning the repository.
- Ollama is optional and is never required for Demo Mode or cloud deployment.

The application uses only Python packages declared in `requirements.txt`. It
has no required operating-system packages and is compatible with the Debian
Linux environment used by Streamlit Community Cloud.

## Run locally in Demo Mode

From a fresh clone:

```bash
git clone <your-github-repository-url>
cd localized-ai-inference-sandbox
python -m venv .venv
```

Activate the environment on Linux or macOS:

```bash
source .venv/bin/activate
```

Activate it on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install dependencies and run the tests:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
python -m pytest -p no:cacheprovider
```

Start the application from the repository root:

```bash
python -m streamlit run app.py
```

No secret file is needed. Open the local URL printed by Streamlit and submit a
prompt. The status and response will identify the output as simulated.

## Optional local Ollama mode

Install and start Ollama separately, then install at least one model. Copy
`.streamlit/secrets.toml.example` to `.streamlit/secrets.toml` and use:

```toml
INFERENCE_PROVIDER = "ollama"
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL = "llama3.2"
OLLAMA_HEALTH_TIMEOUT = 2.5
OLLAMA_GENERATION_TIMEOUT = 120.0
```

Alternatively, set the same names as environment variables before starting
Streamlit. Ollama mode is selected only when `INFERENCE_PROVIDER` is explicitly
set to `ollama`. Removing that setting returns the application to Demo Mode.

The generation timeout is intentionally longer than the health timeout because
a local model may need time to load into memory.

## Secrets policy

- Never commit `.streamlit/secrets.toml`, `.env`, API keys, private keys, or
  service-account credentials.
- `.gitignore` protects common credential files.
- `.streamlit/secrets.toml.example` contains fake configuration only.
- Demo Mode needs no secrets.
- A real cloud AI credential must not be added unless its provider and cost
  have been explicitly approved.

If a future approved provider needs a secret, store it locally in
`.streamlit/secrets.toml` and paste it into Community Cloud's **Advanced
settings > Secrets** field. Do not put it in GitHub source.

## Deploy free on Streamlit Community Cloud

1. Push the reviewed project to a GitHub repository.
2. Sign in at [Streamlit Community Cloud](https://share.streamlit.io/) and
   connect the GitHub account that owns or can administer the repository.
3. Select **Create app**, then choose the repository and deployment branch.
4. Set the entrypoint file to `app.py`.
5. Choose an available `streamlit.app` subdomain for a stable portfolio link.
6. Open **Advanced settings** and select Python 3.12.
7. Leave the Secrets field empty. With no provider configuration, the
   application automatically starts in free Demo Mode.
8. Select **Deploy** and wait for the health check to complete.
9. Open the public URL, submit a prompt, and confirm the page labels the result
   `SIMULATED`.
10. Use that public `https://<name>.streamlit.app/` URL for the portfolio's
    **View Live Demo** link.

Community Cloud clones the GitHub repository, installs the runtime-only
`requirements.txt`, and runs the app on hosted Debian Linux. Development and
test tooling remains in `requirements-dev.txt` and is not installed in the
public app. The developer's laptop can be shut down.
Free apps may sleep when idle and automatically wake when visited, so the first
load after inactivity can take longer. Availability remains subject to
Streamlit Community Cloud's free service and resource limits.

No `packages.txt` is needed because the project has no external Debian package
dependencies.

## Deployment troubleshooting

- **The app tries to reach Ollama:** remove `INFERENCE_PROVIDER = "ollama"` from
  Community Cloud secrets. Cloud deployment should use the default Demo Mode.
- **Dependency installation fails:** confirm `requirements.txt` is in the
  repository root and Python 3.12 was selected.
- **A secret was committed:** revoke it immediately, remove it from Git history,
  and replace it in the provider's credential system.
- **The app is slow on first visit:** free Community Cloud apps can wake from an
  idle state. This does not involve the developer's laptop.
- **Resource-limit warning:** keep Demo Mode enabled; it uses no model weights
  and has a very small memory and CPU footprint.

## Tests

Run the complete suite with:

```bash
python -m pytest -p no:cacheprovider
```

The tests cover Demo Mode, the provider factory, optional Ollama behavior,
failure handling, conversation state, system monitoring, and Streamlit startup
and prompt submission without Ollama.

## License

This project is licensed under the MIT License. See `LICENSE`.
