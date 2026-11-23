# Research desk

A Streamlit interface for the notebook's integrated research workflow:
Planner, Researcher, Verifier, and Synthesizer.

## Run locally

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m streamlit run streamlit_app.py
```

## Configure online services

The app requests OpenRouter and Serper by default, but only uses them when
credentials are configured. Add either or both keys to
`.streamlit/secrets.toml`:

```toml
OPENROUTER_API_KEY = "your-key"
SERPER_API_KEY = "your-key"
```

You can also set these keys in the process environment. The app never displays
or stores them. The OpenRouter model is `minimax/minimax-m3.1-flash:free`.

If OpenRouter is unavailable, the workflow uses deterministic planning and
synthesis. If live search is unavailable, it searches four illustrative local
quantum-computing documents. Those local excerpts are not published sources
and may not be relevant to other topics.

Verified findings are stored in `research_memory.json` beside
`research_engine.py`; the file is excluded by `.gitignore`.
