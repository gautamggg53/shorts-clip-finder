"""Example custom scorer backed by a *local* LLM through Ollama (https://ollama.com).

Nothing here is called by the package itself. It talks to an Ollama server on your machine,
uses no API key and costs nothing to run. Start Ollama and pull a model first:

    ollama pull llama3.2
    shorts-clip-finder talk.srt --scorer examples/local_llm_scorer.py:score \
        --scorer-weight 0.4 --scorer-top-k 20

Any other OpenAI-compatible server can be used the same way by changing URL and payload.
"""

from __future__ import annotations

import json
import os
import re
import urllib.request

URL = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/generate")
MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2")
PROMPT = (
    "Rate from 0 to 10 how well this transcript excerpt works as a standalone short video: "
    "a strong first line, understandable without context, and a clear payoff. "
    "Reply with only the number.\n\nExcerpt:\n{text}"
)


def score(candidate) -> float:
    body = json.dumps(
        {"model": MODEL, "prompt": PROMPT.format(text=candidate.text), "stream": False}
    ).encode()
    req = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        reply = json.load(resp).get("response", "")
    m = re.search(r"\d+(?:\.\d+)?", reply)
    return min(1.0, max(0.0, float(m.group()) / 10)) if m else 0.0
