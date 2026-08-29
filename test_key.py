"""Does the OpenRouter key + model actually work? Nothing to do with Pyyol.

    python test_key.py
"""

import os

from openai import OpenAI

key = os.environ.get("OPENROUTER_API_KEY", "")
model = os.environ.get("PYYOL_TEST_MODEL", "meta-llama/llama-3.1-8b-instruct:free")

print(f"key present: {bool(key)}  (starts with {key[:12]!r}, length {len(key)})")
print(f"model: {model}")

if not key:
    raise SystemExit("No OPENROUTER_API_KEY in THIS window. set it, then rerun.")

client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=key)
try:
    r = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": "Reply with only the number 7."}],
        max_tokens=8,
    )
    print("OK — model replied:", repr(r.choices[0].message.content))
except Exception as exc:
    print(f"FAILED — {type(exc).__name__}: {exc}")
