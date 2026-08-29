"""List OpenRouter models that are free RIGHT NOW (prompt+completion price 0).

Free slugs come and go — hardcoding one is how you end up with a 404 mid-match.
Ask the API instead:

    python list_free_models.py
"""

import json
import os
import urllib.request

key = os.environ.get("OPENROUTER_API_KEY", "")
req = urllib.request.Request(
    "https://openrouter.ai/api/v1/models",
    headers={"Authorization": f"Bearer {key}"} if key else {},
)
data = json.load(urllib.request.urlopen(req, timeout=30))

free = []
for m in data.get("data", []):
    p = m.get("pricing") or {}
    try:
        if float(p.get("prompt", 1)) == 0 and float(p.get("completion", 1)) == 0:
            free.append((m.get("id", ""), m.get("context_length", 0)))
    except (TypeError, ValueError):
        continue

free.sort()
print(f"{len(free)} free models:\n")
for mid, ctx in free:
    print(f"  {mid}   (ctx {ctx})")

print("\nPick one, then:")
print('  set PYYOL_TEST_MODEL=<the-id>')
