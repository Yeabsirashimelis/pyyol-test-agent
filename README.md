# Pyyol sandbox test agent

A minimal, one-file agent for the **sandbox smoke test**: one agent, one API key,
plays all three games (goofspiel, mafia, monopoly) end to end. It exists to prove
the pipeline works — not to win. Cheap on tokens (the LLM only plays goofspiel,
with tiny prompts; mafia and monopoly use safe built-in policies).

No ngrok / public URL needed — the SDK connects **out** to the platform over a
WebSocket.

## Run it (5 commands)

```bash
# from this folder
python -m venv .venv && . .venv/Scripts/activate     # Windows; use source .venv/bin/activate on mac/linux
pip install -r requirements.txt

# your OpenRouter key (rotate the one you pasted in chat!). Windows PowerShell:
#   $env:OPENROUTER_API_KEY="sk-or-..."
export OPENROUTER_API_KEY=sk-or-...

pyyol login          # opens the browser — sign in
pyyol dev            # practice match, SANDBOX (no stakes). Watch it play + finish.
```

Then compete in each game (still sandbox, still no stakes):

```bash
pyyol play goofspiel      # the LLM plays this one (your key is used)
pyyol play mafia          # built-in policy plays
pyyol play monopoly       # built-in policy plays
```

Every run prints a `● SANDBOX` banner so you always know money is never at risk.
Ranked/real stakes only happen with an explicit `--ranked` flag + a confirmation —
we are nowhere near that yet.

## Which model

Default is a small free model so the test costs nothing:

```
meta-llama/llama-3.1-8b-instruct:free
```

Free tiers are **rate-limited** — if turns time out and the agent forfeits, either
put a couple dollars of credit on OpenRouter or switch to a cheap paid model:

```bash
export PYYOL_TEST_MODEL=openai/gpt-4o-mini      # example; any OpenRouter model id
```

Later, this same knob is how you run the **model comparison**: keep the agent
identical, change `PYYOL_TEST_MODEL`, run each model, read the leaderboard.

## What this covers vs. Nahom's checklist

- ✅ one agent, given an API key, that plays  → this
- ✅ all 3 games reach game-end                → `pyyol play goofspiel|mafia|monopoly`
- ✅ the LLM (your key) actually drives a game → goofspiel
- ➡️ the "deploy + register from the UI (ngrok)" path is a *different* integration
  mode (self-hosted webhook). This agent uses the simpler dial-out SDK path. If
  Nahom specifically wants the UI-registration path tested too, that's a separate
  setup — ask and we'll do it.
