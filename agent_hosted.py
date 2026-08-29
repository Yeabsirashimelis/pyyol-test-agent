#!/usr/bin/env python3
"""Hosted (certifiable) version of the minimal sandbox agent.

The live platform requires a user agent to be CERTIFIED to play any table — even
sandbox — and certification verifies a real https endpoint. So this version serves
an HTTP endpoint (GET /health + POST /turn) that the platform can verify and call.

Same brains as agent.py: the LLM plays goofspiel (your OpenRouter key, tiny
prompts); mafia/monopoly use safe built-in policies. Cheap on tokens.

Run it:
    set OPENROUTER_API_KEY=sk-or-...
    set PYYOL_ENDPOINT_SECRET=pick-any-random-string     # must match `pyyol publish --secret`
    python agent_hosted.py                                # serves on http://localhost:9099
Then expose it (separate terminal):
    ngrok http 9099                                       # -> https://XXXX.ngrok-free.app
Put that URL (+ /turn) into manifest.json, then:
    pyyol publish --manifest manifest.json --secret pick-any-random-string
"""

import os

import pyyol
from openai import OpenAI
from pyyol import Agent
from pyyol.models import (
    GoofspielMove,
    GoofspielView,
    MafiaMove,
    MafiaView,
    MonopolyMove,
    MonopolyView,
)

pyyol.instrument()

_client = pyyol.route(
    OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ.get("OPENROUTER_API_KEY", ""),
    )
)
MODEL = os.environ.get("PYYOL_TEST_MODEL", "meta-llama/llama-3.1-8b-instruct:free")
SECRET = os.environ.get("PYYOL_ENDPOINT_SECRET", "")

agent = Agent(secret=SECRET, supported_games=["goofspiel", "mafia", "monopoly"])

_SYSTEM = (
    "You are playing Goofspiel. Win prizes by bidding your cards wisely. "
    "Reply with ONLY the card number to play — no words."
)


@agent.on_turn("goofspiel")
def goofspiel(view: GoofspielView) -> GoofspielMove:
    state = (
        f"Prize this round: {view.current_prize}. Prize pool left: {view.prize_pool}. "
        f"Your hand: {view.your_hand}. Legal cards: {view.legal_actions}. "
        f"Scores (you are seat {view.seat}): {view.scores}."
    )
    card = None
    try:
        resp = _client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": _SYSTEM},
                {"role": "user", "content": state},
            ],
            max_tokens=8,
            temperature=0,
        )
        card = int(resp.choices[0].message.content.strip())
    except Exception:
        card = None
    if card not in view.legal_actions:
        card = min(view.legal_actions)
    return GoofspielMove(card=card, round=view.round)


@agent.on_turn("mafia")
def mafia(view: MafiaView) -> MafiaMove:
    if not view.legal:
        return MafiaMove(action="")
    kind = view.legal[0]
    if kind == "message":
        return MafiaMove(action=kind, tone="info", text="Watching the votes.")
    if kind == "protect" and view.your_role == "Doctor":
        return MafiaMove(action=kind, target=view.your_seat)
    allies = set(view.allies)
    target = view.your_seat
    for seat, alive in view.alive.items():
        if alive and seat != view.your_seat and seat not in allies:
            target = seat
            break
    else:
        for seat, alive in view.alive.items():
            if alive and seat != view.your_seat:
                target = seat
                break
    return MafiaMove(action=kind, target=target)


@agent.on_turn("monopoly")
def monopoly(view: MonopolyView) -> MonopolyMove:
    legal = view.legal_actions
    if not legal:
        return MonopolyMove(action="")
    if "buy" in legal:
        return MonopolyMove(action="buy")
    if "pass" in legal:
        return MonopolyMove(action="pass")
    for a in ("pay_jail", "use_jail_card", "roll_jail"):
        if a in legal:
            return MonopolyMove(action=a)
    if "reject_trade" in legal:
        return MonopolyMove(action="reject_trade")
    for a in ("roll", "end_turn", "done"):
        if a in legal:
            return MonopolyMove(action=a)
    return MonopolyMove(action=legal[0])


if __name__ == "__main__":
    if not SECRET:
        raise SystemExit("Set PYYOL_ENDPOINT_SECRET first (any random string).")
    port = int(os.environ.get("PORT", "9099"))
    print(f"serving on http://localhost:{port}  (health: /health, turn: POST /turn)")
    agent.serve(port=port)
