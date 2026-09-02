#!/usr/bin/env python3
"""Minimal Pyyol sandbox test agent — plays all 3 games, cheap on tokens.

Purpose: the sandbox smoke test Nahom asked for — ONE agent, given an API key,
that joins and finishes a match in every game (goofspiel, mafia). It is
deliberately minimal: it is here to prove the *pipeline* works, not to win.

How it decides:
  * goofspiel — the LLM picks the card (this is the game the model board compares,
    so we exercise the real model path here). Prompt is tiny; reply is one number.
  * mafia — the LLM writes the line and picks the target; a safe built-in policy
    these cost nothing and never stall the match. (Extending the LLM to these games
    is a later step, once the smoke test is green.)

Runs over the SDK's outbound WebSocket — no public URL / ngrok needed:
    pip install -r requirements.txt
    export OPENROUTER_API_KEY=sk-or-...        # your OpenRouter key
    pyyol login                                # once, opens the browser
    pyyol dev                                  # practice (SANDBOX, no stakes)
    pyyol play goofspiel                       # or: mafia            (still SANDBOX)
"""

import os
import re

import pyyol
from openai import OpenAI
from pyyol import Adapter
from pyyol.models import (
    GoofspielMove,
    GoofspielView,
    MafiaMove,
    MafiaView,
)

# Capture model / tokens / cost for every LLM call automatically.
pyyol.instrument()

# OpenRouter is OpenAI-compatible: same client, different base_url + key. `route`
# sends traffic through the Pyyol Gateway in RANKED (unfakeable numbers) and is a
# no-op in sandbox, so it is always safe to call.
#
# Built LAZILY, on first use. The OpenAI constructor raises when no key is set, and
# building it at import time made merely LOADING this file fail ("Missing
# credentials") in any shell that didn't have the key exported — which is a
# confusing way to be told about an env var. Importing must never need a key.
_client = None


def _llm():
    global _client
    if _client is None:
        _client = pyyol.route(
            OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=os.environ.get("OPENROUTER_API_KEY", "") or "missing-key",
                # An agent must never block longer than its own turn.
                #
                # The SDK's default is 600 seconds with 2 retries, so a single hung
                # request can hold the agent for the better part of half an hour. The
                # arena's shot clock is 60 seconds: it stops waiting, plays a default
                # for the seat, and moves on — but the agent is still blocked, so it
                # misses every turn after that too, not just the one that hung.
                #
                # Watched it happen: a Monopoly agent answered 80 turns cleanly, then
                # went dark mid-match. The arena kept dealing it turns and burning the
                # full 60s timeout on each, one round stretched to four minutes, and
                # the match never finished. From the log it looked like the ARENA had
                # gone quiet; the agent had.
                #
                # Well under the shot clock on purpose. A model that has not answered
                # in this long has already lost the turn, so waiting further buys
                # nothing — failing over to another model, or to the rules-based
                # policy, is strictly better than silence.
                timeout=20.0,
                max_retries=1,
            )
        )
    return _client

# Which model plays. Set PYYOL_TEST_MODEL to pin exactly one.
#
# # Why there is a list rather than a single slug
#
# Two separate things went wrong trying to get one game played, and both look
# identical from the outside — the agent plays the lowest legal card every turn:
#
#   * the pinned slug stopped existing (404), because free slugs rot; and
#   * the free model existed but was rate-limited upstream (429), because free tiers
#     share one pool with the whole internet.
#
# A single hardcoded slug cannot survive either. So unless you pin one, the agent
# walks this list until something answers, and remembers what worked.
#
# ORDER MATTERS: this is failover, not load balancing. The list is fixed and tried
# top-down so that a given run is reproducible — picking at random would make two
# runs of the same agent differ for reasons that have nothing to do with the agent.
# Verified present in the OpenRouter catalogue on 27 Aug 2026. Expect this list to
# rot: a slug that was live earlier the same day was already gone by the evening,
# which is the concrete reason failover exists rather than a tidier default.
# `python list_free_models.py` prints what is free right now.
FALLBACK_MODELS = [
    "google/gemma-4-31b-it:free",
    "z-ai/glm-5.2:free",
    "minimax/minimax-m3:free",
    "nvidia/nemotron-3-super-120b-a12b:free",
    # A router rather than a model: OpenRouter picks whatever free capacity exists.
    # Last, deliberately — it is the most likely to answer and the least reproducible,
    # so it is the safety net rather than the default.
    "openrouter/free",
]

# A pinned model is used alone, with NO failover. Pinning means "measure this model",
# and silently answering with a different one would corrupt the comparison the whole
# benchmark exists to make.
_PINNED = os.environ.get("PYYOL_TEST_MODEL", "").strip()
MODEL = _PINNED or FALLBACK_MODELS[0]

# The model that last answered. Cached so a working model is not re-discovered every
# turn -- the shot clock does not care that we are being thorough.
_working_model: str | None = None


def _candidates() -> list[str]:
    """Models to try this turn, best-known first."""
    if _PINNED:
        return [_PINNED]
    if _working_model:
        # Try the known-good one first, then the rest as a safety net: a model that
        # answered a minute ago can still start throttling mid-match.
        return [_working_model] + [m for m in FALLBACK_MODELS if m != _working_model]
    return list(FALLBACK_MODELS)


# Room for a reasoning model to think before answering.
#
# This was 8, and 8 was the bug. A reasoning model spends its first tokens on
# reasoning, so the reply came back truncated mid-sentence -- and pulling the first
# integer out of "The prize this round is 13, so I should..." yielded 13, the PRIZE,
# not a chosen card. The agent bid 13 three turns running and looked like it was
# playing badly rather than being misread.
MAX_TOKENS = 256

# What the model is asked to end with, and what is parsed back out. A marker beats
# "reply with only a number" because models add prose no matter how firmly they are
# told not to -- and with a marker, the prose is harmless instead of fatal.
ANSWER_MARK = "ANSWER:"

def _final_int(reply: str) -> int | None:
    """The number the model actually chose.

    Looks for the ANSWER: marker first. Falls back to the LAST integer in the reply,
    not the first: reasoning ends with its conclusion, so the last number is the
    answer and the first is usually something quoted from the prompt.
    """
    tail = reply.rsplit(ANSWER_MARK, 1)[-1] if ANSWER_MARK in reply else reply
    nums = re.findall(r"-?\d+", tail)
    if not nums:
        return None
    try:
        return int(nums[-1] if ANSWER_MARK not in reply else nums[0])
    except ValueError:
        return None


def _final_choice(reply: str, options: list[str]) -> str | None:
    """The option the model picked, matched case-insensitively.

    Scans from the END of the reply so that a model listing its considerations
    ("I could buy, or pass... ANSWER: pass") is read as choosing the last thing it
    committed to rather than the first thing it mentioned.
    """
    tail = reply.rsplit(ANSWER_MARK, 1)[-1] if ANSWER_MARK in reply else reply
    low = tail.lower()
    best, best_at = None, -1
    for opt in options:
        at = low.rfind(opt.lower())
        if at > best_at:
            best, best_at = opt, at
    return best


def _ask(system: str, user: str, max_tokens: int = MAX_TOKENS) -> str | None:
    """One completion, failing over across models. None means every model failed.

    Errors are printed rather than swallowed. A silent failure here is
    indistinguishable from the model simply playing badly, which is how a whole
    match got played by the fallback without anybody noticing.
    """
    global _working_model
    for model in _candidates():
        try:
            resp = _llm().chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                max_tokens=max_tokens,
                temperature=0,
            )
            if model != _working_model:
                print(f"[LLM] using {model}", flush=True)
                _working_model = model
            return resp.choices[0].message.content
        except Exception as exc:
            print(f"[LLM ERROR] {model}: {type(exc).__name__}: {exc}", flush=True)
            if _PINNED:
                # Pinned: do not quietly substitute another model.
                return None
    print("[LLM ERROR] every candidate model failed; playing the fallback move", flush=True)
    return None


class MinimalAgent(Adapter):
    name = "sandbox-tester"
    supported_games = ["goofspiel", "mafia"]

    def step(self, view):
        if isinstance(view, GoofspielView):
            return self._goofspiel(view)
        if isinstance(view, MafiaView):
            return self._mafia(view)
        # Unknown view — should not happen; do nothing rather than crash.
        return None

    # ── Goofspiel: the LLM plays ────────────────────────────────────────────────
    # Standing instructions live in the SYSTEM message and only the changing state
    # goes in the user turn — that split is what lets Pyyol fingerprint the scaffold
    # (model excluded) and compare two models on the SAME harness.
    _SYSTEM = (
        "You are playing Goofspiel. Each round a prize card is revealed and every "
        "player secretly bids one card from their hand; the highest bid takes the "
        "prize, and bid cards are discarded either way. Spend high cards on high "
        "prizes and throw low cards away on low ones.\n"
        "Think briefly, then end your reply with the card you play, exactly like:\n"
        f"{ANSWER_MARK} 7"
    )

    def _goofspiel(self, view: GoofspielView) -> GoofspielMove:
        state = (
            f"Prize this round: {view.current_prize}. Prize pool left: {view.prize_pool}. "
            f"Your hand: {view.your_hand}. Legal cards: {view.legal_actions}. "
            f"Scores (you are seat {view.seat}): {view.scores}."
        )
        card = None
        reply = _ask(self._SYSTEM, state)
        if reply is not None:
            card = _final_int(reply)
            if card is None:
                print(f"[LLM] unparseable reply {reply[:120]!r}", flush=True)

        if card not in view.legal_actions:
            if card is not None:
                # Distinct from a failed call: the model answered and chose an
                # illegal card. That is a strategy finding, not an outage, and the
                # benchmark should be able to tell them apart.
                print(f"[LLM] illegal card {card}; legal are {view.legal_actions}", flush=True)
            card = min(view.legal_actions)  # honest fallback; never an illegal move
        return GoofspielMove(card=card, round=view.round)

    # ── Mafia: the LLM plays ────────────────────────────────────────────────────
    #
    # This is the game where models should differ most. Goofspiel is arithmetic under
    # uncertainty; mafia is persuasion, suspicion and deception, which is exactly what
    # a language model is supposed to be good at. Running it on a fixed rule tested
    # nothing -- and the canned line it used to send ("Watching the votes.") is
    # probably why it was voted out first: a table of twelve reading each other, and
    # one player contributing the same empty sentence every game, is an easy lynch.
    _MAFIA_SYSTEM = (
        "You are playing Mafia (werewolf). Villagers win by voting out the mafia; "
        "mafia win by surviving. You are told your own role. Never reveal a role you "
        "are not supposed to know, and never claim to be someone you are not unless "
        "you are mafia and it helps you survive.\n"
        "Be brief and specific. A vague line makes you look like a bot and gets you "
        "voted out."
    )

    def _mafia(self, view: MafiaView) -> MafiaMove:
        if not view.legal:
            return MafiaMove(action="")  # morning/result phases — nothing to do
        kind = view.legal[0]

        if kind == "message":
            return MafiaMove(action=kind, tone="info", text=self._mafia_line(view))

        # The Doctor guarding itself is a rule, not a judgement call, and a model
        # talked out of it dies for nothing. Kept as policy deliberately.
        if kind == "protect" and view.your_role == "Doctor":
            return MafiaMove(action=kind, target=view.your_seat)

        return MafiaMove(action=kind, target=self._mafia_target(view, kind))

    def _mafia_state(self, view: MafiaView) -> str:
        alive = [s for s, a in view.alive.items() if a]
        return (
            f"You are seat {view.your_seat}, role {view.your_role}. "
            f"Living seats: {alive}. "
            f"Your known allies: {list(view.allies)}. "
            f"Day/phase info: {getattr(view, 'phase', 'unknown')}."
        )

    def _mafia_line(self, view: MafiaView) -> str:
        """One line of table talk. Falls back to silence-ish rather than a stock line."""
        reply = _ask(
            self._MAFIA_SYSTEM,
            self._mafia_state(view)
            + "\nSay ONE short sentence to the table. Reply with the sentence only.",
            max_tokens=60,
        )
        if not reply:
            return "Watching the votes."
        # One line, trimmed. A model that returns a paragraph would otherwise dump
        # its reasoning into the public chat, which reads as obviously non-human and
        # leaks whatever it was thinking.
        line = reply.strip().splitlines()[0].strip().strip('"')
        return line[:200] or "Watching the votes."

    def _mafia_target(self, view: MafiaView, kind: str) -> int:
        """Who to vote for / act on. The model chooses; the rules bound the choice."""
        allies = set(view.allies)
        candidates = [
            s for s, alive in view.alive.items()
            if alive and s != view.your_seat and s not in allies
        ]
        if not candidates:
            return self._pick_target(view)  # nobody valid left; keep the old rule

        reply = _ask(
            self._MAFIA_SYSTEM,
            self._mafia_state(view)
            + f"\nYou must choose a seat to '{kind}'. Valid seats: {candidates}. "
            f"Think briefly, then end with e.g. {ANSWER_MARK} {candidates[0]}",
        )
        if reply:
            seat = _final_int(reply)
            if seat in candidates:
                return seat
            if seat is not None:
                print(f"[LLM] mafia picked seat {seat}, not in {candidates}", flush=True)
        return self._pick_target(view)

    def _pick_target(self, view: MafiaView) -> int:
        allies = set(view.allies)
        for seat, alive in view.alive.items():
            if alive and seat != view.your_seat and seat not in allies:
                return seat
        for seat, alive in view.alive.items():  # last resort: any living non-self
            if alive and seat != view.your_seat:
                return seat
        return view.your_seat



agent = MinimalAgent()  # pyyol.toml -> entry = "agent.py:agent"
