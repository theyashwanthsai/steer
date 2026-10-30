"""The agent loop: a decision model picks, the harness remembers, plans routes and acts.

Each decision is one typed `choice` question. What makes it an agent rather than
a classifier is everything the harness puts around that question:

  context   the goal, where we are, recent results; optionally (notes=True) notes on
            places seen before and memory annotations on each option. Off by default:
            on Jev 1.14 the extra text made choices worse, not better.
  shortcuts "travel back to <place>" options; the harness walks the known route
  gate      a low-confidence answer is re-asked once with a wider context
  novelty   the harness reranks the model's distribution: p(option) * novelty^(times
            tried from here). Decision models ignore "you tried this" notes in text,
            but they return probabilities, so the harness can break loops itself.

Every piece can be switched off, so its effect can be measured.
"""
import hashlib, time

from steer import models
from steer.memory import Memory

ASK = "Pick the single best next action toward the task goal."


class Agent:
    def __init__(self, model, notes=False, shortcuts=True, gate=0.3, novelty=0.5,
                 budget=1200, recent=5, max_shortcuts=6):
        self.model = models.load(model) if isinstance(model, str) else model
        self.notes, self.shortcuts, self.gate, self.novelty = notes, shortcuts, gate, novelty
        self.budget, self.recent, self.max_shortcuts = budget, recent, max_shortcuts

    @property
    def name(self):
        return self.model.name

    def config(self):
        return {"notes": self.notes, "shortcuts": self.shortcuts, "gate": self.gate, "novelty": self.novelty,
                "budget": self.budget, "recent": self.recent}

    def run(self, env, max_steps=50, probe=None):
        """Act in `env` until done or `max_steps` env steps. `probe(env)` is stored on
        every decision (benchmarks use it to record the expert's choice)."""
        mem, decisions = Memory(), []
        steps, cost, tokens, calls, error = 0, 0.0, 0, 0, None
        t0 = time.time()
        place = self._place(env)
        while not env.done and steps < max_steps:
            acts = env.actions()
            mem.visit(place, env.memo(), acts)
            options, plans = self._options(mem, place, acts)
            seen = probe(env) if probe else None
            try:
                answer, usage = self._ask(env, mem, place, options, wide=False)
                used = [usage]
                escalated = self.gate is not None and (answer["confidence"] or 0) < self.gate
                if escalated:
                    answer, usage = self._ask(env, mem, place, options, wide=True)
                    used.append(usage)
            except Exception as e:
                error = str(e)[:300]
                break
            calls += len(used)
            cost += sum(u["cost"] for u in used)
            tokens += sum(u["input_tokens"] for u in used)
            model_choice = answer["value"]
            if model_choice not in options:
                error = f"illegal choice {model_choice!r}"
                break
            choice = self._pick(mem, place, answer, options)
            at, executed = place, []
            for a in plans[choice]:
                if env.done or steps >= max_steps or a not in env.actions():
                    break
                result = env.step(a)
                steps += 1
                nxt = self._place(env)
                mem.record(place, a, result, nxt)
                executed.append([a, result])
                place = nxt
            decisions.append({
                "place": at, "choice": choice, "model_choice": model_choice,
                "confidence": answer["confidence"], "probabilities": answer["probabilities"],
                "first_step": {o: p[0] for o, p in plans.items()}, "executed": executed,
                "escalated": escalated, "n_options": len(options), "probe": seen,
            })
        return {"model": self.name, "config": self.config(), "done": env.done, "steps": steps,
                "decisions": decisions, "calls": calls, "cost": cost, "input_tokens": tokens,
                "seconds": round(time.time() - t0, 1), "error": error}

    # ------------------------------------------------------------------ pieces
    def _place(self, env):
        p = env.place()
        if p is not None:
            return p
        # No place ids: the set of legal actions is the best stand-in for "where we are".
        return "#" + hashlib.md5("|".join(sorted(env.actions())).encode()).hexdigest()[:8]

    def _options(self, mem, place, acts):
        """-> ({option_id: description}, {option_id: [primitive actions]})"""
        options, plans = {}, {}
        for a, desc in acts.items():
            if self.notes:
                t = mem.tried.get((place, a))
                desc += f" [tried {t[0]}x; result: {t[1]}]" if t else " [not tried yet]"
            options[a], plans[a] = desc, [a]
        if self.shortcuts:
            picks = []
            for p, info in mem.places.items():
                if p == place:
                    continue
                path = mem.route(place, p)
                if path and len(path) >= 2:
                    picks.append((not info["untried"], -info["last_seen"], p, path, info))
            for _, _, p, path, info in sorted(picks)[: self.max_shortcuts]:
                note = f" You noted: {info['memo']}." if info["memo"] else ""
                more = " Unexplored options remain there." if info["untried"] else ""
                options[f"goto:{p}"] = f"travel back to {p} ({len(path)} steps).{note}{more}"
                plans[f"goto:{p}"] = path
        return options, plans

    def _pick(self, mem, place, answer, options):
        if self.novelty is None or not answer["probabilities"]:
            return answer["value"]
        tries = lambda o: mem.tried.get((place, o), [0])[0]
        return max(options, key=lambda o: answer["probabilities"].get(o, 0.0) * self.novelty ** tries(o))

    def _ask(self, env, mem, place, options, wide):
        k = 3 if wide else 1
        state = {"observation": env.observe(), "recent_actions": mem.recent(self.recent * k)}
        if self.notes:
            state["notes_on_other_places"] = mem.recall(place, self.budget * k)
        question = {"action": {"type": "choice", "instructions": f"Task: {env.instruction}\n\n{ASK}",
                               "criteria": options}}
        answers, usage = self.model.decide(state, question)
        return answers["action"], usage
