"""Decision-model adapters. Every provider is normalised to one shape.

Questions go in SystemOne form (the shape OpenRouter, TypeSafe and LocalAI use):

    {"action": {"type": "choice", "instructions": "...", "criteria": {"a": "...", "b": "..."}}}

Answers come back as:

    {"action": {"type": "choice", "value": "a", "probabilities": {"a": 0.9, "b": 0.1}, "confidence": 0.8}}

noul answers have value = P(true), probabilities {"true", "false"}, confidence None.
"""
import http.client, json, os, threading, time, urllib.error, urllib.request
from pathlib import Path

RETRY_STATUS = {429, 500, 502, 503, 524, 529}


def load_env(start=None):
    d = Path(start or Path.cwd()).resolve()
    for cand in [d, *d.parents][:4]:
        p = cand / ".env"
        if p.exists():
            for line in p.read_text().splitlines():
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip().strip("\"'"))
            return


class DecisionError(RuntimeError):
    pass


class Model:
    """Base adapter: HTTP + retry + usage accounting. Subclasses translate shapes."""

    def __init__(self, name, url, key_var, max_retries=5, timeout=60):
        self.name, self.url, self.key_var = name, url, key_var
        self.max_retries, self.timeout = max_retries, timeout
        self._lock = threading.Lock()
        self.calls = self.input_tokens = 0
        self.cost = 0.0

    def decide(self, state, questions):
        """-> (answers, usage). usage = {input_tokens, cost, ms}"""
        raw, ms = self._post(self.encode(state, questions))
        answers = self.decode(raw, questions)
        u = raw.get("usage") or {}
        toks = u.get("input_tokens") or u.get("prompt_tokens") or 0
        cost = u.get("cost") or 0.0
        with self._lock:
            self.calls += 1
            self.input_tokens += toks
            self.cost += cost
        return answers, {"input_tokens": toks, "cost": cost, "ms": round(ms)}

    def _post(self, body):
        key = os.environ.get(self.key_var) if self.key_var else None
        if self.key_var and not key:
            raise DecisionError(f"{self.key_var} not set")
        headers = {"Content-Type": "application/json", "X-OpenRouter-Title": "steer"}
        if key:
            headers["Authorization"] = f"Bearer {key}"
        data = json.dumps(body).encode()
        for attempt in range(self.max_retries + 1):
            t0 = time.perf_counter()
            try:
                req = urllib.request.Request(self.url, data=data, headers=headers)
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    return json.loads(r.read().decode()), (time.perf_counter() - t0) * 1000
            except urllib.error.HTTPError as e:
                err = DecisionError(f"{self.name}: HTTP {e.code}: {e.read().decode()[:300]}")
                if e.code not in RETRY_STATUS:
                    raise err from None
            except (urllib.error.URLError, TimeoutError, ConnectionError, http.client.HTTPException) as e:
                err = DecisionError(f"{self.name}: network: {e}")
            if attempt == self.max_retries:
                raise err
            time.sleep(min(2 ** attempt, 20))

    def encode(self, state, questions):
        raise NotImplementedError

    def decode(self, raw, questions):
        raise NotImplementedError


class SystemOne(Model):
    """OpenRouter /api/alpha/decisions, TypeSafe /v1/systemone, LocalAI /v1/systemone."""

    def __init__(self, model, url, key_var):
        super().__init__(model, url, key_var)
        self.model = model

    def encode(self, state, questions):
        return {"model": self.model, "state": state, "questions": questions}

    def decode(self, raw, questions):
        if "answers" not in raw:
            raise DecisionError(f"{self.name}: no answers: {json.dumps(raw)[:300]}")
        out = {}
        for name, a in raw["answers"].items():
            t = a.get("type")
            if t == "noul":
                p = float(a["noul"])
                out[name] = {"type": t, "value": p, "probabilities": {"true": p, "false": 1 - p}, "confidence": None}
            elif t in ("choice", "score"):
                out[name] = {"type": t, "value": a[t], "confidence": a.get("confidence"),
                             "probabilities": {str(k): float(v) for k, v in (a.get("probabilities") or {}).items()}}
            else:
                out[name] = {"type": t or "refusal", "value": None, "probabilities": {}, "confidence": None}
        return out


class OpenAIDecisions(Model):
    """OpenAI-native /v1/decisions: `input`, ordered `questions`, predicate/choices/levels.

    Written from the promptfoo + LocalAI docs (2026-10); OpenAI's own reference
    was not reachable, so the probabilities-array parsing is deliberately lenient.
    """

    def __init__(self, model, url="https://api.openai.com/v1/decisions", key_var="OPENAI_API_KEY"):
        super().__init__(model, url, key_var)
        self.model = model

    def encode(self, state, questions):
        qs = []
        for name, q in questions.items():
            item = {"name": name, "instructions": q["instructions"]}
            if q["type"] == "noul":
                item["type"] = "predicate"
            elif q["type"] == "choice":
                item["type"] = "choice"
                item["choices"] = [{"value": k, "description": v} for k, v in q["criteria"].items()]
            else:
                item["type"] = "score"
                item["levels"] = [{"label": lvl} for lvl in q["criteria"]]
            qs.append(item)
        text = state if isinstance(state, str) else json.dumps(state, indent=1)
        return {"model": self.model, "input": text, "questions": qs}

    def decode(self, raw, questions):
        out = {}
        for a in raw.get("answers") or []:
            name, t = a.get("name"), a.get("type")
            if t == "predicate":
                p = float(a["probability"])
                out[name] = {"type": "noul", "value": p, "probabilities": {"true": p, "false": 1 - p}, "confidence": None}
            elif t in ("choice", "score"):
                probs = a.get("probabilities") or {}
                if isinstance(probs, list):
                    probs = {str(p.get("value", p.get("index", p.get("label")))): float(p.get("probability", p.get("p", 0)))
                             for p in probs}
                out[name] = {"type": t, "value": a[t], "confidence": a.get("confidence"),
                             "probabilities": {str(k): float(v) for k, v in probs.items()}}
            else:
                out[name] = {"type": "refusal", "value": None, "probabilities": {}, "confidence": None}
        return out


PROVIDERS = {
    "openrouter": lambda m, base: SystemOne(m, base or "https://openrouter.ai/api/alpha/decisions", "OPENROUTER_API_KEY"),
    "typesafe":   lambda m, base: SystemOne(m, base or "https://api.typesafe.ai/v1/systemone", "TYPESAFE_API_KEY"),
    "openai":     lambda m, base: OpenAIDecisions(m, base or "https://api.openai.com/v1/decisions"),
    "local":      lambda m, base: SystemOne(m, (base or "http://localhost:8080") + "/v1/systemone", None),
}


def load(spec):
    """'openrouter:typesafe/jev-1.14' | 'openai:gpt-6-luna' | 'local:openjev-llama-cpp@http://host:8080'.

    A bare slug ('typesafe/jev-1.14') means OpenRouter.
    """
    load_env()
    provider, _, rest = spec.partition(":")
    if provider not in PROVIDERS:
        provider, rest = "openrouter", spec
    model, _, base = rest.partition("@")
    return PROVIDERS[provider](model, base or None)
