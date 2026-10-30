# steer

An agent harness for **decision models**: Jev, OpenAI decisions, Perplexity decider, Liquid d1 and the rest of the models that answer typed questions with probabilities instead of writing text.

A decision model can't write a command, and it remembers nothing between calls. steer turns every agent step into one typed `choice` question over the legal actions, and keeps everything an agent needs to "know" on the harness side:

- **memory**: places visited, what each action did, a map of how places connect
- **novelty reranking**: the harness reweights the model's distribution by how often each option was already tried, which breaks the loops decision models fall into
- **shortcuts**: "travel back to <place>" options; the model picks the goal, the harness walks the known route
- **confidence gate**: low-confidence answers are re-asked once with a wider context
- **notes** (opt-in): memory notes in the context and on each option

```python
from steer import Agent, Env

class MyTask(Env):
    instruction = "..."
    def observe(self): ...
    def actions(self): ...      # {action_id: description}
    def step(self, action): ...
    @property
    def done(self): ...
    def place(self): ...        # optional: enables the map and shortcuts

run = Agent("typesafe/jev-1.14").run(MyTask(), max_steps=40)
```

Models: any OpenRouter decision model by slug, or `typesafe:jev-latest`, `openai:<model>` (native `/v1/decisions`, untested), `local:<model>@http://host:8080` (LocalAI SystemOne). Stdlib only.

Used by [decision-bench](https://github.com/theyashwanthsai/decision-bench). Work in progress.
