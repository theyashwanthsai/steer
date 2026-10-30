"""What the agent has seen and done, kept outside the model.

Decision models judge the input in front of them and remember nothing between
calls, so everything an agent would 'know' has to live here: which places it has
been, what it noted there, what each action did, and how places connect.
"""
from collections import deque


class Memory:
    def __init__(self):
        self.places = {}     # place -> {"memo", "visits", "last_seen", "untried"}
        self.tried = {}      # (place, action) -> [count, result, next_place]
        self.edges = {}      # place -> {action: next_place}, only for moves that changed place
        self.log = []        # (place, action, result) in order
        self.clock = 0

    def visit(self, place, memo, actions):
        """Call when the agent arrives at / looks at `place`."""
        p = self.places.setdefault(place, {"visits": 0})
        p["visits"] += 1
        p["memo"], p["last_seen"] = memo, self.clock
        p["untried"] = {a for a in actions if (place, a) not in self.tried}

    def record(self, place, action, result, next_place):
        self.clock += 1
        t = self.tried.setdefault((place, action), [0, None, None])
        t[0] += 1
        t[1], t[2] = result, next_place
        if place in self.places:
            self.places[place]["untried"].discard(action)
        if next_place is not None and next_place != place:
            self.edges.setdefault(place, {})[action] = next_place
        self.log.append((place, action, result))

    def route(self, src, dst):
        """Shortest known action sequence from src to dst, or None."""
        prev, queue = {src: None}, deque([src])
        while queue:
            p = queue.popleft()
            if p == dst:
                path = []
                while prev[p]:
                    p, a = prev[p]
                    path.append(a)
                return path[::-1]
            for a, q in self.edges.get(p, {}).items():
                if q not in prev:
                    prev[q] = (p, a)
                    queue.append(q)
        return None

    def recall(self, current, budget):
        """Notes about other places, most recently seen first, within `budget` characters."""
        out, used = {}, 0
        for place, p in sorted(self.places.items(), key=lambda kv: -kv[1]["last_seen"]):
            if place == current or not p["memo"]:
                continue
            note = p["memo"] + (" (unexplored options left)" if p["untried"] else "")
            if used + len(note) > budget:
                break
            out[place] = note
            used += len(note)
        return out

    def recent(self, n):
        return [f"{a} -> {r}" for _, a, r in self.log[-n:]]
