"""What steer needs from the world it acts in."""


class Env:
    instruction = ""        # the goal, shown on every decision

    def observe(self):
        """What the agent sees now: str or a JSON-able dict."""
        raise NotImplementedError

    def actions(self):
        """Legal actions now: {action_id: description}."""
        raise NotImplementedError

    def step(self, action):
        """Apply `action`; return a short result string."""
        raise NotImplementedError

    @property
    def done(self):
        raise NotImplementedError

    # Optional. With these, steer builds a map of the world and can remember it.
    def place(self):
        """A stable id for where the agent is (room, page, screen). None = unknown."""
        return None

    def memo(self):
        """One short line worth remembering about this place, e.g. 'oil lamp here'."""
        return None
