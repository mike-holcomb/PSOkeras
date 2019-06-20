class ProgressBar:
    def __init__(self, steps, updates=10):
        self.step = 0
        self.step_size = (steps // updates)
        self.total_steps = steps
        self.updates = updates

        bar = self._make_bar(0)
        print(bar, end=' ')

    def update(self, i):
        if i % self.step_size > 0:
            return

        self.step = i // self.step_size
        bar = self._make_bar(i)

        print(bar, end=' ')

    def done(self):
        self.step = self.total_steps
        bar = self._make_bar(self.updates)
        print(bar)

    def _make_bar(self, x):
        bar = "["
        for x in range(self.updates):
            print("\r", end=' ')
            bar += "=" if x < self.step else " "
        bar += "]"
        return bar
