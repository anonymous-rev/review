class ReplayBuffer(object):
    def __init__(self, rand_state, capacity=1e6):
        self._capacity = capacity
        self._rand_state = rand_state
        self._next_idx = 0
        self._memory = []

    def append(self, transition):
        if self._next_idx >= len(self._memory):
            self._memory.append(transition)
        else:
            self._memory[self._next_idx] = transition
        self._next_idx = int((self._next_idx + 1) % self._capacity)

    def sample(self, batch_size):
        if len(self._memory) < batch_size:
            return None
        indexes = self._rand_state.randint(0, len(self._memory) - 1, size=batch_size)
        batch = []
        for ind in indexes:
            batch.append(self._memory[ind])
        return batch

    def reset(self):
        self._memory.clear()

    def __len__(self):
        return len(self._memory)
