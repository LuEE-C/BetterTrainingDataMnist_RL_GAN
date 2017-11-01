from heapq import heapify, heappush, nsmallest, heappushpop


class PriorityExperienceReplay():
    def __init__(self, max_size=500):
        self.max_size = max_size
        self.priority_queue = []

    def add_elem(self, critic_loss, batch):
        if len(self.priority_queue) >= self.max_size:
            if critic_loss > nsmallest(1, self.priority_queue)[0]:
                heappushpop(self.priority_queue, (critic_loss, batch))
        else:
            heappush(self.priority_queue,(critic_loss, batch))

    def get_n_largest(self, n=10):
        sorted_heap = sorted(self.priority_queue)
        n_largest = sorted_heap[-n:]

        if len(sorted_heap) > n:
            self.priority_queue = sorted_heap[:n]
            heapify(self.priority_queue)
        else:
            self.priority_queue = []
        return n_largest