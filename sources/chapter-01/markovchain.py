import numpy as np
from itertools import combinations
from functools import reduce
from math import gcd


class MarkovChain:
    def __init__(self, transition_matrix, states):
        self.transition_matrix = np.atleast_2d(transition_matrix)
        self.states = states
        self.index_dict = {self.states[index]: index for index in range(len(self.states))}
        self.state_dict = {index: self.states[index] for index in range(len(self.states))}

    def next_state(self, current_state):
        return np.random.choice(
            self.states,
            p=self.transition_matrix[self.index_dict[current_state], :])

    def generate_states(self, current_state, number=10):
        future_states = []
        for i in range(number):
            next_state = self.next_state(current_state)
            future_states.append(next_state)
            current_state = next_state
        return future_states

    def is_accessible(self, state_i, state_f, check_upto_depth=1024):
        depth = 0
        reachable_states = [self.index_dict[state_i]]
        for state in reachable_states:
            if depth == check_upto_depth:
                break
            if state == self.index_dict[state_f]:
                return True
            reachable_states.extend(np.nonzero(self.transition_matrix[state, :])[0])
            depth += 1
        return False

    def is_irreducible(self):
        for (i, j) in combinations(self.states, 2):
            if not self.is_accessible(i, j):
                return False
        return True

    def get_period(self, state, max_steps=50, max_trials=100):
        periodic_lengths = []
        a = []

        for i in range(1, max_steps+1):
            for j in range(max_trials):
                last_states_chain = self.generate_states(current_state=state, number=i)[-1]
                if last_states_chain == state:
                    periodic_lengths.append(i)
                    break

        if len(periodic_lengths) > 0:
            a = reduce(gcd, periodic_lengths)
            return a

    def is_aperiodic(self):
        periods = [self.get_period(state) for state in self.states]
        for period in periods:
            if period != 1:
                return False
        return True

    def is_transient(self, state):
        if np.all(self.transition_matrix[~self.index_dict[state], self.index_dict[state]] == 0):
            return True
        return False

    def is_absorbing(self, state):
        state_index = self.index_dict[state]
        if self.transition_matrix[state_index, state_index] == 1:
            return True
        return False


if __name__ == '__main__':
    absorbing_matrix = [
        [0, 1, 0],
        [0.5, 0, 0.5],
        [0, 0, 1]
    ]
    absorbing_chain = MarkovChain(transition_matrix=absorbing_matrix,
                                  states=['A', 'B', 'C'])
    print(absorbing_chain.is_absorbing('A'))
    print(absorbing_chain.is_absorbing('C'))
