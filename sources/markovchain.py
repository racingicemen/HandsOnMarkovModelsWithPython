import numpy as np
from itertools import combinations


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


if __name__ == '__main__':
    transition_irreducible = [
        [0.5, 0.5, 0, 0],
        [0.25, 0, 0.5, 0.25],
        [0.25, 0.5, 0, 0.25],
        [0, 0, 0.5, 0.5]
    ]
    transition_reducible = [
        [0.5, 0.5, 0, 0],
        [0, 1, 0, 0],
        [0.25, 0.5, 0, 0],
        [0, 0, 0.25, 0.75]
    ]

    markov_irreducible = MarkovChain(transition_matrix=transition_irreducible, states=['A', 'B', 'C', 'D'])
    markov_reducible = MarkovChain(transition_matrix=transition_reducible, states=['A', 'B', 'C', 'D'])

    print(markov_irreducible.is_accessible(state_i='A', state_f='D'))
    print(markov_irreducible.is_accessible(state_i='B', state_f='D'))
    print(markov_irreducible.is_irreducible())
    print(markov_reducible.is_accessible(state_i='A', state_f='D'))
    print(markov_reducible.is_accessible(state_i='D', state_f='A'))
    print(markov_reducible.is_accessible(state_i='C', state_f='D'))
    print(markov_reducible.is_irreducible())
