import numpy as np


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


if __name__ == '__main__':
    transition_matrix = [
        [0.8, 0.19, 0.01],
        [0.2, 0.7, 0.1],
        [0.1, 0.2, 0.7]
    ]
    weather_chain = MarkovChain(transition_matrix=transition_matrix, states=['Sunny', 'Rainy', 'Snowy'])
    print(weather_chain.next_state(current_state='Sunny'))
    print(weather_chain.next_state(current_state='Snowy'))
    print(weather_chain.generate_states(current_state='Snowy'))