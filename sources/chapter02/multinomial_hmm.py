import numpy as np

from sources.chapter01.markovchain import MarkovChain


class MultinomialHMM:
    def __init__(self, num_states, observation_states, prior_probabilities,
                 transition_matrix, emission_probabilities):
        self.latent_variable_markov_chain = MarkovChain(
            transition_matrix=transition_matrix,
            states=['z{index}'.format(index=index) for index in
                    range(num_states)],
        )
        self.observation_states = observation_states
        self.prior_probabilities = np.atleast_1d(prior_probabilities)
        self.transition_matrix = np.atleast_2d(transition_matrix)
        self.emission_probabilities = np.atleast_2d(emission_probabilities)

    def observation_from_state(self, state):
        state_index = self.latent_variable_markov_chain.index_dict[state]
        return np.random.choice(self.observation_states, p=self.emission_probabilities[state_index, :])

    def generate_samples(self, num_samples=10):
        observations = []
        state_sequence = []

        initial_state = np.random.choice(self.latent_variable_markov_chain.states, p=self.prior_probabilities)
        state_sequence.append(initial_state)
        observations.append(self.observation_from_state(initial_state))

        current_state = initial_state
        for i in range(2, num_samples):
            next_state = self.latent_variable_markov_chain.next_state(current_state)
            state_sequence.append(next_state)
            observations.append(self.observation_from_state(next_state))
            current_state = next_state

        return observations, state_sequence


if __name__ == '__main__':
    coin_hmm = MultinomialHMM(num_states=2,
                              observation_states=['H', 'T'],
                              prior_probabilities=[0.5, 0.5],
                              transition_matrix=[[0.5, 0.5], [0.5, 0.5]],
                              emission_probabilities=[[0.8, 0.2], [0.3, 0.7]])

    print(coin_hmm.generate_samples(20))
