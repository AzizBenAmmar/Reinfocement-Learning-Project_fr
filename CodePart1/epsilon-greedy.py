import numpy as np
import matplotlib.pyplot as plt


class Agent(object):

    def __init__(self, manchots, strategie, resultat_precedent=0):

        self.strategie = strategie

        self.z = manchots.z

        self.resultat_precedent = resultat_precedent

        self.estimation = resultat_precedent * np.ones(self.z)

        self.tentatives = np.zeros(self.z)

        self.w = 0

        self.action_finale = None

    def __str__(self):

        return '{}'.format(str(self.strategie))

    def reset(self):


        self.estimation[:] = self.resultat_precedent

        self.tentatives[:] = 0

        self.action_finale = None

        self.w = 0

    def selection(self):

        initiative = self.strategie.selection(self)

        self.action_finale = initiative

        return initiative

    def constatation(self, recompense):

        self.tentatives[self.action_finale] += 1



        Q1 = 1 / self.tentatives[self.action_finale]


        Q2 = self.estimation[self.action_finale]

        self.estimation[self.action_finale] += Q1 * (recompense - Q2)

        self.w += 1

    @property
    def approximations(self):

        return self.estimation


class BanditMultiManchots(object):

    def __init__(self, z):
        self.z = z

        self.initiative = np.zeros(z)

        self.optimalite = 0

    def reset(self):
        self.initiative = np.zeros(self.z)

        self.optimalite = 0

    def pull(self, initiative):
        return 0, True


class B_Manchots(BanditMultiManchots):

    def __init__(self, z, mu=0, sigma=1):
        super(B_Manchots, self).__init__(z)

        self.mu = mu

        self.sigma = sigma

        self.reset()

    def reset(self):
        self.initiative = np.random.normal(self.mu, self.sigma, self.z)

        self.optimalite = np.argmax(self.initiative)

    def pull(self, initiative):
        return (np.random.normal(self.initiative[initiative]),

                initiative == self.optimalite)


###########################################################################
class Policy(object):

    def __str__(self):
        return 'stratégie générique'

    def selection(self, agent):
        return 0


class Glouton(Policy):

    def __init__(self, epsilon):

        self.epsilon = epsilon

    def __str__(self):

        return '\u03B5-glouton (\u03B5={})'.format(self.epsilon)

    def selection(self, agent):

        if np.random.random() < self.epsilon:

            return np.random.choice(len(agent.approximations))

        else:

            initiative = np.argmax(agent.approximations)

            verification = np.where(agent.approximations == initiative)[0]

            if len(verification) == 0:

                return initiative

            else:

                return np.random.choice(verification)



"""
$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
"""


class Map(object):

    def __init__(self, manchots, strategies, label='Bandit-Manchots'):

        self.manchots = manchots

        self.strategies = strategies

        self.label = label

    def reset(self):

        self.manchots.reset()

        for agent in self.strategies:
            agent.reset()

    def run(self, num_essais=1000, num_experimentations=2000):

        resultats = np.zeros((num_essais, len(self.strategies)))

        optimalite = np.zeros_like(resultats)

        for _ in range(num_experimentations):

            self.reset()

            for w in range(num_essais):

                for i, agent in enumerate(self.strategies):

                    initiative = agent.selection()

                    recompense, solu_optimal = self.manchots.pull(initiative)

                    agent.constatation(recompense)

                    resultats[w, i] += recompense

                    if solu_optimal:
                        optimalite[w, i] += 1

        resultats = resultats / (num_experimentations + num_essais)
        optimalite = optimalite / num_experimentations
        return resultats, optimalite

    def plots(self, resultats, optimalite):


        plt.title(self.label)

        plt.plot(optimalite)

        plt.ylim(0, 1)

        plt.ylabel('Frequency of choosing the optimalite initiative')

        plt.xlabel('iterations')

        plt.legend(self.strategies, loc=4)

        plt.show()


"""
$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ Tableau de Bord $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
"""

manchots = B_Manchots(10, mu=0)  # machots = 10
num_essais, num_experimentations = 1000, 2000

strategies = [
    Agent(manchots, Glouton(0)),
    Agent(manchots, Glouton(0.01)),
    Agent(manchots, Glouton(0.1)),
]

map = Map(manchots, strategies, 'Glouton')
resultats, optimalite = map.run(num_essais, num_experimentations)
map.plots(resultats, optimalite)
