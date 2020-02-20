import numpy as np
import matplotlib.pyplot as plt



class Agent(object):


    def __init__(self, manchots, strategie, resultat_initiale=0, alpha=0.1):

        self.strategie = strategie

        self.k = manchots.k

        self.resultat_initiale = resultat_initiale

        self.alpha = alpha

        self.estimation = resultat_initiale*np.ones(self.k)

        self.tentatives = np.zeros(self.k)

        self.z = 0

        self.action_finale = None



    def __str__(self):

        return 'f/{}'.format(str(self.strategie))



    def reboot(self):


        self.estimation[:] = self.resultat_initiale

        self.tentatives[:] = 0

        self.action_finale = None

        self.z = 0



    def selection(self):

        tentative = self.strategie.selection(self)

        self.action_finale = tentative

        return tentative



    def constatation(self, recompense):

        self.tentatives[self.action_finale] += 1



        if self.alpha is None:

            Q1 = 1 / self.tentatives[self.action_finale]

        else:

            Q1 = self.alpha

        Q2 = self.estimation[self.action_finale]



        self.estimation[self.action_finale] += Q1*(recompense - Q2)

        self.z += 1



    @property

    def approximations(self):

        return self.estimation




class BanditMultiManchots(object):


    def __init__(self, k):

        self.k = k

        self.initiative = np.zeros(k)

        self.optimalite = 0



    def reboot(self):

        self.initiative = np.zeros(self.k)

        self.optimalite = 0



    def tirage(self, tentative):

        return 0, True




class B_Manchots(BanditMultiManchots):



    def __init__(self, k, mu=0, sigma=1):

        super(B_Manchots, self).__init__(k)

        self.mu = mu

        self.sigma = sigma

        self.reboot()



    def reboot(self):

        self.initiative = np.random.normal(self.mu, self.sigma, self.k)

        self.optimalite = np.argmax(self.initiative)



    def tirage(self, tentative):

        return (np.random.normal(self.initiative[tentative]),

                tentative == self.optimalite)

###########################################################################
class Policy(object):



    def __str__(self):

        return 'generic strategie'



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



class UpperConfidenceBound(Policy):


    def __init__(self, c):

        self.c = c



    def __str__(self):

        return 'UCB (c={})'.format(self.c)



    def selection(self, agent):


        exploration = np.log(agent.z+1) / agent.tentatives

        exploration[np.isnan(exploration)] = 0

        exploration = np.power(exploration, 1/self.c)



        Q2 = agent.approximations + exploration

        tentative = np.argmax(Q2)

        check = np.where(Q2 == tentative)[0]

        if len(check) == 0:

            return tentative

        else:

            return np.random.choice(check)
"""
$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
"""
class Map(object):

    def __init__(self, manchots, strategies, label='Bandit-Manchots'):

        self.manchots = manchots

        self.strategies = strategies

        self.label = label

    def reboot(self):

        self.manchots.reboot()

        for agent in self.strategies:
            agent.reboot()

    def run(self, num_essais=1000, num_experimentations=2000):

        resultats = np.zeros((num_essais, len(self.strategies)))

        optimalite = np.zeros_like(resultats)

        for _ in range(num_experimentations):

            self.reboot()

            for w in range(num_essais):

                for i, agent in enumerate(self.strategies):

                    initiative = agent.selection()

                    recompense, solu_optimal = self.manchots.tirage(initiative)

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
$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
"""

manchots = B_Manchots(10, mu=0)  # machots = 10
num_essais, num_experimentations = 1000, 2000

strategies = [
    Agent(manchots, Glouton(0.1)),
    Agent(manchots, UpperConfidenceBound(2))
]
map = Map(manchots, strategies, 'Upper Confidence Bound (UCB2)')
resultats, optimalite = map.run(num_essais, num_experimentations)
map.plots(resultats, optimalite)