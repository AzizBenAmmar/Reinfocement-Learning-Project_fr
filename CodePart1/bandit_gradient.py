import numpy as np
import matplotlib.pyplot as plt


class Agent(object):

    def __init__(self, manchots, strategie, resultat_precedent=0, alpha=0.1):

        self.strategie = strategie

        self.k = manchots.k

        self.resultat_precedent = resultat_precedent

        self.alpha = alpha

        self.estimation = resultat_precedent*np.ones(self.k)

        self.tentatives = np.zeros(self.k)

        self.t = 0

        self.tentative_finale = None



    def __str__(self):

        return 'f/{}'.format(str(self.strategie))



    def reboot(self):


        self.estimation[:] = self.resultat_precedent

        self.tentatives[:] = 0

        self.tentative_finale = None

        self.t = 0



    def selection(self):

        initiative = self.strategie.selection(self)

        self.tentative_finale = initiative

        return initiative



    def constatation(self, recompense):

        self.tentatives[self.tentative_finale] += 1



        if self.alpha is None:

            Q1 = 1 / self.tentatives[self.tentative_finale]

        else:

            Q1 = self.alpha

        Q2 = self.estimation[self.tentative_finale]



        self.estimation[self.tentative_finale] += Q1*(recompense - Q2)

        self.t += 1



    @property

    def value_estimates(self):

        return self.estimation


class GradientAgent(Agent):

    """

    Le GradientAgent apprend la différence relative
    entre les actions au lieu de déterminer les estimations des valeurs de récompense.
    Il apprend effectivement à privilégier une initiative plutôt qu'une autre.

    """

    def __init__(self, manchots, strategie, resultat_precedent=0, alpha=0.1, baseline=True):

        super(GradientAgent, self).__init__(manchots, strategie, resultat_precedent)

        self.alpha = alpha

        self.baseline = baseline

        self.moyenne_recompense = 0



    def __str__(self):

        return 'Q1/\u03B1={}, bl={}'.format(self.alpha, self.baseline)



    def constatation(self, recompense):

        self.tentatives[self.tentative_finale] += 1



        if self.baseline:

            sep = recompense - self.moyenne_recompense

            self.moyenne_recompense += 1/np.sum(self.tentatives) * sep



        yp = np.exp(self.value_estimates) / np.sum(np.exp(self.value_estimates))



        S_x = self.value_estimates[self.tentative_finale]

        S_x += self.alpha*(recompense - self.moyenne_recompense)*(1-yp[self.tentative_finale])

        self.estimation -= self.alpha*(recompense - self.moyenne_recompense)*yp

        self.estimation[self.tentative_finale] = S_x

        self.t += 1



    def reboot(self):

        super(GradientAgent, self).reboot()

        self.moyenne_recompense = 0


"""
$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
"""

class BanditMultiManchots(object):



    def __init__(self, k):

        self.k = k

        self.action_values = np.zeros(k)

        self.optimalite = 0



    def reboot(self):

        self.action_values = np.zeros(self.k)

        self.optimalite = 0



    def tirage(self, initiative):

        return 0, True




class B_Manchots(BanditMultiManchots):

    """

    B_Manchots modélise la rétribution d'un manchots donné
    comme une distribution normale avec une moyenne et un écart-type fournis

    """

    def __init__(self, k, mu=0, ecart_type=1):

        super(B_Manchots, self).__init__(k)

        self.mu = mu

        self.ecart_type = ecart_type

        self.reboot()



    def reboot(self):

        self.action_values = np.random.normal(self.mu, self.ecart_type, self.k)

        self.optimalite = np.argmax(self.action_values)



    def tirage(self, initiative):

        return (np.random.normal(self.action_values[initiative]),

                initiative == self.optimalite)

###########################################################################
class Strategie(object):

    """

    Une stratégie prescrit une initiative à entreprendre en fonction de la mémoire d'un agent

    """

    def __str__(self):

        return 'generic strategie'



    def selection(self, agent):

        return 0








class Glouton(Strategie):

    """

    La stratégie Epsilon-Greedy choisira une initiative aléatoire avec probabilité epsilon
    et adoptera la meilleure approche apparente avec probabilité 1-epsilon.
    Si plusieurs actions sont à égalité pour le meilleur choix,
    alors une initiative aléatoire de ce sous-ensemble est choisie.

    """

    def __init__(self, epsilon):

        self.epsilon = epsilon



    def __str__(self):

        return '\u03B5-greedy (\u03B5={})'.format(self.epsilon)



    def selection(self, agent):

        if np.random.random() < self.epsilon:

            return np.random.choice(len(agent.value_estimates))

        else:

            initiative = np.argmax(agent.value_estimates)

            check = np.where(agent.value_estimates == initiative)[0]

            if len(check) == 0:

                return initiative

            else:

                return np.random.choice(check)






class banditGradient(Strategie):

    """

    La stratégie banditGradient convertit les récompenses estimées du bras en probabilités
    puis prélève des échantillons au hasard de la distribution résultante.
    Cette stratégie est principalement utilisée par l'agent de gradient
    pour l'apprentissage des préférences relatives.

    """

    def __str__(self):

        return 'SM'



    def selection(self, agent):

        gen = agent.value_estimates

        yp = np.exp(gen) / np.sum(np.exp(gen))

        cdf = np.cumsum(yp)

        s = np.random.random()

        return np.where(s < cdf)[0][0]
"""
$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
"""
class Map(object):

    def __init__(self, manchots, strategies, label='Multi-Armed Bandit'):

        self.manchots = manchots

        self.strategies = strategies

        self.label = label



    def reboot(self):

        self.manchots.reboot()

        for agent in self.strategies:

            agent.reboot()



    def run(self, trials=1000, experiments=2000):

        resultats = np.zeros((trials, len(self.strategies)))

        optimalite = np.zeros_like(resultats)



        for _ in range(experiments):

            self.reboot()

            for t in range(trials):

                for i, agent in enumerate(self.strategies):

                    initiative = agent.selection()

                    recompense, is_optimal = self.manchots.tirage(initiative)
                    agent.constatation(recompense)



                    resultats[t, i] += int(recompense)

                    if is_optimal:

                        optimalite[t, i] += 1

        resultats = resultats / np.float(experiments)

        optimalite = optimalite / np.float(experiments)
        return resultats , optimalite



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

strategie = banditGradient()
strategies = [
    GradientAgent(manchots, strategie, alpha=0.1),
    Agent(manchots, Glouton(0))

]
map = Map(manchots, strategies, 'Gradient Agents')
resultats, optimalite = map.run(num_essais, num_experimentations)
map.plots(resultats, optimalite)

