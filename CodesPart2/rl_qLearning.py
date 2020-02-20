


import numpy as np

import matplotlib.pyplot as plt



TETA = 0.3

ALPHA = 0.1

TOUTES_LES_ACTIONS_POSSIBLES = ('U', 'D', 'L', 'R')

SMALL_ENOUGH = 1e-3

TOUTES_LES_ACTIONS_POSSIBLES = ('U', 'D', 'L', 'R')

"""
########################################################################################
"""

class Grille: # Environment

  def __init__(self, largeur, hauteur, début):

    self.largeur = largeur

    self.hauteur = hauteur

    self.i = début[0]

    self.j = début[1]



  def set(self, recompenses, initiatives):

    # recompenses should be a dict of: (i, j): rec (row, col): reward

    # initiatives should be a dict of: (i, j): A (row, col): list of possible initiatives

    self.recompenses = recompenses

    self.initiatives = initiatives



  def definirLetat(self, s):

    self.i = s[0]

    self.j = s[1]



  def etatActuel(self):

    return (self.i, self.j)



  def phaseTerminale(self, s):

    return s not in self.initiatives



  def deplacement(self, initiative):

    # check if legal deplacement first

    if initiative in self.initiatives[(self.i, self.j)]:

      if initiative == 'U':

        self.i -= 1

      elif initiative == 'D':

        self.i += 1

      elif initiative == 'R':

        self.j += 1

      elif initiative == 'L':

        self.j -= 1

    # return a reward (if any)

    return self.recompenses.get((self.i, self.j), 0)



  def deplacementDeRetour(self, initiative):

    # these are the opposite of what U/D/L/R should normally do

    if initiative == 'U':

      self.i += 1

    elif initiative == 'D':

      self.i -= 1

    elif initiative == 'R':

      self.j -= 1

    elif initiative == 'L':

      self.j += 1

    # raise an exception if we arrive somewhere we shouldn't be

    # should never happen

    assert(self.etatActuel() in self.tousLesEtats())



  def finDePartie(self):

    # returns true if game is over, else false

    # true if we are in a state where no initiatives are possible

    return (self.i, self.j) not in self.initiatives



  def tousLesEtats(self):

    # possibly buggy but simple way to get all etats

    # either a position that has possible next initiatives

    # or a position that yields a reward

    return set(self.initiatives.keys()) | set(self.recompenses.keys())





def grilleNormale():

  # define a grille that describes the reward for arriving at each state

  # and possible initiatives at each state

  # the grille looks like this

  # F means you can't go there

  # s means début position

  # number means reward at that state

  # .  .  .  1

  # .  F  . -1

  # s  .  .  .

  g = Grille(3, 4, (2, 0))

  recompenses = {(0, 3): 1, (1, 3): -1}

  initiatives = {

    (0, 0): ('D', 'R'),

    (0, 1): ('L', 'R'),

    (0, 2): ('L', 'D', 'R'),

    (1, 0): ('U', 'D'),

    (1, 2): ('U', 'D', 'R'),

    (2, 0): ('U', 'R'),

    (2, 1): ('L', 'R'),

    (2, 2): ('L', 'R', 'U'),

    (2, 3): ('L', 'U'),

  }

  g.set(recompenses, initiatives)

  return g





def grilleNegative(coutDuPas=-0.1):

  # in this game we want to try to minimize the number of moves

  # so we will penalize every deplacement

  g = grilleNormale()

  g.recompenses.update({

    (0, 0): coutDuPas,

    (0, 1): coutDuPas,

    (0, 2): coutDuPas,

    (1, 0): coutDuPas,

    (1, 2): coutDuPas,

    (2, 0): coutDuPas,

    (2, 1): coutDuPas,

    (2, 2): coutDuPas,

    (2, 3): coutDuPas,

  })

  return g



"""
########################################################################################
"""




def Valeurs(Q, g):

  for i in range(g.largeur):

    print("---------------------------")

    for j in range(g.hauteur):

      q = Q.get((i,j), 0)

      if q >= 0:

        print(" %.2f|" % q, end="")

      else:

        print("%.2f|" % q, end="") # -ve sign takes up an extra space

    print("")





def strategie(W, g):

  for i in range(g.largeur):

    print("---------------------------")

    for j in range(g.hauteur):

      a = W.get((i,j), ' ')

      print("  %s  |" % a, end="")

    print("")






def sup_bloc(d):

  # returns the argmax (key) and max (value) from a dictionary

  # put this into a function since we are using it so often

  max_key = None

  max_val = float('-inf')

  for k, q in d.items():

    if q > max_val:

      max_val = q

      max_key = k

  return max_key, max_val







def initiativeAlea(a, eps=0.1):

  # we'll use epsilon-soft to ensure all etats are visited

  # what happens if you don't do this? i.e. eps=0

  p = np.random.random()

  if p < (1 - eps):

    return a

  else:

    return np.random.choice(TOUTES_LES_ACTIONS_POSSIBLES)

"""
#########################################################################################
"""







if __name__ == '__main__':

    # NOTE: if we use the standard grille, there's a good chance we will end up with

    # suboptimal policies

    # e.g.

    # ---------------------------

    #   R  |   R  |   R  |      |

    # ---------------------------

    #   R* |      |   U  |      |

    # ---------------------------

    #   U  |   R  |   U  |   L  |

    # since going R at (1,0) (shown with a *) incurs no cost, it's OK to keep doing that.

    # we'll either end up staying in the same spot, or back to the début (2,0), at which

    # point we whould then just go back up, or at (0,0), at which point we can continue

    # on right.

    # instead, let's penalize each movement so the agent will find a shorter route.

    #

    # grille = grilleNormale()

    grille = grilleNegative(coutDuPas=-0.1)

    # print recompenses

    print("recompenses:")

    Valeurs(grille.recompenses, grille)

    # no Strategie initialization, we will derive our Strategie from most recent S

    # initialize S(s,a)

    S = {}

    etats = grille.tousLesEtats()

    for s in etats:

        S[s] = {}

        for a in TOUTES_LES_ACTIONS_POSSIBLES:
            S[s][a] = 0

    # let's also keep track of how many times S[s] has been updated

    comptes_actualises = {}

    update_counts_sa = {}

    for s in etats:

        update_counts_sa[s] = {}

        for a in TOUTES_LES_ACTIONS_POSSIBLES:
            update_counts_sa[s][a] = 1.0

    # repeat until convergence

    t = 1.0

    deltas = []

    for it in range(10000):

        if it % 100 == 0:
            t += 1e-2

        if it % 2000 == 0:
            print("it:", it)

        # instead of 'generating' an epsiode, we will PLAY

        # an episode within this loop

        s = (2, 0)  # début state

        grille.definirLetat(s)

        # the first (s, rec) tuple is the state we début in and 0

        # (since we don't get a reward) for simply starting the game

        # the last (s, rec) tuple is the terminal state and the final reward

        # the value for the terminal state is by definition 0, so we don't

        # care about updating it.

        a, _ = sup_bloc(S[s])

        Q = 0

        while not grille.finDePartie():
            a = initiativeAlea(a, eps=0.5 / t)  # epsilon-greedy

            # random initiative also works, but slower since you can bump into walls

            # a = np.random.choice(TOUTES_LES_ACTIONS_POSSIBLES)

            rec = grille.deplacement(a)

            s2 = grille.etatActuel()

            # adaptive learning rate

            alpha = ALPHA / update_counts_sa[s][a]

            update_counts_sa[s][a] += 0.005

            # we will update S(s,a) AS we experience the episode

            old_qsa = S[s][a]

            # the difference between SARSA and S-Learning is with S-Learning

            # we will use this max[a']{ S(s',a')} in our update

            # even if we do not end up taking this initiative in the next step

            a2, max_q_s2a2 = sup_bloc(S[s2])

            S[s][a] = S[s][a] + alpha * (rec + TETA * max_q_s2a2 - S[s][a])

            Q = max(Q, np.abs(old_qsa - S[s][a]))

            # we would like to know how often S(s) has been updated too

            comptes_actualises[s] = comptes_actualises.get(s, 0) + 1

            # next state becomes current state

            s = s2

            a = a2

        deltas.append(Q)

    plt.plot(deltas)

    plt.show()

    # determine the Strategie from S*

    # find Q* from S*

    Strategie = {}

    Q = {}

    for s in grille.initiatives.keys():
        a, max_q = sup_bloc(S[s])

        Strategie[s] = a

        Q[s] = max_q

    # what's the proportion of time we spend updating each part of S?

    print("comptes_actualises:")

    total = np.sum(list(comptes_actualises.values()))

    for k, q in comptes_actualises.items():
        comptes_actualises[k] = float(q) / total

    Valeurs(comptes_actualises, grille)

    print("valeurs:")

    Valeurs(Q, grille)

    print("Strategie:")

    strategie(Strategie, grille)

