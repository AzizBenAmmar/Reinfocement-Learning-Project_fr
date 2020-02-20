import numpy as np





class Grille:

  def __init__(self, largeur, hauteur, debut):

    self.largeur = largeur

    self.hauteur = hauteur

    self.u = debut[0]

    self.w = debut[1]



  def set(self, recompenses, initiatives):


    self.recompenses = recompenses

    self.initiatives = initiatives



  def definirLetat(self, s):

    self.u = s[0]

    self.w = s[1]



  def etatActuel(self):

    return (self.u, self.w)



  def phaseTerminale(self, s):

    return s not in self.initiatives



  def deplacement(self, initiative):


    if initiative in self.initiatives[(self.u, self.w)]:

      if initiative == 'U':

        self.u -= 1

      elif initiative == 'D':

        self.u += 1

      elif initiative == 'R':

        self.w += 1

      elif initiative == 'L':

        self.w -= 1


    return self.recompenses.get((self.u, self.w), 0)



  def deplacementDeRetour(self, initiative):


    if initiative == 'U':

      self.u += 1

    elif initiative == 'D':

      self.u -= 1

    elif initiative == 'R':

      self.w -= 1

    elif initiative == 'L':

      self.w += 1


    assert(self.etatActuel() in self.tousLesEtats())



  def finDePartie(self):


    return (self.u, self.w) not in self.initiatives



  def tousLesEtats(self):


    return set(self.initiatives.keys()) | set(self.recompenses.keys())





def grilleNormale():


  o = Grille(3, 4, (2, 0))

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

  o.set(recompenses, initiatives)

  return o





def grilleNegative(coutDuPas=-0.1):


  o = grilleNormale()

  o.recompenses.update({

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

  return o

"""
##################################################################"
"""

epsilon = 1e-3



def Valeurs(Q, o):

  for u in range(o.largeur):

    print("---------------------------")

    for w in range(o.hauteur):

      q = Q.get((u,w), 0)

      if q >= 0:

        print(" %.2f|" % q, end="")

      else:

        print("%.2f|" % q, end="")

    print("")





def strategie(W, o):

  for u in range(o.largeur):

    print("---------------------------")

    for w in range(o.hauteur):

      a = W.get((u,w), ' ')

      print("  %s  |" % a, end="")

    print("")







epsilon = 1e-3

TETA = 0.3

TOUTES_LES_ACTIONS_POSSIBLES = ('U', 'D', 'L', 'R')

if __name__ == '__main__':


    grille = grilleNegative()


    print("recompenses:")

    Valeurs(grille.recompenses, grille)


    Strategie = {}

    for s in grille.initiatives.keys():
        Strategie[s] = np.random.choice(TOUTES_LES_ACTIONS_POSSIBLES)

    print("Strategie initiale:")

    strategie(Strategie, grille)


    Q = {}

    etats = grille.tousLesEtats()

    for s in etats:


        if s in grille.initiatives:

            Q[s] = np.random.random()

        else:



            Q[s] = 0



    while True:

        grandChangement = 0

        for s in etats:

            ancien_q = Q[s]


            if s in Strategie:

                nv_q = float('-inf')

                for a in TOUTES_LES_ACTIONS_POSSIBLES:

                    grille.definirLetat(s)

                    rec = grille.deplacement(a)

                    q = rec + TETA * Q[grille.etatActuel()]

                    if q > nv_q:
                        nv_q = q

                Q[s] = nv_q

                grandChangement = max(grandChangement, np.abs(ancien_q - Q[s]))

        if grandChangement < epsilon:
            break


    for s in Strategie.keys():

        meilleur_b = None

        meilleur_valeur = float('-inf')



        for a in TOUTES_LES_ACTIONS_POSSIBLES:

            grille.definirLetat(s)

            rec = grille.deplacement(a)

            q = rec + TETA * Q[grille.etatActuel()]

            if q > meilleur_valeur:
                meilleur_valeur = q

                meilleur_b = a

        Strategie[s] = meilleur_b


    print("valeurs:")

    Valeurs(Q, grille)

    print("Strategie:")

    strategie(Strategie, grille)