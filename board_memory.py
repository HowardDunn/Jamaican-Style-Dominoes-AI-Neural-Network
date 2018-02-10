from domino import Domino

class BoardMemory(object):

    def __init__ (self):

        self.blanks = []
        self.ones = []
        self.twos = []
        self.threes = []
        self.fours = []
        self.fives = []
        self.sixes = []

    def getCount(self,suite):

        if(suite == 0):
            return len(self.blanks)
        elif(suite == 1):
            return len(self.ones)
        elif(suite == 2):
            return len(self.twos)
        elif(suite == 3):
            return len(self.threes)
        elif(suite == 4):
            return len(self.fours)
        elif(suite == 5):
            return len(self.fives)
        elif(suite == 6):
            return len(self.sixes)

    def update(self,suite1,suite2):

        if(suite1 == 0):
            self.blanks.append(suite2)
        elif(suite1 == 1):
            self.ones.append(suite2)
        elif(suite1 == 2):
            self.twos.append(suite2)
        elif(suite1 == 3):
            self.threes.append(suite2)
        elif(suite1 == 4):
            self.fours.append(suite2)
        elif(suite1 == 5):
            self.fives.append(suite2)
        elif(suite1 == 6):
            self.sixes.append(suite2)

        if(suite1 == suite2):
            return None

        if(suite2 == 0):
            self.blanks.append(suite1)
        elif(suite2 == 1):
            self.ones.append(suite1)
        elif(suite2 == 2):
            self.twos.append(suite1)
        elif(suite2 == 3):
            self.threes.append(suite1)
        elif(suite2 == 4):
            self.fours.append(suite1)
        elif(suite2 == 5):
            self.fives.append(suite1)
        elif(suite2 == 6):
            self.sixes.append(suite1)

class HandMemory(BoardMemory):

    def getStrongCard(self):

        cardSuites = []
        cardSuites.extend([self.blanks,self.ones,self.twos,self.threes])
        cardSuites.extend([self.fours,self.fives,self.sixes])
        strongest = 0
        largest = -1

        for i in range(0,6):
            if(len(cardSuites[i]) > largest):

                largest = len(cardSuites[i])
                strongest = i

        return strongest

    def remove(self,suite1,suite2):

        if(suite1 == 0):
            self.blanks.remove(suite2)
        elif(suite1 == 1):
            self.ones.remove(suite2)
        elif(suite1 == 2):
            self.twos.remove(suite2)
        elif(suite1 == 3):
            self.threes.remove(suite2)
        elif(suite1 == 4):
            self.fours.remove(suite2)
        elif(suite1 == 5):
            self.fives.remove(suite2)
        elif(suite1 == 6):
            self.sixes.remove(suite2)

        if(suite1 == suite2):
            return None

        if(suite2 == 0):
            self.blanks.remove(suite1)
        elif(suite2 == 1):
            self.ones.remove(suite1)
        elif(suite2 == 2):
            self.twos.remove(suite1)
        elif(suite2 == 3):
            self.threes.remove(suite1)
        elif(suite2 == 4):
            self.fours.remove(suite1)
        elif(suite2 == 5):
            self.fives.remove(suite1)
        elif(suite2 == 6):
            self.sixes.remove(suite1)