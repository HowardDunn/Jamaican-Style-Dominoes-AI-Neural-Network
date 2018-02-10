class Domino(object):

    def __init__ (self):
        self.suite1 = -1
        self.suite2 = -1
    def __str__(self):
        return str(self.suite1) +'/' + str(self.suite2)
    def isDouble(self):
        return (self.suite1 == self.suite2)
    def getCount(self):
        return (self.suite1 + self.suite2)
    def setSuites(self,suiteLeft,suiteRight):
        self.suite1 = suiteLeft
        self.suite2 = suiteRight
    def isCompatible(self,suite):
        if(self.suite1 == suite):
            return True
        if(self.suite2 == suite):
            return True

        return False