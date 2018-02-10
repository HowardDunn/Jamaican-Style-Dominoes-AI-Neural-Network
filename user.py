from board_memory import HandMemory
from domino import Domino
import monitor


class OnlineUser(object):

    def __init__(self, wins):
        self.wins = wins
        self.hand = []
        self.playerNumber = 0
        self.cardsRemaining = 7
        self.email = ''
        self.status = 'online'
        self.played = False
        self.connection = ()
        self.playerType = 'computer'
        self.handMemory = HandMemory()
        self.playerState = 'Play'
        self.playerName = 'Computer'
        self.game = ()
        self.passed_on = []


    def getHandCount(self,game):

        count = 0

        for i in range(0,7):
            if(self.hand[i] == -1):
                continue
            count += game.dominoes[self.hand[i]].getCount()

        return count

    def Listen(self,connection,game):
        print ("Listening to player")
        self.connection = connection
        self.game = game
        while True:

            client_msg = []
            while len(client_msg) == 0:
                client_msg = connection.recv(1024).split()

            print(client_msg)

            if(client_msg[0] == 'play'):

                print("Playing card")
                #ideally this is after a proper response
                choice = int(client_msg[1])

                side = client_msg[2].rstrip()
                monitor.player_response.acquire()

                self.game.response = self.game.acceptCard(self.game.players[self.game.playerTurn], choice, side)
                print ("Game Responding with: ", self.game.response)
                self.game.waiting_for_player = False
                monitor.player_response.notify()

                monitor.player_response.release()
            elif(client_msg[0] == 'leave'):

                self.playerType = 'computer'

            elif(client_msg[0] == 'resume'):

                self.playerType = 'human'



    def Send(self,message):
        self.connection.send(message)

    def Pose(self,game):

        propensity = 0
        choice = 0

        if(game.gameCount == 0 or game.gameDrawn == True):
            for i in range(0,7):
                if(self.hand[i] == 27):
                    self.hand[i] = -1
                    return 27
        else:
            strong = self.handMemory.getStrongCard()

            for i in range(0,7):

                temp = 1 + game.dominoes[self.hand[i]].getCount()

                if(game.dominoes[self.hand[i]].isDouble()):
                    temp *= 10
                    if(game.dominoes[self.hand[i]].suite1 == strong):
                        temp *= 40
                if(temp > propensity):
                    choice = i
                    propensity = temp

            val = self.hand[choice]
            self.hand[choice] = -1
            self.cardsRemaining -= 1
        return val


    def canPlay(self,game,suiteLeft,suiteRight):

        for i in range(0,7):
            if(self.hand[i] < 0):
                continue
            elif(game.dominoes[self.hand[i]].isCompatible(suiteLeft)  or
                     game.dominoes[self.hand[i]].isCompatible(suiteRight)):
                return True
        if not suiteLeft in self.passed_on:
            self.passed_on.append[suitLeft]
        if not suiteRight in self.passed_on:
            self.passed_on.append[suitLeft]
        return False


    def playCard(self, suiteLeft, suiteRight,game,boardMemory):

        leftPropensity = -1
        leftChoice = -1

        rightPropensity = -1
        rightChoice = -1

        strongCard = self.handMemory.getStrongCard()

        # Complexity n+n
        for i in range(0,7):

            if(self.hand[i] < 0):
                continue
            elif(not game.dominoes[self.hand[i]].isCompatible(suiteLeft)):
                continue
            else:
                propensity = 1 + game.dominoes[self.hand[i]].getCount()

                if( strongCard != suiteLeft ):
                    propensity += 10
                if( boardMemory.getCount(suiteLeft) < 5):
                    propensity += 10
                if(game.dominoes[self.hand[i]].isDouble()):
                    propensity *= 5
                if(boardMemory.getCount(game.dominoes[self.hand[i]].suite1) == 6):
                    propensity = 1
                if(boardMemory.getCount(game.dominoes[self.hand[i]].suite2) == 6):
                    propensity = 1

                if(propensity > leftPropensity):
                    leftChoice = i
                    leftPropensity = propensity
        # Complexity n+n
        for i in range(0,7):

            if(self.hand[i] < 0):
                continue
            elif(not game.dominoes[self.hand[i]].isCompatible(suiteRight)):
                continue
            else:
                propensity = 1 + game.dominoes[self.hand[i]].getCount()

                if( strongCard != suiteRight ):
                    propensity += 10
                if( boardMemory.getCount(suiteRight) < 5):
                    propensity += 10
                if(game.dominoes[self.hand[i]].isDouble()):
                    propensity *= 5
                if(boardMemory.getCount(game.dominoes[self.hand[i]].suite1) == 6):
                    propensity = 1
                if(boardMemory.getCount(game.dominoes[self.hand[i]].suite2) == 6):
                    propensity = 1

                if(propensity > rightPropensity):
                    rightChoice = i
                    rightPropensity = propensity

        if(leftPropensity == -1 and rightPropensity == -1):

            return (0,-1)
        elif(leftPropensity > rightPropensity):

            val = self.hand[leftChoice]

            if(self.hand[leftChoice] > 0):
                self.handMemory.remove(game.dominoes[self.hand[leftChoice]].suite1,
                                game.dominoes[self.hand[leftChoice]].suite2)
                self.cardsRemaining -= 1

            self.hand[leftChoice] = -1
            return (0,val)
        else:
            val = self.hand[rightChoice]

            if(self.hand[rightChoice] > 0):
                self.handMemory.remove(game.dominoes[self.hand[rightChoice]].suite1,
                    game.dominoes[self.hand[rightChoice]].suite2)
                self.cardsRemaining -= 1

            self.hand[rightChoice] = -1
            return (1,val)
