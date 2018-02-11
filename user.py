from board_memory import HandMemory
from domino import Domino
import monitor
from get_predicted_reward import get_predicted_reward
from game_state_capture import build_state

class Player(object):

    def __init__(self, wins,player_strategy='normal'):
        self.wins = wins
        self.hand = []
        self.playerNumber = 0
        self.cardsRemaining = 7
        self.played = False
        self.connection = ()
        self.playerType = 'computer'
        self.handMemory = HandMemory()
        self.playerName = 'Computer'
        self.game = ()
        self.passed_on = []
        self.player_strategy = player_strategy


    def getHandCount(self,game):

        count = 0

        for i in range(0,7):
            if(self.hand[i] == -1):
                continue
            count += game.dominoes[self.hand[i]].getCount()

        return count

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

    def playCard(self, suiteLeft, suiteRight,game,boardMemory,hand_sizes = [],passed_arrays=[]):
        
        if self.player_strategy == 'normal':
            return self.playCardNormal(suiteLeft, suiteRight,game,boardMemory)
        elif self.player_strategy == 'neural_network':
            return self.playCardNN(suiteLeft, suiteRight,game,hand_sizes = hand_sizes, passed_arrays=passed_arrays)


    def playCardNN(self, suiteLeft, suiteRight,game,hand_sizes = [],passed_arrays=[]):
        
        chosen_card = -1
        chosen_index = -1
        chosen_side = 0
        highest_reward = -100000000000000000

        for i in range(0,7):
    
            if(self.hand[i] < 0):
                continue
            elif(not game.dominoes[self.hand[i]].isCompatible(suiteLeft)):
                continue
            else:
                temp_hand = list(self.hand)
                temp_hand[i] = -1
                
                game_state = build_state(self.hand[i],temp_hand,suiteLeft,suiteRight,
                                        hand_sizes = hand_sizes, passed_arrays=passed_arrays,side=0)
                reward = get_predicted_reward(game_state)
                
                if reward > highest_reward:
                    chosen_card = self.hand[i]
                    highest_reward = reward
                    chosen_side = 0
                    chosen_index = i
            
        for i in range(0,7):
        
            if(self.hand[i] < 0):
                continue
            elif(not game.dominoes[self.hand[i]].isCompatible(suiteRight)):
                continue
            else:
                temp_hand = list(self.hand)
                temp_hand[i] = -1
                
                game_state = build_state(self.hand[i],temp_hand,suiteLeft,suiteRight,
                                        hand_sizes = hand_sizes, passed_arrays=passed_arrays,side=1)
                reward = get_predicted_reward(game_state)
                
                if reward > highest_reward:
                    chosen_card = self.hand[i]
                    highest_reward = reward
                    chosen_side = 1
                    chosen_index = i

        if chosen_card == -1:
            return (0,-1)
        self.handMemory.remove(game.dominoes[chosen_card].suite1,
                                game.dominoes[chosen_card].suite2)
        self.cardsRemaining -= 1

        self.hand[chosen_index] = -1
        return (chosen_side,chosen_card)
        
    def playCardNormal(self, suiteLeft, suiteRight,game,boardMemory):

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
