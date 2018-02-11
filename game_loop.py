from board_memory import *
from user import *
from domino import Domino
import socket,os,sys,random, time
from threading import Thread
import monitor
from game_state_capture import *
print_on = False

def print_jsd(vals):
    
    if print_on == True:
        print(vals)

class GameLoop(object):

    def __init__(self,type,use_nn=False):

        self.type = type
        self.gameCount = 0
        self.playerTurn = 0
        self.playerCount = 0
        self.suiteLeft = 6
        self.suiteRight = 6
        self.start_time = 0
        self.end_time = 0
        self.delay_start = 0
        self.posed = False
        self.response = ''
        self.previousWinner = -1
        player_strategy = 'normal'
        if use_nn:
            player_strategy = 'neural_network'
        self.player1 = Player(wins=0,player_strategy=player_strategy)
        self.player2 = Player(wins=0)
        self.player3 = Player(wins=0)
        self.player4 = Player(wins=0)
        self.player1.playerNumber = 0
        self.player2.playerNumber = 1
        self.player3.playerNumber = 2
        self.player4.playerNumber = 3
        self.dominoes =  [ Domino() for i in range(28)]
        self.dominoes[0].setSuites(0, 0)
        self.dominoes[14].setSuites(4, 4)
        self.dominoes[1].setSuites(1, 0)
        self.dominoes[15].setSuites(5, 0)
        self.dominoes[2].setSuites(1, 1)
        self.dominoes[16].setSuites(5, 1)
        self.dominoes[3].setSuites(2, 0)
        self.dominoes[17].setSuites(5, 2)
        self.dominoes[4].setSuites(2, 1)
        self.dominoes[18].setSuites(5, 3)
        self.dominoes[5].setSuites(2, 2)
        self.dominoes[19].setSuites(5, 4)
        self.dominoes[6].setSuites(3, 0)
        self.dominoes[20].setSuites(5, 5)
        self.dominoes[7].setSuites(3, 1)
        self.dominoes[21].setSuites(6, 0)
        self.dominoes[8].setSuites(3, 2)
        self.dominoes[22].setSuites(6, 1)
        self.dominoes[9].setSuites(3, 3)
        self.dominoes[23].setSuites(6, 2)
        self.dominoes[10].setSuites(4, 0)
        self.dominoes[24].setSuites(6, 3)
        self.dominoes[11].setSuites(4, 1)
        self.dominoes[25].setSuites(6, 4)
        self.dominoes[12].setSuites(4, 2)
        self.dominoes[26].setSuites(6, 5)
        self.dominoes[13].setSuites(4, 3)
        self.dominoes[27].setSuites(6, 6)
        self.boardMemory = BoardMemory()
        self.waiting_for_player = True
        self.players = []
        self.players.append(self.player1)
        self.players.append(self.player2)
        self.players.append(self.player3)
        self.players.append(self.player4)
        self.gameDrawn = False
        self.gameData = {'posed':27,'LEFT'.lower() : [], 'right': [], 'order': [],
                         'player1': {'name:':self.player1.playerName,'wins':self.player1.wins,
                                     'cardsRemaining: ': self.player1.cardsRemaining},
                         'player2': {'name:': self.player2.playerName, 'wins': self.player2.wins,
                                     'cardsRemaining: ': self.player2.cardsRemaining},
                         'player3': {'name:': self.player1.playerName, 'wins': self.player1.wins,
                                     'cardsRemaining: ': self.player1.cardsRemaining},
                         'player4': {'name:': self.player4.playerName, 'wins': self.player4.wins,
                                     'cardsRemaining: ': self.player4.cardsRemaining},
                         'hand': [],

                         'status': [],
                         }

        self.gameState = "Wait for players"


    def initializeHands(self):

        distribution = random.sample(range(0,28),28)
        self.player1.hand = []
        self.player2.hand = []
        self.player3.hand = []
        self.player4.hand = []
        self.boardMemory = BoardMemory()

        self.player1.handMemory = HandMemory()
        self.player2.handMemory = HandMemory()
        self.player3.handMemory = HandMemory()
        self.player4.handMemory = HandMemory()

        for i in range(0,7):
            self.player1.cardsRemaining = 7
            self.player2.cardsRemaining = 7
            self.player3.cardsRemaining = 7
            self.player4.cardsRemaining = 7

            self.player1.handMemory.update(self.dominoes[distribution[i]].suite1,
                                                self.dominoes[distribution[i]].suite2)
            self.player2.handMemory.update(self.dominoes[distribution[i+7]].suite1,
                                                self.dominoes[distribution[i+7]].suite2)
            self.player3.handMemory.update(self.dominoes[distribution[i+14]].suite1,
                                                self.dominoes[distribution[i+14]].suite2)
            self.player4.handMemory.update(self.dominoes[distribution[i+21]].suite1,
                                                self.dominoes[distribution[i+21]].suite2)

            self.player1.hand.append(distribution[i])
            self.player2.hand.append(distribution[i+7])
            self.player3.hand.append(distribution[i+14])
            self.player4.hand.append(distribution[i+21])

            self.player1.passed_on = []
            self.player2.passed_on = []
            self.player3.passed_on = []
            self.player4.passed_on = []


    def findDoubleSix(self):

        for i in range(0,7):

            if(self.player1.hand[i] == 27):
                return 0
            elif(self.player2.hand[i] == 27):
                return 1
            elif(self.player3.hand[i] == 27):
                return 2
            elif(self.player4.hand[i] == 27):
                return 3

    def getWinner(self):

        if(self.player1.wins > 5):
                return 0
        elif(self.player2.wins > 5):
                return 1
        elif(self.player3.wins > 5):
                return 2
        elif(self.player4.wins > 5):
                return 3

        return -1

    def getRoundWinner(self):

        if(self.player1.cardsRemaining <= 0):
            return 0
        elif(self.player2.cardsRemaining <= 0):
            return 1
        elif(self.player3.cardsRemaining <= 0):
            return 2
        elif(self.player4.cardsRemaining <= 0):
            return 3
        else:
            return -1

    def acceptCard(self,player,dominoNumber,side):

        if (self.playerTurn != player.playerNumber):
            return 'NOT YOUR TURN'

        if(self.gameState == 'Pose'):
            if(self.gameCount == 0 and dominoNumber != 27):
                player.Send('INVALID, YOU MUST POSE DOUBLE SIX')
                return 'INVALID, YOU MUST POSE DOUBLE SIX'

            self.suiteLeft = self.dominoes[dominoNumber].suite1
            self.suiteRight = self.dominoes[dominoNumber].suite2

            self.boardMemory.update(self.dominoes[dominoNumber].suite1,
                                    self.dominoes[dominoNumber].suite2)

            if dominoNumber in player.hand:
                change = player.hand.index(dominoNumber)
                player.hand[change] = -1
                player.cardsRemaining -= 1

            return 'SUCCESS'

        if not dominoNumber in player.hand:
           return 'INVALID, CARD NOT IN HAND'

        if (self.dominoes[dominoNumber].suite1 == self.suiteLeft and side == 'left'):

            print_jsd('Left side matched')
            self.suiteLeft = self.dominoes[dominoNumber].suite2


        elif(self.dominoes[dominoNumber].suite2 == self.suiteLeft and side == 'left'):

            print_jsd('Left side matched')
            self.suiteLeft = self.dominoes[dominoNumber].suite1


        elif side == 'left' :
            return 'INVALID CARD'


        # RIGHT SIDE
        if (self.dominoes[dominoNumber].suite1 == self.suiteRight and side == 'right' ):

            print_jsd('Right side matched')
            self.suiteRight = self.dominoes[dominoNumber].suite2



        elif(self.dominoes[dominoNumber].suite2 == self.suiteRight and side == 'right' ):

            print_jsd ('Right side matched')
            self.suiteRight = self.dominoes[dominoNumber].suite1


        elif side == 'right':
            return 'INVALID CARD'


        self.boardMemory.update(self.dominoes[dominoNumber].suite1,
                                    self.dominoes[dominoNumber].suite2)

        if dominoNumber in player.hand:
            change = player.hand.index(dominoNumber)
            player.hand[change] = -1
            player.cardsRemaining -= 1

        return 'SUCCESS'

    def run(self):
        print_jsd ('Running Game on Thread')

        count = 0
        while self.getWinner() == -1:

            if(self.gameState == "Wait for players"):
                self.gameState = "Shuffle"
                if (self.playerCount > 0):
                    self.gameState = "Shuffle"


            if(self.gameState == "Shuffle"):
                global print_on
                #print_on = False
                self.initializeHands()
                print_jsd(self.player1.hand)
                print_jsd (self.player2.hand)
                print_jsd (self.player3.hand)
                print_jsd (self.player4.hand)
                reset_board() # this is for the capture
                if(self.previousWinner == -1 or self.gameDrawn == True):
                    self.playerTurn = self.findDoubleSix()
                    print_jsd ('It is player ' + str(self.playerTurn + 1) + 's turn')
                else:
                    self.playerTurn = self.previousWinner

                self.gameState = "Pose"

            elif(self.gameState == "Pose" ):

                if (self.players[self.playerTurn].playerType == 'computer'):

                    card =   self.players[self.playerTurn].Pose(self)

                    self.boardMemory.update(self.dominoes[card].suite1,
                                                    self.dominoes[card].suite2)

                    print_jsd ('Player ' + str(self.playerTurn + 1) + ' posed ' + str(self.dominoes[card]))
                    capture_state(self.playerTurn,card,self.players[self.playerTurn].hand,self.suiteLeft,self.suiteRight,
                    hand_sizes=[self.players[(self.playerTurn + 1)%4].cardsRemaining,
                     self.players[(self.playerTurn + 2)%4].cardsRemaining,
                     self.players[(self.playerTurn + 3)%4].cardsRemaining],
                     passed_arrays=[self.players[(self.playerTurn + 1)%4].passed_on,
                      self.players[(self.playerTurn + 2)%4].passed_on,
                      self.players[(self.playerTurn + 3)%4].passed_on,
                     ],side=0)

                    self.playerTurn += 1
                    self.playerTurn = self.playerTurn % 4
                    self.gameDrawn = False
                    self.suiteLeft = self.dominoes[card].suite1
                    self.suiteRight = self.dominoes[card].suite2


                    self.gameState = 'Delay'
                    self.delay_start = time.time()
                else:
                    #TODO change this back to local
                    print_jsd('Waiting for Player.....')
                    message = ''
                    for i in range(0, 7):
                        if (self.players[self.playerTurn].hand[i] != -1):
                            print_jsd('Card: ' + str(self.dominoes[self.players[self.playerTurn].hand[i]]) + ' -> ' + str(
                                self.players[self.playerTurn].hand[i]))
                            message += 'Card: ' + str(
                                self.dominoes[self.players[self.playerTurn].hand[i]]) + ' -> ' + str(
                                self.players[self.playerTurn].hand[i]) + '\n'


                    message += 'Enter your choice: '
                    self.players[self.playerTurn].Send(message)

                    monitor.player_response.acquire()
                    print_jsd ("Acquiring Lock to wait for player response")
                    while (self.waiting_for_player):
                        monitor.player_response.wait()

                    monitor.player_response.release()

                    self.waiting_for_player = True

                    if (self.response != 'SUCCESS'):
                        message = 'INVALID, Enter your choice: '
                        self.players[self.playerTurn].Send(message)
                        continue
                    else:
                        self.playerTurn += 1
                        self.playerTurn = self.playerTurn % 4
                        self.gameState = 'Delay'


            elif(self.gameState == 'Delay'):

                if(time.time()-self.delay_start > 0):
                    self.gameState = 'WaitForPlayer'

            elif(self.gameState == "WaitForPlayer"):

                count += 1
                print_jsd ('Turn Count: ' + str(count) + ' Left: ' + str(self.suiteLeft) + ' Right: ' + str(self.suiteRight))

                print_jsd ('It is player ' + str(self.playerTurn + 1) + 's turn')
                print_jsd ('Hand' +  str(self.playerTurn + 1) +': ' + str(self.players[self.playerTurn].hand))

                if(self.players[self.playerTurn].playerType == 'computer'):

                    card = self.players[self.playerTurn].playCard(self.suiteLeft,
                                        self.suiteRight,self,self.boardMemory,hand_sizes=[self.players[(self.playerTurn + 1)%4].cardsRemaining,
                                            self.players[(self.playerTurn + 2)%4].cardsRemaining,
                                            self.players[(self.playerTurn + 3)%4].cardsRemaining],
                                            passed_arrays=[self.players[(self.playerTurn + 1)%4].passed_on,
                                                self.players[(self.playerTurn + 2)%4].passed_on,
                                                self.players[(self.playerTurn + 3)%4].passed_on,
                                            ])


                    if(card[1] != -1):
                        self.boardMemory.update(self.dominoes[card[1]].suite1,
                                                self.dominoes[card[1]].suite2)
                        if(card[0] == 0):

                            if(self.dominoes[card[1]].suite1 == self.suiteLeft):
                                self.suiteLeft = self.dominoes[card[1]].suite2
                            else:
                                self.suiteLeft = self.dominoes[card[1]].suite1

                            print_jsd ('Player ' + str(self.playerTurn + 1) + ' played left: ' + str(self.dominoes[card[1]]) )
                            capture_state(self.playerTurn,card[1],self.players[self.playerTurn].hand,self.suiteLeft,self.suiteRight,
                                            hand_sizes=[self.players[(self.playerTurn + 1)%4].cardsRemaining,
                                            self.players[(self.playerTurn + 2)%4].cardsRemaining,
                                            self.players[(self.playerTurn + 3)%4].cardsRemaining],
                                            passed_arrays=[self.players[(self.playerTurn + 1)%4].passed_on,
                                                self.players[(self.playerTurn + 2)%4].passed_on,
                                                self.players[(self.playerTurn + 3)%4].passed_on,
                                            ],side=0)

                        else:

                            if(self.dominoes[card[1]].suite1 == self.suiteRight):
                                self.suiteRight = self.dominoes[card[1]].suite2
                            else:
                                self.suiteRight = self.dominoes[card[1]].suite1
                            print_jsd ('Player ' + str(self.playerTurn + 1) + ' played right: ' + str(self.dominoes[card[1]]) )
                            capture_state(self.playerTurn,card[1],self.players[self.playerTurn].hand,self.suiteLeft,self.suiteRight,
                                            hand_sizes=[self.players[(self.playerTurn + 1)%4].cardsRemaining,
                                            self.players[(self.playerTurn + 2)%4].cardsRemaining,
                                            self.players[(self.playerTurn + 3)%4].cardsRemaining],
                                            passed_arrays=[self.players[(self.playerTurn + 1)%4].passed_on,
                                                self.players[(self.playerTurn + 2)%4].passed_on,
                                                self.players[(self.playerTurn + 3)%4].passed_on,
                                            ],side=1)
                    
                    else:
                        print_jsd ('Player ' + str(self.playerTurn + 1) + ' passed')
                        reward_player(self.playerTurn,-7) # reward the player -7 because they passed
                    self.playerTurn += 1
                    self.playerTurn = self.playerTurn % 4
                    self.gameState = 'CheckGameState'
                    print_jsd  ('\n')
                else:
                    print_jsd('Waiting for Player.....')
                    message = ''
                    for i in range(0,7):
                        if(self.players[self.playerTurn].hand[i] != -1):
                            print_jsd ('Card: ' + str(self.dominoes[self.players[self.playerTurn].hand[i]]) + ' -> ' + str(self.players[self.playerTurn].hand[i]))
                            message += 'Card: ' + str(self.dominoes[self.players[self.playerTurn].hand[i]]) + ' -> ' + str(self.players[self.playerTurn].hand[i]) + '\n'

                    if(not self.players[self.playerTurn].canPlay(self,self.suiteLeft,self.suiteRight)):
                        print_jsd ('Player ' + str(self.playerTurn + 1) + ' passed')
                        message += ('Player ' + str(self.playerTurn + 1) + ' passed\n')
                        self.players[self.playerTurn].Send(message)
                        self.playerTurn += 1
                        self.playerTurn = self.playerTurn % 4
                        self.gameState = 'CheckGameState'
                        continue

                    message += 'Enter your choice: '
                    self.players[self.playerTurn].Send(message)

                    monitor.player_response.acquire()
                    print_jsd ("Acquiring Lock to wait for player response")
                    while(self.waiting_for_player):
                        monitor.player_response.wait()

                    monitor.player_response.release()

                    self.waiting_for_player = True

                    if(self.response != 'SUCCESS'):
                        message = 'INVALID, Enter your choice: '
                        self.players[self.playerTurn].Send(message)
                        continue
                    else:
                        self.playerTurn += 1
                        self.playerTurn = self.playerTurn % 4
                        self.gameState = 'CheckGameState'

            elif(self.gameState == 'CheckGameState'):

                winner = self.getWinner()
                roundWinner = self.getRoundWinner()

                if winner != -1:
                    print_jsd('Player ' + str(winner + 1) + ' has won the game')
                    print_jsd('Player1 wins: ' + str(self.player1.wins))
                    print_jsd('Player2 wins: ' + str(self.player2.wins))
                    print_jsd('Player3 wins: ' + str(self.player3.wins))
                    print_jsd('Player4 wins: ' + str(self.player4.wins))
                    


                elif roundWinner != -1:

                    print_jsd('Player ' + str(roundWinner + 1) + ' has won the round\n')
                    self.players[roundWinner].wins += 1
                    self.previousWinner = roundWinner
                    self.gameCount += 1
                    self.gameState = "Shuffle"
                    reward_player(roundWinner,50)

                elif (self.suiteLeft == self.suiteRight):
                    if(self.boardMemory.getCount(self.suiteLeft) == 7):
                        print_jsd('Game Blocked')

                        count1 = self.player1.getHandCount(self)
                        count2 = self.player2.getHandCount(self)
                        count3 = self.player3.getHandCount(self)
                        count4 = self.player4.getHandCount(self)

                        print_jsd ('Player1: ' + str(count1))
                        print_jsd ('Player2: ' + str(count2))
                        print_jsd ('Player3: ' + str(count3))
                        print_jsd ('Player4: ' + str(count4))
                        print_jsd ('\n')
                        lowest = 0
                        lowestCount = count1

                        if(count2 < lowestCount):
                            lowest = 1
                            lowestCount = count2
                        elif(count2 == lowestCount):
                            lowest = 5
                        if(count3 < lowestCount):
                            lowest = 2
                            lowestCount = count3
                        elif(count3 == lowestCount):
                            lowest = 5
                        if(count4 < lowestCount):
                            lowest = 3
                            lowestCount = count4
                        elif(count4 == lowestCount):
                            lowest = 5

                        if(lowest == 5):
                            self.gameDrawn = True
                            print_jsd ('Game Drawn') 
                        else:
                            self.players[lowest].wins += 1
                            self.previousWinner = lowest
                            reward_player(lowest,50)


                        self.gameCount += 1

                        self.gameState = 'Shuffle'
                    else:
                        self.gameState = 'Delay'
                        self.delay_start = time.time()
                else:
                    self.gameState = 'Delay'
                    self.delay_start = time.time()
        
        print_jsd('Player ' + str(self.getWinner()+1) + ' has won the game')
        print_jsd('Player1 wins: ' + str(self.player1.wins))
        print_jsd('Player2 wins: ' + str(self.player2.wins))
        print_jsd('Player3 wins: ' + str(self.player3.wins))
        print_jsd('Player4 wins: ' + str(self.player4.wins))
        print_jsd ('Games played: ' + str(self.gameCount))
        average_opponent = self.player2.wins + self.player3.wins + self.player4.wins
        average_opponent /= 3.0
        return self.player1.wins,average_opponent, self.gameCount
        