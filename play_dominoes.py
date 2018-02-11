#!/usr/bin/python
import socket,os,sys
from threading import Thread
import random
from game_loop import *
from board_memory import *
from domino import *
from user import *
from game_state_capture import load_data
import get_predicted_reward
gameType = sys.argv[1] 

gameloop = GameLoop(type=gameType)



def StartGame():
    print ("Starting Game")
    load_data("dummy.csv")
    global gameloop
    total_wins = 0
    average_opponent_wins = 0
    total_games = 0
    for i in range(0,2):
        wins,average_opponent, total = gameloop.run()
        total_wins += wins
        total_games += total
        average_opponent_wins += average_opponent
        gameloop = GameLoop(type=gameType)
        print("wins = ",wins, 'Average Opponent = ', average_opponent,'games = ', total)
        print("Game Iteration: ", (i+1))

    print("Total wins = ",total_wins, 'Average Opponent = ', average_opponent_wins,'Total games = ', total_games)
    print("Win percentage: ", (total_wins/total_games))
    print("Opponent win percentage: ", (average_opponent_wins/total_games))
    print_actions()


StartGame()

