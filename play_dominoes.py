#!/usr/bin/python
import socket,os,sys
from threading import Thread
import random
from game_loop import *
from board_memory import *
from domino import *
from user import *
from game_state_capture import load_data
from get_predicted_reward import open_tf_session, close_tf_session
import get_predicted_reward


global gameloop
global gameType
global num_games
gameType = None
num_games = None
gameloop = None



def StartGame(num_games=10):
    
    load_data("dummy.csv")
    print('loaded  data')
    global gameloop
    total_wins = 0
    average_opponent_wins = 0
    total_games = 0
    print ("Starting Game")
    for i in range(0,num_games):
        wins,average_opponent, total = gameloop.run()
        total_wins += wins
        total_games += total
        average_opponent_wins += average_opponent
        gameloop = GameLoop(type=gameType,use_nn=True,training=True)
        print("wins = ",wins, 'Average Opponent = ', average_opponent,'games = ', total)
        print("Game Iteration: ", (i+1))
    
        try:
            os.system("scp log1.txt root@clients.axiomarray.com:/home/dahdunn/Jamaican-Style-Dominoes-AI-Neural-Network")
        except:
            print("Unable to update file")
            
    print("Total wins = ",total_wins, 'Average Opponent = ', average_opponent_wins,'Total games = ', total_games)
    print("Win percentage: ", (total_wins/total_games))
    print("Opponent win percentage: ", (average_opponent_wins/total_games))
    print_actions()

if len(sys.argv) == 3:
    gameType = sys.argv[1] 
    num_games = int(sys.argv[2])
    gameloop = GameLoop(type=gameType,use_nn=True,training=True)
    open_tf_session()
    StartGame(num_games=num_games)
    close_tf_session()
else:
    gameloop = GameLoop(type='cutthroat',use_nn=False,training=False)
    wins,average_opponent, total = gameloop.run()
    print("wins = ",wins, 'Average Opponent = ', average_opponent,'games = ', total)
    print_actions()


