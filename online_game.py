#!/usr/bin/python
import socket,os,sys
from threading import Thread
import random
from game_loop import *
from board_memory import *
from domino import *
from user import *
from game_state_capture import load_data

gameType = sys.argv[1] 

gameloop = GameLoop(type=gameType)



def StartGame():
    print ("Starting Game")
    load_data("dummy.csv")
    global gameloop
    for i in range(0,500):
        gameloop.run()
        gameloop = GameLoop(type=gameType)
        print("Game Iteration: ", (i+1))


    print_actions()

StartGame()

