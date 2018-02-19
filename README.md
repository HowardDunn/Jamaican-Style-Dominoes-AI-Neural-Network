# Jamaican-Style-Dominoes-AI-Neural-Network
Testing the viability of creating AI for Jamaican Style Dominoes using principles from Reinforcement Learning and Neural Networks (tensorflow)

## Goal to train a model that can consistently beat a human in game of dominoes ( Caribbean rules)

## Approach use principles of reinforcement learning to train a Neural network model to estimate the best play

The game of dominoes consists of 28 tiles, there are 4 players, each player selects 7 tiles. Each tile is a combination of 2 numbers ranging from [0-6]. e.g. 6/6, 6/5, 3/4 etc.
Each player matches one end of the tile on the board in an anti-clockwise fashion, the first person to play all their tiles wins. A person loses a turn
if they dont have a tile that matches what is on the board. If no one can match a tile on the board, the game is considered to be blocked and the person
with the smallest numerical count of their hand wins the game. E.g 6/6 is valued at 12, 3/4 is valued at 7.

The search space for the game is very large, a good AI can by made by making some guesses to what a good play is but this requires knowning what is in
the other player's hand and also could lead to bias as to what the programmer decides to be a good state.

A trained model using reinforcement learning might be able to create a model that performs consistently well against the best humans.

## Current approach

The rewards are estimated by playing a game, capture the state of the board backtracking if the person won and applying discounting rewards to the previous 
actions ( actions further in the past receives less reward).

The input vector is a 0/1 vector encoding the state of the board and the action taken, this is then mapped to a reward. the input vector size is 141.
Currently the network is a 4 layer model ( 2 hidden layers ) with each hidden layer comprising of 50 neurons. Since the problem isn't classification,
only the loss is minimized and the model is evaluated by testing it against an existing heurisitc algorithm that has 24% success rate against humans.

The model is trained by creating 4 models that play against each other, to reduce the raandomness caused by the hand shuffle, the same game is replayed
with the same hand 50 times to get a good approximation of what is the optimal way to play. For example if there are 4 models and model 1 wins, model
2, 3 and 4 will adjust their estimates of rewards and take a different approach for the next round since they have a chance to play with the same 
intiatial state. If model 1 loses then model 1 has then now update its approach to win. The goal is to reach some form of equilibrium that the model
with the best hand will win most of the time. This is then repeated for different hand selections so the models get good exposure and experience.


## How to run:

### if you're training for the first time:
python3 play_dominoes.py
python3 train_model.py 

### Run this to train a more robust model:
python3 play_dominoes.py cutthroat <number_of_game_iterations>


### How to evaluate:

python3 evaluate_dominoes.py