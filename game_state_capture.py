


# [
#  <card played: 0-27>, <agent_hand: 28-55>, <left side: 56-62>, <right side: 63-69>, <cards played: 70-97>,
#   <player1_hand size: 98-104>, <player1_passed: 105-111>,
#   <player2_hand size: 112-118>, <player1_passed: 119 - 125>,
#   <player3_hand size: 126-132>, <player1_passed: 133 - 140>
# # ]
import sqlite3
import csv

board_state = [0]*28
player_actions = [[],[],[],[]]
total_actions = {}

def build_state(card,agent_hand,left_side,right_side,hand_sizes = [],passed_arrays=[],side=0):
    global board_state
    provisional_board = list(board_state)
    game_state = [0]*141
    provisional_board[card] = 1
    game_state[card] = 1

        # capture the agent hand
    for card in agent_hand:
        
        if card != -1:
            
            game_state[28 + card] = 1

     #capture the values of the left and right side when the card is 
    # played
        game_state[56 + left_side] = 1
    game_state[63 + right_side] = 1

        # capture the cards played in this instant
    for i in range(0,28):
        game_state[70 + i] = provisional_board[i]
        
    game_state[98 + hand_sizes[0]] = 1
    game_state[112 + hand_sizes[1]] = 1
    game_state[126 + hand_sizes[2]] = 1
    for passed_card in passed_arrays[0]:
            game_state[105 + passed_card] = 1
        
    for passed_card in passed_arrays[0]:
        game_state[119 + passed_card] = 1

    for passed_card in passed_arrays[0]:
        game_state[133 + passed_card] = 1

    game_state[140] = side

    return game_state


def capture_state(player_turn,card_played, agent_hand, left_side, 
            right_side, hand_sizes = [],passed_arrays=[],side=0):

        game_state = [0]*141 # stores the state of the game
        global board_state
        board_state[card_played] = 1

        game_state[card_played] = 1

        # capture the agent hand
        for card in agent_hand:
            
            if card != -1:
                
                game_state[28 + card] = 1

        #capture the values of the left and right side when the card is 
        # played
        game_state[56 + left_side] = 1
        game_state[63 + right_side] = 1

        # capture the cards played in this instant
        for i in range(0,28):
            game_state[70 + i] = board_state[i]
        
        game_state[98 + hand_sizes[0]] = 1
        game_state[112 + hand_sizes[1]] = 1
        game_state[126 + hand_sizes[2]] = 1

        for passed_card in passed_arrays[0]:
            game_state[105 + passed_card] = 1
        
        for passed_card in passed_arrays[0]:
            game_state[119 + passed_card] = 1

        for passed_card in passed_arrays[0]:
            game_state[133 + passed_card] = 1

        game_state[140] = side
        player_actions[player_turn].append((game_state,1))

def reward_player(player, reward):
    num_actions = len(player_actions[player])
    for i,actions in enumerate(player_actions[player]):
        new_reward = actions[1] + (reward / (num_actions-i))
        new_action = (actions[0],new_reward)
        player_actions[player][i] = new_action

def reset_board():
    
    global board_state
    global total_actions
    global player_actions
    board_state = [0]*28
    
    update_data()
    player_actions = [[],[],[],[]]


def update_data():
    global player_actions
    global total_actions

    for i,player in enumerate(player_actions):
        
        for action in player:

            if repr(action[0])[1:-1] in total_actions:
                
                total_actions[repr(action[0])[1:-1]][0] += 1
                total_actions[repr(action[0])[1:-1]][1] += action[1]
                total_actions[repr(action[0])[1:-1]][2] = total_actions[repr(action[0])[1:-1]][1]/total_actions[repr(action[0])[1:-1]][0]
            else:
                total_actions[repr(action[0])[1:-1]] = [1,action[1],action[1],i+1]


def update_nn_data():
    global player_actions
    for action in player_actions[0]:

            if repr(action[0]) in total_actions:
                total_actions[repr(action[0])[1:-1]][0] += 1
                total_actions[repr(action[0])[1:-1]][1] += action[1]
                total_actions[repr(action[0])[1:-1]][2] = total_actions[repr(action[0])[1:-1]][1]/total_actions[repr(action[0])[1:-1]][0]
            else:
                total_actions[repr(action[0])[1:-1]] = [1,action[1],action[1],1]


def WriteDictToCSV(csv_file,csv_columns,dict_data):
    try:
        with open(csv_file, 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
            writer.writeheader()
            for data in dict_data:
                writer.writerow({   'Game_State': data, 
                                    'Times_Seen': dict_data[data][0],
                                    'Total_Reward': dict_data[data][1],
                                    'Reward': dict_data[data][2],
                                    'Player': dict_data[data][3]  
                                    })
                #print(data)
                
    except IOError:
            print("I/O error({0}): {1}".format(errno, strerror))    
    return  

def load_data(csv_file):
    global total_actions
    try:
        with open(csv_file) as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                total_actions[row['Game_State']] = [row['Times_Seen'],row['Total_Reward'],row['Reward'], row['Player']]
    except IOError:
            print("ERROR")

    
    return total_actions

def print_actions():
    global total_actions
    print(len(total_actions))
    csv_columns = ['Game_State','Times_Seen','Total_Reward','Reward','Player']
    WriteDictToCSV("dummy.csv",csv_columns,total_actions)

        

def save_actions():
    global total_actions
    csv_columns = ['Game_State','Times_Seen','Total_Reward','Reward','Player']
    WriteDictToCSV("dummy.csv",csv_columns,total_actions)



def get_actions_and_rewards(player=5):
    if player == 5:
        return total_actions
    
    new_actions = {}

    for action in total_actions:
        if int(total_actions[action][3]) == player:
            new_actions[action] = total_actions[action]
    return new_actions