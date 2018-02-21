from game_state_capture import load_data,save_actions
from get_predicted_reward import open_tf_session, close_tf_session
import datetime
import tensorflow as tf
from game_loop import GameLoop
gameType='cutthroat'
gameloop = GameLoop(type=gameType,use_nn=True)


def PlayGame(num_games=20):
    open_tf_session()
    print("TF session restored")
    load_data("dummy.csv")
    print("Loaded data")
    global gameloop
    total_wins = 0
    average_opponent_wins = 0
    total_games = 0
    global gameType
    print ("Playing ",num_games, "to get win percentage")
    for i in range(0,num_games):
        wins,average_opponent, total = gameloop.run()
        total_wins += wins
        total_games += total
        average_opponent_wins += average_opponent
        gameloop = GameLoop(type=gameType,use_nn=True)
        
    file = open('metrics.txt','a')
    file.write(str(total_wins/total_games) + ',' + str(average_opponent_wins/total_games) + ',' + str(total_wins) + ',' + str(average_opponent_wins) + ',' + str(total_games) + ','  + (str(datetime.datetime.now())) + '\n')
    file.close()

    print("Win percentage: ", (total_wins/total_games))
    print("Opponent win percentage: ", (average_opponent_wins/total_games))
    print("Total wins = ",total_wins, 'Average Opponent wins = ', average_opponent_wins,'Total games = ', total_games)
    save_actions()
    close_tf_session()

    return (total_wins/total_games)


print(PlayGame())