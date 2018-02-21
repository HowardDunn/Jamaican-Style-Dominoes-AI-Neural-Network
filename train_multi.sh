#!/bin/bash



python3 play_dominoes.py cutthroat 100 > log1.txt
python3 evaluate_model.py > evaluat1.txt
scp log1.txt root@clients.axiomarray.com:/home/dahdunn/Jamaican-Style-Dominoes-AI-Neural-Network
scp evaluat1.txt root@clients.axiomarray.com:/home/dahdunn/Jamaican-Style-Dominoes-AI-Neural-Network
python3 play_dominoes.py cutthroat 1000 >> log1.txt
python3 evaluate_model.py >> evaluat1.txt
scp log1.txt root@clients.axiomarray.com:/home/dahdunn/Jamaican-Style-Dominoes-AI-Neural-Network
scp evaluat1.txt root@clients.axiomarray.com:/home/dahdunn/Jamaican-Style-Dominoes-AI-Neural-Network
python3 play_dominoes.py cutthroat 2000 >>log1.txt
python3 evaluate_model.py >> evaluat1.txt
scp log1.txt root@clients.axiomarray.com:/home/dahdunn/Jamaican-Style-Dominoes-AI-Neural-Network
scp evaluat1.txt root@clients.axiomarray.com:/home/dahdunn/Jamaican-Style-Dominoes-AI-Neural-Network
python3 play_dominoes.py cutthroat 3000 >> log1.txt
python3 evaluate_model.py >> evaluat1.txt
scp log1.txt root@clients.axiomarray.com:/home/dahdunn/Jamaican-Style-Dominoes-AI-Neural-Network
scp evaluat1.txt root@clients.axiomarray.com:/home/dahdunn/Jamaican-Style-Dominoes-AI-Neural-Network
python3 play_dominoes.py cutthroat 4000 >>log1.txt
python3 evaluate_model.py >> evaluat1.txt
scp log1.txt root@clients.axiomarray.com:/home/dahdunn/Jamaican-Style-Dominoes-AI-Neural-Network
scp evaluat1.txt root@clients.axiomarray.com:/home/dahdunn/Jamaican-Style-Dominoes-AI-Neural-Network
python3 play_dominoes.py cutthroat 5000 >> log1.txt
python3 evaluate_model.py >> evaluat1.txt
scp log1.txt root@clients.axiomarray.com:/home/dahdunn/Jamaican-Style-Dominoes-AI-Neural-Network
scp evaluat1.txt root@clients.axiomarray.com:/home/dahdunn/Jamaican-Style-Dominoes-AI-Neural-Network
python3 play_dominoes.py cutthroat 6000 >>log1.txt
python3 evaluate_model.py >> evaluat1.txt
scp log1.txt root@clients.axiomarray.com:/home/dahdunn/Jamaican-Style-Dominoes-AI-Neural-Network
scp evaluat1.txt root@clients.axiomarray.com:/home/dahdunn/Jamaican-Style-Dominoes-AI-Neural-Network