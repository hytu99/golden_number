# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

# Use `pip install pyswagger requests` to install pyswagger and requests
from pyswagger import App
from pyswagger.contrib.client.requests import Client

import random
import time
import argparse

# Use `pip install numpy pandas` to install numpy and pandas
import numpy as np
import pandas as pd

# Below class QLearningTable is copy from MorvanZhou's tutorials
# https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow/blob/master/contents/2_Q_Learning_maze/RL_brain.py
class QLearningTable:
    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):
        self.actions = actions  # a list
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy
        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)
        
    def choose_action(self, state):
        self.check_state_exist(state)
        # action selection
        if np.random.uniform() < self.epsilon:
            # choose best action
            state_action = self.q_table.loc[state, :]
            # some actions may have the same value, randomly choose on in these actions
            action = np.random.choice(state_action[state_action == np.max(state_action)].index)
        else:
            # choose random action
            action = np.random.choice(self.actions)
        return action

    def learn(self, s, a, r, s_):
        self.check_state_exist(s_)
        q_predict = self.q_table.loc[s, a]
        if s_ != 'terminal':
            q_target = r + self.gamma * self.q_table.loc[s_, :].max()  # next state is not terminal
        else:
            q_target = r  # next state is terminal
        self.q_table.loc[s, a] += self.lr * (q_target - q_predict)  # update

    def check_state_exist(self, state):
        if state not in self.q_table.index:
            # append new state to q table
            self.q_table = self.q_table.append(
                pd.Series(
                    [0]*len(self.actions),
                    index=self.q_table.columns,
                    name=state,
                )
            )


def action1(gArray):
    number = 28
    if len(gArray) != 0:
        number = gArray[-1]
    return number, number

def action2(gArray):
    number = 28
    if len(gArray) != 0:
        number = gArray[-1]*0.618
    return number, number

def action3(gArray):
    number = 28
    if len(gArray) != 0:
        number = np.average(gArray[-5:])
    return number, number

def action4(gArray):
    number = 28
    if len(gArray) != 0:
        number = np.average(gArray[-5:])*0.618
    return number, number

def action5(gArray):
    number = 28
    if len(gArray) != 0:
        number = np.average(gArray[-10:])
    return number, number

def action6(gArray):
    number = 28
    if len(gArray) != 0:
        number = np.average(gArray[-10:])*0.618
    return number, number

def action7(gArray):
    if len(gArray) == 0:
        return 28, 28
    if len(gArray) == 1:
        return gArray[0], gArray[0]
    number = gArray[-1] / gArray[-2] * gArray[-1]
    if number <= 0:
        number = 0.001
    if number >= 100:
        number = 100*0.618
    return number, number

def action8(gArray):
    if len(gArray) == 0:
        return 28, 28
    number1=50
    number2=50/30*0.618+np.average(gArray[-5:])
    return number1, number2

actions=[]
actions.append(action1)
actions.append(action2)
actions.append(action3)
actions.append(action4)
actions.append(action5)
actions.append(action6)
actions.append(action7)
actions.append(action8)

n_actions = len(actions)
RL = QLearningTable(actions=list(range(n_actions)))

def getState(gArray):
    if len(gArray) == 0 or len(gArray) == 1:
        return '0_0'
    else:        
        sub = np.array(gArray[-10:])
        sub1 = sub[:-1]
        sub2 = sub[1:]
        dif = sub1 - sub2
        up = sum(1 for e in dif if e < 0)
        down = sum(1 for e in dif if e > 0)
        return '{}_{}'.format(up, down)

lastState=None
lastAction=None

fixed_action = None

def GeneratePredictionNumbers(goldenNumberList, lastScore, numberCount):
    global lastState
    global lastAction
    
    state = getState(goldenNumberList)
    
    if lastState != None and lastAction != None:
        RL.learn(lastState, lastAction, lastScore, state)
        
    action = RL.choose_action(state)
    number1, number2 = actions[fixed_action](goldenNumberList)
    
    lastState = state
    lastAction = action
    
    return number1, number2

# Init swagger client
host = 'https://goldennumber.aiedu.msra.cn/'
jsonpath = '/swagger/v1/swagger.json'
app = App._create_(host + jsonpath)
client = Client()

def main(roomId):
    if roomId is None:
        # Input the roomid if there is no roomid in args
        roomId = input("Input room id: ")
        try:
            roomId = int(roomId)
        except:
            roomId = 0
            print('Parse room id failed, default join in to room 0')

    userInfoFile = "simplebot{}_userinfo.txt".format(fixed_action)
    userId = None
    nickName = None
    try:
        # Use an exist player
        with open(userInfoFile) as f:
            userId, nickName = f.read().split(',')[:2]
        print('Use an exist player: ' + nickName + '  Id: ' + userId)
    except:
        # Create a new player
        userResp = client.request(
            app.op['NewUser'](
                nickName='Simple Bot ' + str(fixed_action)
            ))
        assert userResp.status == 200
        user = userResp.data
        userId = user.userId
        nickName = user.nickName
        print('Create a new player: ' + nickName + '  Id: ' + userId)

        with open(userInfoFile, "w") as f:
            f.write("%s,%s" % (userId, nickName))

    print('Room id: ' + str(roomId))

    while True:
        stateResp = client.request(
            app.op['State'](
                uid=userId,
                roomid=roomId
            ))
        if stateResp.status != 200:
            print('Network issue, query again after 1 second')
            time.sleep(1)
            continue
        state = stateResp.data
    
        if state.state == 2:
            print('The game has finished')
            break

        if state.state == 1:
            print('The game has not started, query again after 1 second')
            time.sleep(1)
            continue

        if state.hasSubmitted:
            print('Already submitted this round, wait for next round')
            if state.maxUserCount == 0:
                time.sleep(state.leftTime + 1)
            else:
                # One round can be finished when all players submitted their numbers if the room have set the max count of users, need to check the state every second.
                time.sleep(1)
            continue

        print('\r\nThis is round ' + str(state.finishedRoundCount + 1))

        todayGoldenListResp = client.request(
            app.op['TodayGoldenList'](
                roomid=roomId
            ))
        if todayGoldenListResp.status != 200:
            print('Network issue, query again after 1 second')
            time.sleep(1)
            continue
        todayGoldenList = todayGoldenListResp.data
        if len(todayGoldenList.goldenNumberList) != 0:
            print('Last golden number is: ' + str(todayGoldenList.goldenNumberList[-1]))

        lastRoundResp = client.request(
            app.op['History'](
                roomid=roomId,
                count=1
            ))
        if lastRoundResp.status != 200:
            print('Network issue, query again after 1 second')
            time.sleep(1)
            continue
        lastScore = 0
        if len(lastRoundResp.data.rounds) > 0:
            scoreArray = [user for user in lastRoundResp.data.rounds[0].userNumbers if user.userId == userId]
            if len(scoreArray) == 1:
                lastScore = scoreArray[0].score
        print('Last round score: {}'.format(lastScore))

        number1, number2 = GeneratePredictionNumbers(todayGoldenList.goldenNumberList, lastScore, state.numbers)

        if (state.numbers == 2):
            submitRsp = client.request(
                app.op['Submit'](
                    uid=userId,
                    rid=state.roundId,
                    n1=str(number1),
                    n2=str(number2)
                ))
            if submitRsp.status == 200:
                print('You submit numbers: ' + str(number1) + ', ' + str(number2))
            else:
                print('Error: ' + submitRsp.data.message)
                time.sleep(1)

        else:
            submitRsp = client.request(
                app.op['Submit'](
                    uid=userId,
                    rid=state.roundId,
                    n1=str(number1)
                ))
            if submitRsp.status == 200:
                print('You submit number: ' + str(number1))
            else:
                print('Error: ' + submitRsp.data.message)
                time.sleep(1)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--room', type=int, help='Room ID', required=False)
    parser.add_argument('--action', type=int, choices=list(range(n_actions)))
    args = parser.parse_args()
    fixed_action = args.action

    main(args.room)