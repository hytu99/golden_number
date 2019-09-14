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

import torch
import torch.nn as nn


class DQN(nn.Module):

    def __init__(self, state_size, action_size, mid_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_size, mid_dim),
            nn.ReLU(inplace=True),
            nn.Linear(mid_dim, mid_dim),
            nn.ReLU(inplace=True),
            nn.Linear(mid_dim, action_size)
        )

    def forward(self, x):
        return self.net(x)

# Below class QLearningTable is copy from MorvanZhou's tutorials
# https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow/blob/master/contents/2_Q_Learning_maze/RL_brain.py
# class QLearningTable:
#     def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):
#         self.actions = actions  # a list
#         self.lr = learning_rate
#         self.gamma = reward_decay
#         self.epsilon = e_greedy
#         self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)

#     def choose_action(self, state):
#         self.check_state_exist(state)
#         # action selection
#         if np.random.uniform() < self.epsilon:
#             # choose best action
#             state_action = self.q_table.loc[state, :]
#             # some actions may have the same value, randomly choose on in these actions
#             action = np.random.choice(state_action[state_action == np.max(state_action)].index)
#         else:
#             # choose random action
#             action = np.random.choice(self.actions)
#         return action

#     def learn(self, s, a, r, s_):
#         self.check_state_exist(s_)
#         q_predict = self.q_table.loc[s, a]
#         if s_ != 'terminal':
#             q_target = r + self.gamma * self.q_table.loc[s_, :].max()  # next state is not terminal
#         else:
#             q_target = r  # next state is terminal
#         self.q_table.loc[s, a] += self.lr * (q_target - q_predict)  # update

#     def check_state_exist(self, state):
#         if state not in self.q_table.index:
#             # append new state to q table
#             self.q_table = self.q_table.append(
#                 pd.Series(
#                     [0]*len(self.actions),
#                     index=self.q_table.columns,
#                     name=state,
#                 )
#             )


def action1(gArray):
    number = 28
    if len(gArray) != 0:
        number = gArray[-1]
    return number


def action2(gArray):
    number = 28
    if len(gArray) != 0:
        number = gArray[-1]*0.618
    return number


def action3(gArray):
    number = 28
    if len(gArray) != 0:
        number = np.average(gArray[-5:])
    return number


def action4(gArray):
    number = 28
    if len(gArray) != 0:
        number = np.average(gArray[-5:])*0.618
    return number


def action5(gArray):
    number = 28
    if len(gArray) != 0:
        number = np.average(gArray[-10:])
    return number


def action6(gArray):
    number = 28
    if len(gArray) != 0:
        number = np.average(gArray[-10:])*0.618
    return number


def action7(gArray):
    if len(gArray) == 0:
        return 28
    if len(gArray) == 1:
        return gArray[0]
    number = gArray[-1] / gArray[-2] * gArray[-1]
    if number <= 0:
        number = 0.001
    if number >= 100:
        number = 100 * 0.618
    return number


def action8(gArray):
    if len(gArray) == 0:
        return 28, 28
    number1 = 50
    number2 = 50/30*0.618+np.average(gArray[-5:])
    return number1, number2


def generate_action_functions():
    atomic_actions = [action1, action2, action3,
                      action4, action5, action6, action7]

    def combine_action(action_1, action2):
        return lambda arr: (action_1(arr), action_2(arr))

    result = []
    for i, action_1 in enumerate(atomic_actions):
        for j, action_2 in enumerate(atomic_actions[i+1:]):
            result.append(combine_action(action_1, action_2))

    result.append(action8)
    return result


action_functions = generate_action_functions()

n_states = 10
n_actions = len(action_functions)
policy_net = DQN(n_states, n_actions)
target_net = DQN(n_states, n_actions)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

BATCH_SIZE = 16
TARGET_UPDATE = 10
GAMMA = 0.999
EPSILON = 0.2

optimizer = torch.optim.RMSprop(policy_net.parameters(), lr=1e-2)

gnum_list = np.array([], dtype=np.float32)
action_list = np.array([], dtype=np.int)
reward_list = np.array([], dtype=np.float32)


def sample_data():

    end_points = random.sample(
        list(range(n_states, len(gnum_list) - 1)), 
        BATCH_SIZE - 1
    ) + [len(gnum_list) - 1]
    inputs = [
        gnum_list[end - n_states + 1: end + 1]
        for end in end_points
    ]
    inputs = torch.tensor(inputs).float()
    actions = torch.tensor(action_list[end_points]).long()
    rewards = torch.tensor(reward_list[end_points]).float()

    target_inputs = torch.tensor([
        gnum_list[end - n_states + 1: end + 1]
        for end in end_points
    ]).float()
    targets = target_net(target_inputs).max(1)[0].detach() * GAMMA + rewards

    return inputs, actions, targets


def optimize_model():
    if len(gnum_list) < BATCH_SIZE + n_states + 1:
        return

    inputs, actions, targets = sample_data()
    outputs = policy_net(inputs).gather(1, actions.unsqueeze(1))
    loss = nn.functional.smooth_l1_loss(outputs, targets.unsqueeze(1))
    print('Loss: {}'.format(loss.item()))

    optimizer.zero_grad()
    loss.backward()

    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)

    optimizer.step()

    if len(gnum_list) % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())


def generate_predicted_numbers(lastGNum, lastScore, numberCount):

    global gnum_list, reward_list, action_list

    optimize_model()

    if len(action_list) > 0:
        gnum_list = np.append(gnum_list, [lastGNum])
        reward_list = np.append(reward_list, [lastScore])

    if EPSILON > random.uniform(0, 1) or len(gnum_list) < n_states:
        action_taken = random.choice(list(range(n_actions)))
    else:
        with torch.no_grad():
            inputs = torch.from_numpy(gnum_list[-n_states:]).unsqueeze(0).float()
            print(inputs.dtype, policy_net.net[0].weight.dtype)
            action_taken = policy_net(inputs).max(1)[1].item()

    action_list = np.append(action_list, [action_taken])

    print(gnum_list)
    print(reward_list)
    print(action_list)

    return action_functions[action_taken](gnum_list)    


# Init swagger client
host = 'https://goldennumber.aiedu.msra.cn/'
jsonpath = '/swagger/v1/swagger.json'
app = App._create_(host + jsonpath)
client = Client()


def main(roomId, userInfoFile):
    if roomId is None:
        # Input the roomid if there is no roomid in args
        roomId = input("Input room id: ")
        try:
            roomId = int(roomId)
        except:
            roomId = 0
            print('Parse room id failed, default join in to room 0')

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
                nickName='DQN ' + str(random.randint(0, 9999))
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
            last_gnum = todayGoldenList.goldenNumberList[-1]
            print('Last golden number is: ' + str(last_gnum))
        else:
            last_gnum = None

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
            scoreArray = [
                user for user in lastRoundResp.data.rounds[0].userNumbers if user.userId == userId]
            if len(scoreArray) == 1:
                lastScore = scoreArray[0].score
        print('Last round score: {}'.format(lastScore))

        number1, number2 = generate_predicted_numbers(last_gnum, lastScore, state.numbers)

        if (state.numbers == 2):
            submitRsp = client.request(
                app.op['Submit'](
                    uid=userId,
                    rid=state.roundId,
                    n1=str(number1),
                    n2=str(number2)
                ))
            if submitRsp.status == 200:
                print('You submit numbers: ' +
                      str(number1) + ', ' + str(number2))
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
    parser.add_argument('--user-file', default='DQN_userInfo.txt')
    args = parser.parse_args()

    main(args.room, args.user_file)
