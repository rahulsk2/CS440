# mp7.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
# 
# Created by Chris Benson (cebenso2@illinois.edu) on 09/18/2018
import pygame
from pygame.locals import *
import argparse

from agent import Agent
from pong import PongEnv as Pong

import utils

class Application:
    def __init__(self, args):
        self.args = args
        self.env = Pong(self.args.opponent)
        self.agent = Agent(self.env.get_actions(), two_sided = self.args.opponent)
        
    def execute(self):
        if not self.args.human:
            if self.args.train_eps != 0:
                self.train()
            self.test()
        self.show_games()

    def train(self):
        print("Train")
        self.agent.train()

        window = self.args.window

        self.bounce_results = []
        self.win_results = []
        
        for game in range(1, self.args.train_eps + 1):
            state = self.env.get_state()
            done, won = False, False
            action = self.agent.act(state, 0, done, won)
            count = 0
            while not done:
                count +=1
                state, bounces, done, won = self.env.step(action)
                action = self.agent.act(state, bounces, done, won)
            bounces = self.env.get_bounces()
            self.bounce_results.append(bounces)
            self.win_results.append(won)
            if game % self.args.window == 0:
                print(
                    "Games:", len(self.bounce_results) - window, "-", len(self.bounce_results), 
                    "Bounces (Average:", sum(self.bounce_results[-window:])/window,
                    "Max:", max(self.bounce_results[-window:]),
                    "Min:", min(self.bounce_results[-window:]),")",
                    "Win Rate:" if self.args.opponent else "",
                    sum(self.win_results[-window:])/window if self.args.opponent else "",
                )
            self.env.reset()
        self.agent.save_model(self.args.model_name)

    def test(self):
        print("Test")
        self.agent.eval()
        self.agent.load_model(self.args.model_name)
        bounce_results = []
        win_results = []

        for game in range(1, self.args.test_eps + 1):
            state = self.env.get_state()
            done, won= False, False
            action = self.agent.act(state, 0, done, won)
            count = 0
            while not done:
                count +=1
                state, bounces, done, won = self.env.step(action)
                action = self.agent.act(state, bounces, done, won)
            bounces = self.env.get_bounces()
            bounce_results.append(bounces)
            win_results.append(won)
            self.env.reset()

        print("Number of Games:", len(bounce_results))
        print("Average Bounces:", sum(bounce_results)/len(bounce_results))
        print("Max Bounces:", max(bounce_results))
        print("Min Bounces:", min(bounce_results))
        if self.args.opponent:
            print("Win Rate:", sum(win_results)/len(win_results))

    def show_games(self):
        print("Display Games")
        self.env.display()
        pygame.event.pump()
        self.agent.eval()
        win_results = []
        bounce_results = []
        end = False
        for game in range(1, self.args.show_eps + 1):
            state = self.env.get_state()
            done, won= False, False
            action = self.agent.act(state, 0, done, won)
            count = 0
            while not done:
                count +=1
                pygame.event.pump()
                keys = pygame.key.get_pressed()
                if keys[K_ESCAPE] or self.check_quit():
                    end = True
                    break
                state, bounces, done, won = self.env.step(action)
                if not self.args.human:
                    action = self.agent.act(state, bounces, done, won)
                else:
                    action = -1 if keys[K_UP] else 1 if keys[K_DOWN] else 0
            if end:
                break
            self.env.reset()
            win_results.append(won)
            bounce_results.append(bounces)
            if self.args.opponent:
                print("Game:", str(game)+"/"+str(self.args.show_eps), "Won!" if won else "Lost")
            else:
                print("Game:", str(game)+"/"+str(self.args.show_eps), "Bounces:", bounces)
        if len(bounce_results) + len(win_results) == 0:
            return
        if self.args.opponent:
            print("Win Rate:", sum(win_results)/ len(win_results))
        else:
            print("Average Bounces:", sum(bounce_results)/len(bounce_results))

    def check_quit(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return True
        return False
                
            
def main():
    parser = argparse.ArgumentParser(description='CS440 MP7 Pong')
    
    parser.add_argument('--model_name', dest="model_name", type=str, default = "q_agent.npy",
                        help='name of model to save if training or to load if evaluating - default q_agent')

    parser.add_argument('--opponent', default = False, action = "store_true",
                        help='flag for having an opponnent agent - default False')

    parser.add_argument('--human', default = False, action = "store_true",
                        help='making the game human playable - default False')

    parser.add_argument('--train_episodes', dest="train_eps", type=int, default = 1000,
                        help='number of training episodes - default 1000')

    parser.add_argument('--test_episodes', dest="test_eps", type=int, default = 1000,
                        help='number of testing episodes - default 1000')

    parser.add_argument('--show_episodes', dest="show_eps", type=int, default = 10,
                        help='number of displayed episodes - default 10')

    parser.add_argument('--window', dest="window", type=int, default = 100,
                        help='number of episodes to keep running stats for during training - default 100')
    args = parser.parse_args()
    app = Application(args)
    app.execute()

if __name__ == "__main__":
    main()
