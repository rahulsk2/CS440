# pong.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
# 
# Created by Chris Benson (cebenso2@illinois.edu) on 09/18/2018

import random
import pygame

class PongEnv:

    WHITE = (255, 255, 255)
    ORANGE = (255,140,0)
    RED = (255, 0, 0)
    BLUE = (0, 0, 255)
    BLACK = (0, 0, 0)
    GREEN = (0, 255, 0)

    def __init__(self, two_sided):
        self.game = PongGame(two_sided)
        self.render = False
        self.two_sided = two_sided

    def get_actions(self):
        return self.game.get_actions()

    def reset(self):
        return self.game.reset()
    
    def get_bounces(self):
        return self.game.get_bounces()

    def get_state(self):
        return self.game.get_state()[:-1]

    def step(self, action):
        state, bounces, done, won = self.game.step(action)
        if self.render:
            self.draw(state, bounces, done, won)
        return state[:-1], bounces, done, won

    def draw(self, state, bounces, done, won, delay = True):
        ball_x, ball_y, _, _, paddle_y, opponent_y = state
        self.display.fill(self.WHITE)

        if done:
            if won:
                pygame.draw.line(self.display, self.RED, [50, 50], [50, 450], 2)
                pygame.draw.line(self.display, self.GREEN, [450, 50], [450, 450], 2)
            else:
                pygame.draw.line(self.display, self.RED, [450, 50], [450, 450], 2)
                pygame.draw.line(self.display, self.GREEN, [50, 50], [50, 450], 2)
        
            
        if not self.two_sided:
            pygame.draw.line(self.display, self.BLACK, [50, 50], [50, 450], 5)
        else:
            pygame.draw.line(self.display, self.BLACK, [50, int(opponent_y * 400)+50], [50, int((opponent_y+self.game.paddle_height) * 400)+50], 5)

        pygame.draw.line(self.display, self.BLACK, [50, 50], [450, 50], 5)
        pygame.draw.line(self.display, self.BLACK, [50, 450], [450, 450], 5)
        pygame.draw.circle(self.display, self.ORANGE, [int(ball_x * 400) + 50, int(ball_y * 400) + 50], 5)
        pygame.draw.line(self.display, self.BLUE, [450, int(paddle_y * 400)+50], [450, int((paddle_y+self.game.paddle_height) * 400)+50], 5)
        
        text_surface = self.font.render("Bounces: " + str(bounces), True, self.BLACK)
        text_rect = text_surface.get_rect()
        text_rect.center = ((250),(25))
        self.display.blit(text_surface, text_rect)
        pygame.display.flip()
        if done:
            self.clock.tick(1)
        else:
            self.clock.tick(30)

    def display(self):
        pygame.init()
        pygame.display.set_caption('MP7: Pong')
        self.clock = pygame.time.Clock()
        pygame.font.init()

        self.font = pygame.font.Font(pygame.font.get_default_font(), 30)
        self.display = pygame.display.set_mode((500, 500), pygame.HWSURFACE)
        self.draw(self.game.get_state(), self.game.get_bounces(), False, False, delay = False)
        self.render = True
            
class PongGame:

    paddle_height = 0.2
    area_width = 1.0
    area_height = 1.0
    paddle_move_distance = 0.04
    opponent_move_distance = 0.04
    
    def __init__(self, two_sided = False):
        self.two_sided = two_sided
        self.bounces=0
        self.reset()

    def get_actions(self):
        return [-1,0,1]

    def reset(self):
        self.ball_x = 0.5
        self.ball_y = 0.5
        self.velocity_x = 0.03
        self.velocity_y = 0.01
        self.paddle_y = 0.5 - self.paddle_height/2
        if self.two_sided:
            self.opponent_y = 0.5 - self.paddle_height/2 
        else:
            self.opponent_y = None
        self.paddle_x = 1
        self.bounces = 0

    def get_state(self):
        return [
            self.ball_x,
            self.ball_y,
            self.velocity_x,
            self.velocity_y,
            self.paddle_y,
            self.opponent_y,
        ]

    def step(self, action):
        self.move_paddle(action)
        if self.two_sided:
            self.move_opponenet()
        self.move_ball()
        self.handle_bounces()
        return self.get_state(), self.get_bounces(), self.done(), self.won()

    def move_paddle(self,action):
        if action == 1:
            self.paddle_y = min(1-self.paddle_height, self.paddle_y + self.paddle_move_distance)
        elif action == -1:
            self.paddle_y = max(0, self.paddle_y - self.paddle_move_distance)
            
    def move_ball(self):
        self.ball_x += self.velocity_x
        self.ball_y += self.velocity_y

    def move_opponenet(self):
        if self.opponent_y + self.paddle_height/2 < self.ball_y:
            self.opponent_y = min(1-self.paddle_height, self.opponent_y + self.opponent_move_distance)
        elif self.opponent_y + self.paddle_height/2 > self.ball_y:
            self.opponent_y = max(0, self.opponent_y - self.opponent_move_distance)
        
    def handle_bounces(self):
        if self.ball_y < 0:
            self.ball_y *= -1
            self.velocity_y *= -1

        if self.ball_y > 1:
            self.ball_y = 2 - self.ball_y
            self.velocity_y *=-1

        if self.ball_x < 0:
            if not self.two_sided:
                self.ball_x *= -1
                self.velocity_x *= -1
            else:
                y_at_opponent_location = (self.ball_y -
                                         self.velocity_y *(self.ball_x)/self.velocity_x)
                y_at_opponent_location = min(max(0.0, y_at_opponent_location), 1.0)
                if (y_at_opponent_location >= self.opponent_y and 
                    y_at_opponent_location <= self.opponent_y+self.paddle_height):
                    self.ball_x = - self.ball_x
                    self.velocity_x *= -1
                    self.randomize_velocities()
               
        if self.ball_x > self.paddle_x:
            y_at_paddle_location = (self.ball_y -
                                    self.velocity_y *(self.ball_x - self.paddle_x)/self.velocity_x )
            y_at_paddle_location = min(max(0.0, y_at_paddle_location), 1.0)
            if (y_at_paddle_location >= self.paddle_y and 
                y_at_paddle_location <= self.paddle_y+self.paddle_height):
                self.ball_x = 2 * self.paddle_x - self.ball_x
                self.velocity_x *= -1
                self.randomize_velocities()
                self.bounces +=1
        
    def randomize_velocities(self):
        self.velocity_y += random.uniform(-0.03,0.03)
        self.velocity_x += random.uniform(-0.015,0.015)
        if abs(self.velocity_x) < 0.03:
            self.velocity_x = 0.03 * self.velocity_x/abs(self.velocity_x)
        if abs(self.velocity_x) > 1:
            self.velocity_x = 1 * self.velocity_y/abs(self.velocity_y)
        if abs(self.velocity_y) > 1:
            self.velocity_y = 1 * self.velocity_y/abs(self.velocity_y)

            
    def done(self):
        return self.ball_x > self.paddle_x or self.ball_x < 0

    def won(self):
        return self.ball_x < 0
      
    def get_bounces(self):
        return self.bounces
