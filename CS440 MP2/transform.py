
# transform.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
# 
# Created by Jongdeog Lee (jlee700@illinois.edu) on 09/12/2018

"""
This file contains the transform function that converts the robot arm map
to the maze.
"""
import copy
from arm import Arm
from maze import Maze
from search import *
from geometry import *
from const import *
from util import *

def transformToMaze(arm, goals, obstacles, window, granularity):
    """This function transforms the given 2D map to the maze in MP1.
    
        Args:
            arm (Arm): arm instance
            goals (list): [(x, y, r)] of goals
            obstacles (list): [(x, y, r)] of obstacles
            window (tuple): (width, height) of the window
            granularity (int): unit of increasing/decreasing degree for angles

        Return:
            Maze: the maze instance generated based on input arguments.

    """
    initial_alpha = arm.getArmAngle()[0]
    initial_beta = arm.getArmAngle()[1]
    alpha_limit = arm.getArmLimit()[0]
    beta_limit = arm.getArmLimit()[1]
    rows = int((alpha_limit[1]-alpha_limit[0])/(granularity) + 1)
    columns = int((beta_limit[1]-beta_limit[0])/(granularity) + 1)
    maze_Map = [[SPACE_CHAR for x in range(columns)] for y in range(rows)]
    alpha = alpha_limit[0]
    alpha_index = 0
    while alpha <= alpha_limit[1]:
        beta_index = 0
        beta = beta_limit[0]
        beta_flag = False
        while beta <= beta_limit[1]:
            if beta_flag:                                           #Optimization
                maze_Map[alpha_index][beta_index] = WALL_CHAR
                beta += granularity
                beta_index += 1
                continue
            arm.setArmAngle((alpha, beta))
            arm_tick = arm.getArmPos()[-1][-1]
            if alpha == initial_alpha and beta == initial_beta:
                maze_Map[alpha_index][beta_index] = START_CHAR
            elif doesArmTouchObstacles(arm.getArmPos()[:-1], obstacles):
                beta_flag = True                                    #Triggers optimization
                maze_Map[alpha_index][beta_index] = WALL_CHAR
            elif doesArmTouchObstacles(arm.getArmPos(), obstacles):
                maze_Map[alpha_index][beta_index] = WALL_CHAR
            elif doesArmTouchGoals(arm_tick, goals):
                maze_Map[alpha_index][beta_index] = OBJECTIVE_CHAR
            else:
                maze_Map[alpha_index][beta_index] = SPACE_CHAR
            beta += granularity
            beta_index += 1
        alpha += granularity
        alpha_index += 1
    return Maze(maze_Map, [alpha_limit[0], beta_limit[0]], granularity)