# geometry.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
# 
# Created by Jongdeog Lee (jlee700@illinois.edu) on 09/12/2018

"""
This file contains geometry functions that relate with Part1 in MP2.
"""

import math
import numpy as np
from const import *


def computeCoordinate(start, length, angle):
    """Compute the end cooridinate based on the given start position, length and angle.
        This is an easy calculation using some trigonometry

        Args:
            start (tuple): base of the arm link. (x-coordinate, y-coordinate)
            length (int): length of the arm link
            angle (int): degree of the arm link from x-axis to couter-clockwise

        Return:
            End position of the arm link, (x-coordinate, y-coordinate)
    """
    x = start[0]
    y = start[1]
    new_x = x + length * math.cos(math.radians(angle))
    new_y = y - length * math.sin(math.radians(angle))
    return new_x, new_y


def doesArmTouchObstacles(armPos, obstacles):
    """Determine whether the given arm links touch obstacles.
        We use vector algebra to figure out the distance between a circle and a line segment.

        Args:
            armPos (list): start and end position of all arm links [(start, end)]
            obstacles (list): x-, y- coordinate and radius of obstacles [(x, y, r)]

        Return:
            True if touched. False it not.
    """
    for each_arm in armPos:
        p1 = np.array(each_arm[0])
        p2 = np.array(each_arm[1])
        v = p2 - p1
        for each_obstacle in obstacles:
            Q = np.array([each_obstacle[0], each_obstacle[1]])
            r = each_obstacle[2]
            a = np.dot(v, v)
            b = 2 * np.dot(v, p1 - Q)
            c = np.dot(p1, p1) + np.dot(Q, Q) - 2 * np.dot(p1, Q) - r ** 2
            disc = b ** 2 - 4 * a * c
            if disc < 0:
                continue
            sqrt_disc = math.sqrt(disc)
            t1 = (-b + sqrt_disc) / (2 * a)
            t2 = (-b - sqrt_disc) / (2 * a)
            if 0 <= t1 <= 1 or 0 <= t2 <= 1:
                return True
    return False


def doesArmTouchGoals(armEnd, goals):
    """Determine whether the given arm links touch goals
        This is straightforward distance formula

        Args:
            armEnd (tuple): the arm tick position, (x-coordinate, y-coordinate)
            goals (list): x-, y- coordinate and radius of goals [(x, y, r)]

        Return:
            True if touched. False it not.
    """
    x_a = armEnd[0]
    y_a = armEnd[1]
    for goal in goals:
        x_g = goal[0]
        y_g = goal[1]
        radius = goal[2]
        d = math.sqrt(math.pow(x_a-x_g, 2) + math.pow(y_a-y_g, 2))
        if d <= radius:
            return True
    return False


def isArmWithinWindow(armPos, window):
    """Determine whether the given arm stays in the window

        Args:
            armPos (list): start and end position of all arm links [(start, end)]
            window (tuple): (width, height) of the window

        Return:
            True if all parts are in the window. False it not.
    """
    for arm in armPos:
        x_a = arm[0][0]
        y_a = arm[0][1]
        x_b = arm[1][0]
        y_b = arm[1][1]
        window_x = window[0]
        window_y = window[1]
        if (x_a > window_x or x_a < 0) or (y_a > window_y or y_a < 0) or (x_b > window_x or x_b < 0) or (y_b > window_y or y_b < 0):
            return False
    return True
