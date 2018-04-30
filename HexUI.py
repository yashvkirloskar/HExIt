# Code for UI taken from https://github.com/royalbird/Hex-Game
# UI built by Raman Gupta and Rohan Gulati from Netaji Subhas Institute Of Technology
# All game rule implementations built from scratch

# Icon made by Freepik from www.flaticon.com

from __future__ import print_function
# For python3/python2 compatability
try:
    import Tkinter as tk
    from Tkinter import *
except ModuleNotFoundError:
    import tkinter as tk
    from tkinter import * 
import array
from sys import stdout
from collections import namedtuple
from math import *
from time import sleep
from VectorHex import *
import numpy as np

# Introduction to the game-------------------------------------------------------------------------------------------------------------
"""This is a two player Hex Game. In this game the player has to build a bridge from his side to his other side of 
the hex paralellogram, players take turn alternatively,the one who builds first wins.  A player can place his stone at 
any empty place.As soon as an unbroken chain of stones connects two opposite sides of the board, the game ends 
declaring the winner of the game. This game was invented by Piet Hein. It is a win or lose game  proved by
John Nash an independent inventor of this game.
"""


GRID_SIZE = 5
IMG_SIZE = 35
XPAD = 40
YPAD = 40
WIN_HEIGHT = 2 * YPAD + GRID_SIZE * IMG_SIZE + 100
WIN_WIDTH = 2 * XPAD + (3 * GRID_SIZE - 1) * IMG_SIZE

class gameGrid():
    def __init__(self, frame):
        self.frame = frame
        self.temp = PhotoImage(file="blue_hex.gif")
        self.black = PhotoImage(file="black_hex.gif")
        self.white = PhotoImage(file="white_hex.gif")
        self.widgets = [[None for i in range(GRID_SIZE)] for j in range(GRID_SIZE)]
        self.drawGrid()
        self.p1 = np.random.choice(['ai'])
        print ("P1 is ", self.p1)
        if self.p1 == 'ai':
            self.p2 = 'random'
        else:
            self.p2 = np.random.choice(['ai', 'human'])
        print ("P2 is " + self.p2)
        if (self.p1 == 'ai'):
            print ("Click a tile to make the ai play")
        self.playInfo = VectorHex(GRID_SIZE, p1=self.p1, p2=self.p2)
        self.frame.configure(background="yellow")

    def aiMove(self):
        print ("Enter aiMove")
        turn, move = self.playInfo.ai_move()
        self.toggleColor(self.widgets[move // GRID_SIZE][move % GRID_SIZE], turn)
        if self.playInfo.winner is not None:
            winner = ""
            if turn == 1:
                winner = " 1 ( White ) "
            else:
                winner += " 2 ( Black ) "
            self.display_winner(winner)
            print ("Click on any button to start another game.")

    def drawGrid(self):
        for yi in range(0, GRID_SIZE):
            xi = XPAD + yi * IMG_SIZE
            for i in range(0, GRID_SIZE):
                l = Label(self.frame, image=self.temp)
                l.pack()
                l.image = self.temp
                l.place(anchor=NW, x=xi, y=YPAD + yi * IMG_SIZE)
                l.bind('<Button-1>', lambda e: self.on_click(e))
                self.widgets[yi][i] = l
                xi += 1.5 * IMG_SIZE

    def getCoordinates(self, widget):
        row = (widget.winfo_y() - YPAD) / IMG_SIZE
        col = (widget.winfo_x() - XPAD - row * IMG_SIZE) / (1.5 * IMG_SIZE)
        return int(row), int(col)

    def toggleColor(self, widget, turn = None):
        if turn is None:    
            if self.playInfo.turn == 1:
                widget.config(image=self.white)
                widget.image = self.white
            else:
                widget.config(image=self.black)
                widget.image = self.black
        else:
            if turn == 1:
                widget.config(image=self.white)
                widget.image = self.white
            else:
                widget.config(image=self.black)
                widget.image = self.black

    def display_winner(self, winner, turn=None):
        winner_window = Tk()
        winner_window.wm_title("Winner")
        frame = Frame(winner_window, width=300, height=40)
        frame.pack()
        label = Label(frame,text = "Winner is Player : " + winner )
        label.pack()
        label.place(anchor=tk.NW, x = 20, y = 20)

    def on_click(self, event):
        if self.playInfo.winner is not None:
            self.restart()

        if self.playInfo.turn == 1 and (self.p1 == 'ai' or self.p1 == 'random'):
            self.aiMove()
            return
        if self.playInfo.turn == 2 and (self.p2 == 'ai' or self.p2 == 'random'):
            self.aiMove()
            return
        if event.widget.image != self.temp:
            return
        self.toggleColor(event.widget)
        coord = self.getCoordinates(event.widget)
        self.playInfo.player_move(coord)
        if self.playInfo.winner is not None:
            winner = ""
            if self.playInfo.turn == 1:
                winner = " 1 ( White ) "
            else:
                winner += " 2 ( Black ) "
            self.display_winner(winner)
            print ("Click on any button to start another game.")

    def restart(self):
        self.drawGrid()
        p1 = np.random.choice(['ai', 'human'])
        self.playInfo = VectorHex(GRID_SIZE, p1=p1, p2='human' if p1 == 'ai' else 'ai')

class gameWindow:
    def __init__(self, window):
        self.frame = Frame(window, width=WIN_WIDTH, height=WIN_HEIGHT)
        self.frame.pack()
        self.gameGrid = gameGrid(self.frame)


def main():
    print ("Rules:")
    print ("White plays first")
    print ("Black is trying to get from top row to bottom.")
    print ("White is trying to get from left to right.")
    window = Tk()
    window.wm_title("Hex Game")
    gameWindow(window)
    window.mainloop()


main()