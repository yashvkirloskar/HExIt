# Code for UI taken from https://github.com/royalbird/Hex-Game
# UI built by Raman Gupta and Rohan Gulati from Netaji Subhas Institute Of Technology
# All game rule implementations built from scratch

# Icon made by Freepik from www.flaticon.com

from __future__ import print_function
import Tkinter as tk
from Tkinter import *
import array
from sys import stdout
from collections import namedtuple
from math import *
from time import sleep
from Hex import *

# Introduction to the game-------------------------------------------------------------------------------------------------------------
"""This is a two player Hex Game. In this game the player has to build a bridge from his side to his other side of 
the hex paralellogram, players take turn alternatively,the one who builds first wins.  A player can place his stone at 
any empty place.As soon as an unbroken chain of stones connects two opposite sides of the board, the game ends 
declaring the winner of the game. This game was invented by Piet Hein. It is a win or lose game  proved by
John Nash an independent inventor of this game.
"""


GRID_SIZE = 7
IMG_SIZE = 35
XPAD = 40
YPAD = 40
WIN_HEIGHT = 2 * YPAD + GRID_SIZE * IMG_SIZE + 100
WIN_WIDTH = 2 * XPAD + (3 * GRID_SIZE - 1) * IMG_SIZE

class gameGrid():
    def __init__(self, frame):
        self.frame = frame
        # self.white = PhotoImage(file="white35.gif")
        # self.red = PhotoImage(file="red35.gif")
        # self.blue = PhotoImage(file="blue35.gif")
        self.white = PhotoImage(file="blue_hex.gif")
        self.red = PhotoImage(file="black_hex.gif")
        self.blue = PhotoImage(file="white_hex.gif")
        self.drawGrid()
        self.playInfo = Hex(GRID_SIZE)
        self.frame.configure(background="yellow")

    def drawGrid(self):
        for yi in range(0, GRID_SIZE):
            xi = XPAD + yi * IMG_SIZE
            for i in range(0, GRID_SIZE):
                l = Label(self.frame, image=self.white)
                l.pack()
                l.image = self.white
                l.place(anchor=NW, x=xi, y=YPAD + yi * IMG_SIZE)
                l.bind('<Button-1>', lambda e: self.on_click(e))
                xi += IMG_SIZE

    def getCoordinates(self, widget):
        row = (widget.winfo_y() - YPAD) / IMG_SIZE
        col = (widget.winfo_x() - XPAD - row * IMG_SIZE) / (IMG_SIZE)
        return row, col

    def toggleColor(self, widget):
        if self.playInfo.turn == 1:
            widget.config(image=self.red)
            widget.image = self.red
        else:
            widget.config(image=self.blue)
            widget.image = self.blue

    def display_winner(self, winner):
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
        if event.widget.image != self.white:
            return
        self.toggleColor(event.widget)
        coord = self.getCoordinates(event.widget)
        self.playInfo.player_move(coord)
        if self.playInfo.winner is not None:
            winner = ""
            if self.playInfo.turn == 0:
                winner = " 1 ( Blue ) "
            else:
                winner += " 2 ( Red ) "
            self.display_winner(winner)
            print ("Click on any button to start another game.")

    def restart(self):
        self.drawGrid()
        self.playInfo = Hex(GRID_SIZE)

class gameWindow:
    def __init__(self, window):
        self.frame = Frame(window, width=WIN_WIDTH, height=WIN_HEIGHT)
        self.frame.pack()
        self.gameGrid = gameGrid(self.frame)


def main():
    print ("Rules:")
    print ("Blue is trying to get from top row to bottom.")
    print ("Red is trying to get from left to right.")
    window = Tk()
    window.wm_title("Hex Game")
    gameWindow(window)
    window.mainloop()


main()