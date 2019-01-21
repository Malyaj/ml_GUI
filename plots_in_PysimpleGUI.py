## Animated Matplotlib Graph
# Use the Canvas Element to create an animated graph.  The code is a bit tricky to follow,
# but if you know Matplotlib then this recipe shouldn't be too difficult to copy and modify

# ![animated matplotlib](https://user-images.githubusercontent.com/13696193/44640937-91b9ea80-a992-11e8-9c1c-85ae74013679.jpg)      

import numpy as np

from tkinter import *
import tkinter as Tk

from random import randint
import PySimpleGUI as sg

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, FigureCanvasAgg
from matplotlib.figure import Figure
import matplotlib.backends.tkagg as tkagg


# create an empty figure
fig = Figure()

ax = fig.add_subplot(111)
ax.set_xlabel("X axis")
ax.set_ylabel("Y axis")
ax.grid()

#ax.set_ylim(-3, 3)
#ax.set_xlim(0, 4 * np.pi)
ls = [x for x in dir(ax)]



# PySimpleGUI layout
layout = [[sg.Text('Animated Matplotlib', size=(40, 1), justification='center', font='Helvetica 20')],
          [sg.Canvas(size=(640, 480), key='__canvas__')],
          [sg.Button('Exit', size=(10, 2), pad=((280, 0), 3), font='Helvetica 14')]]


# create the window and show it without the plot
window = sg.Window('Demo Application - Embedding Matplotlib In PySimpleGUI').Layout(layout)
window.Finalize()  # needed to access the canvas element prior to reading the window

canvas_elem = window.FindElement('__canvas__')

graph = FigureCanvasTkAgg(fig, master=canvas_elem.TKCanvas)
canvas = canvas_elem.TKCanvas

flag = True
first_timeout = 2
subsequent_timeout = 10000

t = first_timeout

# Our event loop
while True:

    
    event, values = window.Read(timeout=t)
    #event, values = window.Read()
    if flag:
        t = subsequent_timeout
        flag = False
    
    if event == 'Exit' or event is None:
        break

    else:
        ax.cla()
        ax.grid()
        ax.set_ylim(-3, 3)
        ax.set_xlim(0, 4 * np.pi)

        x = np.linspace(0, 2 * np.pi , 10)
        y = np.sin(x)
        # plotting here
        print(f"x: {x}")
        
        #ax.plot(x, y, color='red')
        
        #ax.errorbar(x, y)
        ax.step(x, y)
        
        graph.draw()
        figure_x, figure_y, figure_w, figure_h = fig.bbox.bounds
        figure_w, figure_h = int(figure_w), int(figure_h)
        photo = Tk.PhotoImage(master=canvas, width=figure_w, height=figure_h)

        canvas.create_image(640 / 2, 480 / 2, image=photo)

        figure_canvas_agg = FigureCanvasAgg(fig)
        figure_canvas_agg.draw()

        tkagg.blit(photo, figure_canvas_agg.get_renderer()._renderer, colormode=2)

window.Close()
