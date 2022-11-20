#!/bin/env python
#
#    putil.py
#    Matplot plotting utilities.
#    $Id: putil.py,v 1.8 2018/03/07 12:48:13 daichi Exp $
#
import matplotlib.pyplot as plt

# set aspect ratio.
def aspect_ratio (r):
    plt.gca().set_aspect (r)

# axes lie on zeros.
def zero_origin():
    ax = plt.gca().axes
    ax.spines['left'].set_position('zero')
    ax.spines['bottom'].set_position('zero')
    
# leave only left and bottom axis.
# eg: putil.simpleaxis()
def simpleaxis():
    ax = plt.gca().axes
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

# one dimensional plot.
def one_dimensional():
    axis = plt.gca().axes
    axis.spines['left'].set_visible(False)
    axis.tick_params (
        left=False,
        labelleft=False
    )

# add 'x' and 'y'.    
def add_xy ():
    ax = plt.gca().axes
    xmax = ax.get_xlim()[1]
    ymax = ax.get_ylim()[1]
    ax.text(0,xmax+0.1,r'$y$',ha='center')
    ax.text(ymax+0.1,0,r'$x$',va='center')

def yticklabels(s):
    plt.gca().set_yticklabels(s)

def savefig (file):
    plt.savefig (file, bbox_inches='tight')
