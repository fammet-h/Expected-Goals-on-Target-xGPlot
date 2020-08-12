import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

def futsal_pitch_h(ax):
    """
    BEFORE
    fig = plt.figure(figsize=(10, 6))
    ax = plt.axes()

    AFTER
    plt.show()
    """
    center_circle = mpatches.Circle(xy=(0, 0), radius=3, ec="k", fill=False)
    dot1 = mpatches.Circle(xy=(-14, 0), radius=0.1, ec="k", fill=True)
    dot2 = mpatches.Circle(xy=(-10, 0), radius=0.1, ec="k", fill=True)
    dot3 = mpatches.Circle(xy=(0, 0), radius=0.1, ec="k", fill=True)
    dot4 = mpatches.Circle(xy=(10, 0), radius=0.1, ec="k", fill=True)
    dot5 = mpatches.Circle(xy=(14, 0), radius=0.1, ec="k", fill=True)
    pitch = mpatches.Rectangle(xy=(-20, -12), width=40, height=24, ec="k", fill=False)
    half = mpatches.Rectangle(xy=(0, -12), width=0, height=24, ec="k", fill=False)
    goal1 = mpatches.Rectangle(xy=(-21, -1.5), width=1, height=3, ec="k", fill=False)
    goal2 = mpatches.Rectangle(xy=(20, -1.5), width=1, height=3, ec="k", fill=False)
    goal_area1 = mpatches.Arc(xy=(-20, 1.5), width=12, height=12, theta1=0, theta2=90, ec="k")
    goal_area2 = mpatches.Arc(xy=(-20, -1.5), width=12, height=12, theta1=-90, theta2=0, ec="k")
    goal_area3 = mpatches.Rectangle(xy=(-14, -1.5), width=0, height=3, ec="k")
    goal_area4 = mpatches.Arc(xy=(20, 1.5), width=12, height=12, theta1=90, theta2=180, ec="k")
    goal_area5 = mpatches.Arc(xy=(20, -1.5), width=12, height=12, theta1=180, theta2=270, ec="k")
    goal_area6 = mpatches.Rectangle(xy=(13.9, -1.5), width=0, height=3, ec="k")
    sub1 = mpatches.Rectangle(xy=(-10, -12.5), width=0, height=1, ec="k")
    sub2 = mpatches.Rectangle(xy=(-5, -12.5), width=0, height=1, ec="k")
    sub3 = mpatches.Rectangle(xy=(5, -12.5), width=0, height=1, ec="k")
    sub4 = mpatches.Rectangle(xy=(10, -12.5), width=0, height=1, ec="k")
    ax.add_patch(center_circle)
    ax.add_patch(dot1)
    ax.add_patch(dot2)
    ax.add_patch(dot3)
    ax.add_patch(dot4)
    ax.add_patch(dot5)
    ax.add_patch(pitch)
    ax.add_patch(half)
    ax.add_patch(goal1)
    ax.add_patch(goal2)
    ax.add_patch(goal_area1)
    ax.add_patch(goal_area2)
    ax.add_patch(goal_area3)
    ax.add_patch(goal_area4)
    ax.add_patch(goal_area5)
    ax.add_patch(goal_area6)
    ax.add_patch(sub1)
    ax.add_patch(sub2)
    ax.add_patch(sub3)
    ax.add_patch(sub4)
    ax.xaxis.set_major_formatter(mpl.ticker.NullFormatter())
    ax.yaxis.set_major_formatter(mpl.ticker.NullFormatter())
    plt.axis("scaled")
    ax.set_aspect("equal")

def soccer_pitch_h(ax):
    """
    BEFORE
    fig = plt.figure(figsize=(12, 8))
    ax = plt.axes()

    AFTER
    plt.show()
    """
    center_circle = mpatches.Circle(xy=(0, 0), radius=10, ec="k", fill=False)
    pitch = mpatches.Rectangle(xy=(-60, -40), width=120, height=80, ec="k", fill=False)
    half = mpatches.Rectangle(xy=(0, -40), width=0, height=80, ec="k")
    goal1 = mpatches.Rectangle(xy=(-61, -4), width=1, height=8, ec="k", fill=False)
    goal2 = mpatches.Rectangle(xy=(60, -4), width=1, height=8, ec="k", fill=False)
    six1 = mpatches.Rectangle(xy=(-60, -10), width=6, height=20, ec="k", fill=False)
    six2 = mpatches.Rectangle(xy=(54, -10), width=6, height=20, ec="k", fill=False)
    eighteen1 = mpatches.Rectangle(xy=(-60, -22), width=18, height=44, ec="k", fill=False)
    eighteen2 = mpatches.Rectangle(xy=(42, -22), width=18, height=44, ec="k", fill=False)
    ax.add_patch(center_circle)
    ax.add_patch(pitch)
    ax.add_patch(half)
    ax.add_patch(goal1)
    ax.add_patch(goal2)
    ax.add_patch(six1)
    ax.add_patch(six2)
    ax.add_patch(eighteen1)
    ax.add_patch(eighteen2)
    plt.axis("scaled")
    ax.set_aspect("equal")
    
def soccer_pitch_v(ax):
    """
    BEFORE
    fig = plt.figure(figsize=(8, 12))
    ax = plt.axes()

    AFTER
    plt.show()
    """
    blank = mpatches.Rectangle(xy=(40, -60), width=40, height=120, ec="w", fill=False)
    center_circle = mpatches.Circle(xy=(0, 0), radius=10, ec="k", fill=False)
    pitch = mpatches.Rectangle(xy=(-40, -60), width=80, height=120, ec="k", fill=False)
    half = mpatches.Rectangle(xy=(-40, 0), width=80, height=0, ec="k")
    goal1 = mpatches.Rectangle(xy=(-4, -61), width=8, height=1, ec="k", fill=False)
    goal2 = mpatches.Rectangle(xy=(-4, 60), width=8, height=1, ec="k", fill=False)
    six1 = mpatches.Rectangle(xy=(-10, -60), width=20, height=6, ec="k", fill=False)
    six2 = mpatches.Rectangle(xy=(-10, 54), width=20, height=6, ec="k", fill=False)
    eighteen1 = mpatches.Rectangle(xy=(-22, -60), width=44, height=18, ec="k", fill=False)
    eighteen2 = mpatches.Rectangle(xy=(-22, 42), width=44, height=18, ec="k", fill=False)
    ax.add_patch(blank)
    ax.add_patch(center_circle)
    ax.add_patch(pitch)
    ax.add_patch(half)
    ax.add_patch(goal1)
    ax.add_patch(goal2)
    ax.add_patch(six1)
    ax.add_patch(six2)
    ax.add_patch(eighteen1)
    ax.add_patch(eighteen2)
    plt.axis("scaled")
    ax.set_aspect("equal")

def arrow(ax, start_x, start_y, end_x, end_y, arrow_width=1, color='gray'):
    """
    BEFORE
    fig = plt.figure(figsize=(8, 12))
    ax = plt.axes()

    AFTER
    plt.show()
    """
    ax.annotate('', xy=[end_x, end_y], xytext=[start_x, start_y], arrowprops=dict(shrink=0, width=arrow_width, headwidth=12*arrow_width, headlength=15*arrow_width, connectionstyle='arc3', facecolor=color, edgecolor=color))

def xGonT_circle(ax, x, y, xG, xGonT, goal=False, home=True):
    x = x - 60 if home else 60 - x
    y = 40 - y if home else y - 40
    edgecolor = 'red' if goal else 'black'
    if np.isnan(xGonT):
        shot = mpatches.Circle(xy=(x, y), radius=np.sqrt(xG * 10), edgecolor=edgecolor, fill=False)
    else:
        facecolor = (xGonT, xGonT, 1 - xGonT, 0.5)
        shot = mpatches.Circle(xy=(x, y), radius=np.sqrt(xG * 10), edgecolor=edgecolor, facecolor=facecolor)
    ax.add_patch(shot)
