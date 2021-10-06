import matplotlib.pyplot as plt
import numpy as np
import turtle

from matplotlib.cm import get_cmap

def plot_scooter_usage(in_use, start_time):
    times = [hour + minute/60 for hour in range(start_time, 24) for minute in range(0, 60)]
    plt.bar(np.arange(len(in_use)), np.array(in_use) / number_of_scooters)
    plt.ylabel('Proportion of scooters in use')
    plt.xlabel(f'Time of day [minutes after {start_time}am]')
    plt.show()

def plot_heatmap(size, paths):
    horizontal = np.zeros((size[0], size[1]-1))
    vertical = np.zeros((size[0]-1, size[1]))
    for _, path in paths:
        for start, end in zip(path[:-1], path[1:]):
            if start[0] == end[0]:
                direction = 0 if end[1]>start[1] else -1
                horizontal[(start[0], start[1] + direction)] += 1
            else:
                direction = 0 if end[0]>start[0] else -1
                vertical[(start[0]+ direction, start[1])] += 1
    m = horizontal.max()
    horizontal /= horizontal.max()
    vertical /= vertical.max()

    cmap = get_cmap('Reds')

    WIDTH = 600
    HEIGHT = 600

    h_dist = WIDTH // (size[1] - 1)
    v_dist = HEIGHT // (size[0] - 1)

    turtle.tracer(0)
    c = turtle.Turtle()
    c.hideturtle()
    
    
    for j in range(size[0]):
        c.penup()
        c.goto(-WIDTH//2, HEIGHT//2 - j * v_dist)
        for i in range(size[1]):
            c.forward(h_dist//4)
            c.color('black')
            c.pensize(1)
            c.pendown()
            c.left(90)
            c.forward(v_dist//4)
            for _ in range(2):
                c.left(90)
                c.forward(h_dist//2)
            c.left(90)
            c.forward(h_dist//4)
            if j != size[0] - 1:
                c.right(90)
                c.color(cmap(vertical[j,i])[:-1])
                c.pensize(3)
                c.forward(v_dist//2)
                c.penup()
                c.left(180)
                c.forward(v_dist//2)
                c.right(90)
                c.color('black')
                c.pensize(1)
                c.pendown()
            c.forward(h_dist//4)
            c.left(90)
            c.forward(h_dist//4)
            c.right(90)
            if i != size[1] - 1:
                c.color(cmap(horizontal[j,i])[:-1])
                c.pensize(3)
                c.forward(h_dist//2)
                c.penup()
                c.forward(h_dist//4)

    y_cbar = 200
    c.penup()
    c.goto(400, y_cbar)
    c.pendown()
    c.pensize(4)
    c.seth(270)
    for i in range(11):
        c.color(cmap(1 - i*0.1)[:-1])
        c.forward(y_cbar/5)
    c.penup()
    c.goto(410, y_cbar -20)
    #c.pendown()
    c.color('black')
    for i in range(11):
        c.write(f"{int(m*(1 - i*0.1))}")
        c.forward(y_cbar/5)
    turtle.update()
    turtle.done()