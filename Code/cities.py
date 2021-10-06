import matplotlib.pyplot as plt
import numpy as np
import turtle

from data_visualization import *
from helper_functions import *

#Class for tracking positions and paths of scooters 
class Scooter_Tracker:
    def __init__(self, initial_positions, velocity):
        self.initial_pos_list = self.get_init_pos(initial_positions)
        self.positions = initial_positions
        self.number_of_scooters = initial_positions.sum()
        self.in_use = []
        self.velocity = velocity

    def get_init_pos(self, initial_positions):
        initial_pos_list = []
        it = np.nditer(initial_positions, flags=['multi_index'])
        for x in it:
            for _ in range(x):
                initial_pos_list.append(it.multi_index)
        return initial_pos_list


    def move_scooter(self, start, goal, t):
        pos = start
        self.positions[tuple(start)] -= 1
        path = [tuple(start)]
        vert_diff = goal[0] - pos[0]
        hor_diff = goal[1] - pos[1]
        end_time = int(t + (np.abs(vert_diff) + np.abs(hor_diff)) / self.velocity)
        assert end_time >= t
        
        while vert_diff != 0 or hor_diff != 0:
            if ((np.random.uniform(0,1)>0.5) and (vert_diff != 0)) or (hor_diff == 0):
                pos[0] += np.sign(vert_diff)
            else:
                pos[1] += np.sign(hor_diff)

            path.append(tuple(pos))
            vert_diff = goal[0] - pos[0]
            hor_diff = goal[1] - pos[1]

        if self.in_use:
            idx = 0
            l = len(self.in_use)
            while idx<=l-1:
                if self.in_use[-1 - idx][0] < end_time:
                    break
                else:
                    idx += 1
            self.in_use.insert(l-idx, (end_time, goal))
        else:
            self.in_use.append((end_time, goal))
        
        return path

    def update_positions(self, t):
        while True:
            if len(self.in_use) == 0:
                break
            else:
                if self.in_use[0][0] == t:
                    (_, goal) = self.in_use.pop(0)
                    self.positions[tuple(goal)] += 1
                else:
                    break
                

#City mode up of a grid of nodes, each containing a population
class City:
    def __init__(self, size, base_pop, time_demand, multimodal=False):
        self.size = size
        self.mid = (size[0]//2, size[1]//2)
        self.city_radius = np.sqrt(self.mid[0]**2 + self.mid[1]**2)
        self.population = self.sample_population(size, base_pop)
        self.time_demand = time_demand
        self.multimodal =  multimodal

    def sample_population(self, size, base_pop):
        population = np.zeros(size, dtype=np.int32)
        y, x = size
        
        for i in range(y):
            for j in range(x):
                distance = np.sqrt((self.mid[0] - i)**2 + (self.mid[1] - j)**2)
                mean = base_pop * (1 - (0.8*distance) / self.city_radius)
                stddev = mean / 5
                population[i,j] = int(np.max([0, np.random.normal(loc=mean, scale=stddev)]))
        return population

    def sample_requests(self, scaling, demand_matrix):
        return np.random.poisson(lam=scaling * (self.population / (500*24*60)) * demand_matrix)

    def sample_goal(self, multimodal=False, centres=[(0,0), (20,20), (22, -22)]):
        p = np.random.uniform()
        if not multimodal:
            scal = (self.city_radius/3)**2
            normal_sample = np.random.multivariate_normal(mean=np.array(self.mid), cov=scal*np.eye(2))
        else:
            if p > 0.25:
                scal = (self.city_radius/3)**2
                mean = np.array(self.mid) + np.array(centres[0])
            elif p > 0.1:
                scal = (self.city_radius/8)**2
                mean = np.array(self.mid) + np.array(centres[1])
            else:
                scal = (self.city_radius/16)**2
                mean = np.array(self.mid) + np.array(centres[2])
            
            normal_sample = np.random.multivariate_normal(mean=mean, cov=scal*np.eye(2))
        
        return np.minimum(np.maximum(np.round_(normal_sample), np.zeros(2)), np.array([self.size])-1).reshape(-1).astype(int)


    def simulate_day(self, initial_positions, start_time):
        #Data that we will collect
        data = {}
        unfullfilled_requests = np.zeros_like(self.population)
        avg_not_in_use = np.zeros_like(self.population).astype(np.float64)
        paths = []
        scooters_in_use = []
        demand_matrix = np.ones_like(self.population)
        if self.multimodal:
            centre1 = np.array(self.mid) + np.array([20,20])
            centre2 = np.array(self.mid) + np.array([22,-22])
            demand_it = np.nditer(demand_matrix, flags=['multi_index'], op_flags=['readwrite'])
            for x in demand_it:
                if ((np.array(demand_it.multi_index) - centre1)**2).sum() <= 2:
                    x[...] = 3
        #Begin Tracking
        tracker = Scooter_Tracker(initial_positions, 1)
        for hour in range(start_time,24):
            for minute in range(60):
                t = hour * 60 + minute
                requests = self.sample_requests(self.time_demand[hour], demand_matrix)
                it = np.nditer(requests, flags=['multi_index'])
                for x in it:
                    for req in range(x):
                        if tracker.positions[it.multi_index]>0:
                            different = False
                            while not different:
                                goal = self.sample_goal(self.multimodal)
                                different = (goal != np.array(it.multi_index)).any()
                            start = list(it.multi_index)
                            path = tracker.move_scooter(start, goal, t)
                            paths.append((t, path))
                        else:
                            unfullfilled_requests[it.multi_index] += 1
                tracker.update_positions(t)
                scooters_in_use.append(len(tracker.in_use))
                avg_not_in_use += tracker.positions / ((24 - start_time) * 60)

        data['number_of_scooters'] = initial_positions.sum()
        data['paths'] = paths
        data['initial_positions'] = tracker.initial_pos_list
        data['unfullfilled_requests'] = unfullfilled_requests
        data['scooters_in_use'] = scooters_in_use
        data['avg_not_in_use'] = avg_not_in_use
        return data

    def visualize_paths(self, init_pos_list, paths, start_time):
        WIDTH = 600
        HEIGHT = 600
        h_dist = WIDTH // (self.size[1] - 1)
        v_dist = HEIGHT // (self.size[0] - 1)

        turtle.tracer(0)
        c = turtle.Turtle()
        c.hideturtle()
        for j in range(self.size[0]):
            c.penup()
            c.goto(-WIDTH//2, HEIGHT//2 - j * v_dist)
            for i in range(self.size[1]):
                c.forward(h_dist//4)
                c.pendown()
                c.left(90)
                c.forward(v_dist//4)
                for _ in range(2):
                    c.left(90)
                    c.forward(h_dist//2)
                c.left(90)
                c.forward(h_dist//4)
                if j != self.size[0] - 1:
                    c.right(90)
                    c.forward(v_dist//2)
                    c.penup()
                    c.left(180)
                    c.forward(v_dist//2)
                    c.right(90)
                    c.pendown()
                c.forward(h_dist//4)
                c.left(90)
                c.forward(h_dist//4)
                c.right(90)
                if i != self.size[1] - 1:
                    c.forward(h_dist//2)
                    c.penup()
                    c.forward(h_dist//4)
        
        all_pos = [(x,y) for x in range(size[1]) for y in range(size[0])]
        all_pos_re = [(h_dist * (x - self.size[1]//2), v_dist * (y - self.size[0]//2)) for (x,y) in all_pos]
        scooters = {pos : [] for pos in all_pos_re}
        scooter_positions = [(h_dist * (x - self.size[1]//2), v_dist * (y - self.size[0]//2))
         for (x,y) in init_pos_list]
        scooter_speed = 2
        for pos in scooter_positions:
            scooter = turtle.Turtle()
            scooter.shape('circle')
            scooter.color('blue')
            scooter.turtlesize(0.75,0.75)
            scooter.speed(scooter_speed)
            scooter.penup()
            scooter.goto(pos)
            scooters[pos].append(scooter)
         
        invisible = turtle.Turtle()
        invisible.speed(scooter_speed+4)
        invisible.penup()
        invisible.goto(400, -200)
        invisible.ht()

        clock = turtle.Turtle()
        clock.speed('fastest')
        clock.penup()
        clock.goto(400, 0)
        clock.ht()
        
        turtle.update()
        turtle.tracer(1)

        def f():
            moving = []
            for hour in range(start_time,24):
                for minute in range(60):
                    turtle.tracer(0)
                    clock.clear()
                    hour_str = f"0{hour}" if hour < 10 else str(hour)
                    minute_str = f"0{minute}" if minute < 10 else str(minute)
                    clock.write(f"{hour_str}:{minute_str}", font=("Arial", 40, "normal"))
                    turtle.update()
                    t = 60 * hour + minute
                    while True:
                        if len(paths)==0:
                            break
                        else:
                            if t == paths[0][0]:
                                _, path = paths.pop(0)
                                rescaled = [(h_dist * (x - self.size[1]//2), v_dist * (y - self.size[0]//2)) for (x,y) in path]
                                scooter = scooters[rescaled[0]].pop(0)
                                scooter.color('red')
                                moving.append((scooter, rescaled[1:]))
                            else:
                                break
                    to_remove = 0
                    if moving: 
                        for (scooter, rescaled_path) in moving:
                            next_point = rescaled_path.pop(0)
                            scooter.goto(next_point)
                            to_remove += (len(rescaled_path) == 0)
                        turtle.update()         
                        if to_remove > 0:
                            for (s,r) in moving:
                                if len(r)==0:
                                    s.color('blue') 
                                    scooters[(int(s.pos()[0]),int(s.pos()[1]))].append(s)
                            moving = [(s, r) for (s, r) in moving if not len(r)==0]
                        for (s,r) in moving:
                            assert r
                    else:
                        invisible.forward(h_dist)
                        invisible.left(180)
        turtle.listen()
        turtle.onkey(f, 'space')
        turtle.done()



if __name__ == "__main__":
    np.random.seed(0)
    size = (31,31)
    base_pop = 1500
    time_demand = create_time_demand()
    start_time = 6
    city = City(size, base_pop, time_demand, multimodal=False)

    number_of_scooters = 50
    radius_scal = 1/3
    initial_positions = gaussian_inital_positions(city, number_of_scooters, radius_scal=radius_scal)
    data = city.simulate_day(initial_positions, start_time)

    city.visualize_paths(data['initial_positions'], data['paths'], start_time)
