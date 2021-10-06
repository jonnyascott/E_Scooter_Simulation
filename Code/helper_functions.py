import numpy as np

#How scooter demand scales with time of day
def create_time_demand():
    time_demand = {}
    for hour in range(24):
        if hour <= 5 :
            time_demand[hour] = 0.05
        elif hour <=7:
            time_demand[hour] = 0.5
        elif hour<=9:
            time_demand[hour] = 4
        elif hour<=15:
            time_demand[hour] = 1
        elif hour<=18:
            time_demand[hour] = 3
        elif hour<=20:
            time_demand[hour] = 2
        elif hour<=22:
            time_demand[hour] = 0.5
        else:
            time_demand[hour] = 0.25
    return time_demand


def gaussian_inital_positions(city, number_of_scooters, radius_scal = 1/3):
    initial_positions = np.zeros(city.size, dtype=int)
    for _ in range(number_of_scooters):
        scal = (city.city_radius * radius_scal)**2
        normal_sample = np.random.multivariate_normal(mean=city.mid, cov=scal*np.eye(2))
        pt = tuple(np.minimum(np.maximum(np.round_(normal_sample), np.zeros(2)), np.array([city.size])-1).reshape(-1).astype(int))
        initial_positions[pt] += 1
    return initial_positions

def multivariate_gaussian_inital_positions(city, number_of_scooters, radius_scal = 1/3, centres=[(0,0), (20,20), (22, -22)]):
    initial_positions = np.zeros(city.size, dtype=int)
    for _ in range(number_of_scooters):
        p = np.random.uniform()
        if p > 0.25:
            scal = (city.city_radius * radius_scal)**2
            mean = np.array(city.mid) + np.array(centres[0])
        elif p > 0.1:
            scal = (city.city_radius * radius_scal * (3/8))**2
            mean = np.array(city.mid) + np.array(centres[1])
        else:
            scal = (city.city_radius * radius_scal * (3/16))**2
            mean = np.array(city.mid) + np.array(centres[2])
        
        normal_sample = np.random.multivariate_normal(mean=mean, cov=scal*np.eye(2))
        pt = tuple(np.minimum(np.maximum(np.round_(normal_sample), np.zeros(2)), np.array([city.size])-1).reshape(-1).astype(int))
        initial_positions[pt] += 1
    return initial_positions


    for _ in range(number_of_scooters):
        scal = (city.city_radius * radius_scal)**2
        normal_sample = np.random.multivariate_normal(mean=city.mid, cov=scal*np.eye(2))
        
    return initial_positions

#Cost-Profit model for scooter operations
def compute_profit(data):
    number_of_scooters = data['number_of_scooters']
    paths = data['paths']
    #Fixed Costs
    depr = 600 / 365
    fixed_costs = number_of_scooters * depr
    #Variable Costs
    scooter_range = 30
    street_segment_length = 0.3
    cost_to_charge = 0.1
    variable_costs = 0
    for _, path in paths:
        variable_costs += (street_segment_length / scooter_range) * cost_to_charge * len(path)
    #Sales
    hire_price = 1
    per_min_price = 0.19
    sales = 0
    for _, path in paths:
        sales += hire_price
        sales += per_min_price * len(path)

    return fixed_costs, variable_costs, sales





        
