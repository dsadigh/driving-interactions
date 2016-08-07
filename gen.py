import pickle
import subprocess
from pylab import *
import csv

dt = 0.1

def ls(pattern):
    output = subprocess.check_output("ls {}".format(pattern), shell=True).splitlines()
    return output

def load(filename):
    with open(filename) as f:
        ret = pickle.load(f)
    u, x = ret
    uh, ur = u
    xh, xr = x
    t = arange(len(xh))*dt
    return {'uh': asarray(uh), 'ur': asarray(ur), 'xh': asarray(xh), 'xr': asarray(xr), 't': t}

def isempty(data):
    return all([len(y)==0 for y in data.values()])

csvs = [[], [], []]

condition = {
    'world0': 'gray',
    'world1': 'orange',
    'world2': 'orange',
    'world3': 'blue',
    'world4': 'orange',
    'world5': 'gray'
}
scenario = {
    'world0': 1,
    'world1': 1,
    'world2': 2,
    'world3': 2,
    'world4': 3,
    'world5': 3
}

for world in ['world{}'.format(i) for i in range(6)]:
    files = ls('saved_data/**/{}*'.format(world))
    for filename in files:
        user = int(filename.split('/')[1].split('-')[0][1:])
        traj = int(filename.split('-')[-1].split('.')[0])
        data = load(filename)
        if isempty(data):
            continue
        for uh, ur, xh, xr, t in zip(data['uh'], data['ur'], data['xh'], data['xr'], data['t']):
            point = {
                'user': user,
                'trajectory': traj,
                'scenario': scenario[world],
                'condition': condition[world],
                'time': t,
                'human_x': xh[0],
                'human_y': xh[1],
                'human_heading': xh[2],
                'human_speed': xh[3],
                'human_steer': uh[0],
                'human_acceleration': uh[1],
                'robot_x': xh[0],
                'robot_y': xh[1],
                'robot_heading': xh[2],
                'robot_speed': xh[3],
                'robot_steer': uh[0],
                'robot_acceleration': uh[1],
            }
            csvs[scenario[world]-1].append(point)

for i, rows in enumerate(csvs):
    with open('csvs/scenario{}.csv'.format(i+1), 'w') as f:
        writer = csv.DictWriter(f, fieldnames = [
            'user',
            'trajectory',
            'scenario',
            'condition',
            'time',
            'human_x',
            'human_y',
            'human_heading',
            'human_speed',
            'human_steer',
            'human_acceleration',
            'robot_x',
            'robot_y',
            'robot_heading',
            'robot_speed',
            'robot_steer',
            'robot_acceleration',
        ])
        writer.writeheader()
        writer.writerows(rows)

opts = ls('data/world*-opt.pickle')
for opt in opts:
    world = opt.split('/')[1].split('-')[0]
    with open('csvs/opt{}-{}.csv'.format(scenario[world], condition[world]), 'w') as f:
        writer = csv.DictWriter(f, fieldnames = [
            'scenario',
            'condition',
            'time',
            'human_x',
            'human_y',
            'human_heading',
            'human_speed',
            'human_steer',
            'human_acceleration',
            'robot_x',
            'robot_y',
            'robot_heading',
            'robot_speed',
            'robot_steer',
            'robot_acceleration',
        ])
        writer.writeheader()
        data = load(opt)
        for uh, ur, xh, xr, t in zip(data['uh'], data['ur'], data['xh'], data['xr'], data['t']):
            point = {
                'scenario': scenario[world],
                'condition': condition[world],
                'time': t,
                'human_x': xh[0],
                'human_y': xh[1],
                'human_heading': xh[2],
                'human_speed': xh[3],
                'human_steer': uh[0],
                'human_acceleration': uh[1],
                'robot_x': xh[0],
                'robot_y': xh[1],
                'robot_heading': xh[2],
                'robot_speed': xh[3],
                'robot_steer': uh[0],
                'robot_acceleration': uh[1],
            }
            writer.writerow(point)
