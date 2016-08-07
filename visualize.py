import pyglet
from pyglet.window import key
import pyglet.gl as gl
import pyglet.graphics as graphics
import numpy as np
import time
import car
import math
import dynamics
import matplotlib.cm
import theano as th
import utils
import feature
import pickle
import sys
from car import Car

class Visualizer(object):
    def __init__(self, dt=0.5, fullscreen=False, name='unnamed', iters=1000, magnify=1.):
        self.autoquit = False
        self.frame = None
        self.subframes = None
        self.visible_cars = []
        self.magnify = magnify
        self.camera_center = None
        self.name = name
        self.output = None
        self.iters = iters
        self.objects = []
        self.event_loop = pyglet.app.EventLoop()
        self.window = pyglet.window.Window(600, 600, fullscreen=fullscreen, caption=name)
        self.grass = pyglet.resource.texture('grass.png')
        self.window.on_draw = self.on_draw
        self.lanes = []
        self.cars = []
        self.dt = dt
        self.anim_x = {}
        self.prev_x = {}
        self.feed_u = None
        self.feed_x = None
        self.prev_t = None
        self.joystick = None
        self.keys = key.KeyStateHandler()
        self.window.push_handlers(self.keys)
        self.window.on_key_press = self.on_key_press
        self.main_car = None
        self.heat = None
        self.heatmap = None
        self.heatmap_valid = False
        self.heatmap_show = False
        self.cm = matplotlib.cm.jet
        self.paused = False
        self.label = pyglet.text.Label(
            'Speed: ',
            font_name='Times New Roman',
            font_size=24,
            x=30, y=self.window.height-30,
            anchor_x='left', anchor_y='top'
        )
        def centered_image(filename):
            img = pyglet.resource.image(filename)
            img.anchor_x = img.width/2.
            img.anchor_y = img.height/2.
            return img
        def car_sprite(color, scale=0.15/600.):
            sprite = pyglet.sprite.Sprite(centered_image('car-{}.png'.format(color)), subpixel=True)
            sprite.scale = scale
            return sprite
        def object_sprite(name, scale=0.15/600.):
            sprite = pyglet.sprite.Sprite(centered_image('{}.png'.format(name)), subpixel=True)
            sprite.scale = scale
            return sprite
        self.sprites = {c: car_sprite(c) for c in ['red', 'yellow', 'purple', 'white', 'orange', 'gray', 'blue']}
        self.obj_sprites = {c: object_sprite(c) for c in ['cone', 'firetruck']}
    def use_world(self, world):
        self.cars = [c for c in world.cars]
        self.lanes = [c for c in world.lanes]
        self.objects = [c for c in world.objects]
    def on_key_press(self, symbol, modifiers):
        if symbol == key.ESCAPE:
            self.event_loop.exit()
        if symbol == key.P:
            pyglet.image.get_buffer_manager().get_color_buffer().save('screenshots/screenshot-%.2f.png'%time.time())
        if symbol == key.SPACE:
            self.paused = not self.paused
        if symbol == key.T:
            self.heatmap_show = not self.heatmap_show
            if self.heatmap_show:
                self.heatmap_valid = False
        if symbol == key.J:
            joysticks = pyglet.input.get_joysticks()
            if joysticks and len(joysticks)>=1:
                self.joystick = joysticks[0]
                self.joystick.open()
        if symbol == key.D:
            self.reset()
        if symbol == key.S:
            with open('data/%s-%d.pickle'%(self.name, int(time.time())), 'w') as f:
                pickle.dump((self.history_u, self.history_x), f)
            self.reset()
    def control_loop(self, _=None):
        #print "Time: ", time.time()
        if self.paused:
            return
        if self.iters is not None and len(self.history_x[0])>=self.iters:
            if self.autoquit:
                self.event_loop.exit()
            return
        if self.feed_u is not None and len(self.history_u[0])>=len(self.feed_u[0]):
            if self.autoquit:
                self.event_loop.exit()
            return
        if self.pause_every is not None and self.pause_every>0 and len(self.history_u[0])%self.pause_every==0:
            self.paused = True
        steer = 0.
        gas = 0.
        if self.keys[key.UP]:
            gas += 1.
        if self.keys[key.DOWN]:
            gas -= 1.
        if self.keys[key.LEFT]:
            steer += 1.5
        if self.keys[key.RIGHT]:
            steer -= 1.5
        if self.joystick:
            steer -= self.joystick.x*3.
            gas -= self.joystick.y
        self.heatmap_valid = False
        for car in self.cars:
            self.prev_x[car] = car.x
        if self.feed_u is None:
            for car in reversed(self.cars):
                car.control(steer, gas)
        else:
            for car, fu, hu in zip(self.cars, self.feed_u, self.history_u):
                car.u = fu[len(hu)]
        for car, hist in zip(self.cars, self.history_u):
            hist.append(car.u)
        for car in self.cars:
            car.move()
        for car, hist in zip(self.cars, self.history_x):
            hist.append(car.x)
        self.prev_t = time.time()
    def center(self):
        if self.main_car is None:
            return np.asarray([0., 0.])
        elif self.camera_center is not None:
            return np.asarray(self.camera_center[0:2])
        else:
            return self.anim_x[self.main_car][0:2]
    def camera(self):
        o = self.center()
        gl.glOrtho(o[0]-1./self.magnify, o[0]+1./self.magnify, o[1]-1./self.magnify, o[1]+1./self.magnify, -1., 1.)
    def set_heat(self, f):
        x = utils.vector(4)
        u = utils.vector(2)
        func = th.function([], f(0, x, u))
        def val(p):
            x.set_value(np.asarray([p[0], p[1], 0., 0.]))
            return func()
        self.heat = val
    def draw_heatmap(self):
        if not self.heatmap_show:
            return
        SIZE = (256, 256)
        if not self.heatmap_valid:
            o = self.center()
            x0 = o-np.asarray([1.5, 1.5])/self.magnify
            x0 = np.asarray([x0[0]-x0[0]%(1./self.magnify), x0[1]-x0[1]%(1./self.magnify)])
            x1 = x0+np.asarray([4., 4.])/self.magnify
            x0 = o-np.asarray([1., 1.])/self.magnify
            x1 = o+np.asarray([1., 1.])/self.magnify
            self.heatmap_x0 = x0
            self.heatmap_x1 = x1
            vals = np.zeros(SIZE)
            for i, x in enumerate(np.linspace(x0[0], x1[0], SIZE[0])):
                for j, y in enumerate(np.linspace(x0[1], x1[1], SIZE[1])):
                    vals[j, i] = self.heat(np.asarray([x, y]))
            vals = (vals-np.min(vals))/(np.max(vals)-np.min(vals)+1e-6)
            vals = self.cm(vals)
            vals[:,:,3] = 0.7
            vals = (vals*255.99).astype('uint8').flatten()
            vals = (gl.GLubyte * vals.size) (*vals)
            img = pyglet.image.ImageData(SIZE[0], SIZE[1], 'RGBA', vals, pitch=SIZE[1]*4)
            self.heatmap = img.get_texture()
            self.heatmap_valid = True
        gl.glClearColor(1., 1., 1., 1.)
        gl.glEnable(self.heatmap.target)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
        gl.glBindTexture(self.heatmap.target, self.heatmap.id)
        gl.glEnable(gl.GL_BLEND)
        x0 = self.heatmap_x0
        x1 = self.heatmap_x1
        graphics.draw(4, gl.GL_QUADS,
            ('v2f', (x0[0], x0[1], x1[0], x0[1], x1[0], x1[1], x0[0], x1[1])),
            ('t2f', (0., 0., 1., 0., 1., 1., 0., 1.)),
            #('t2f', (0., 0., SIZE[0], 0., SIZE[0], SIZE[1], 0., SIZE[1]))
        )
        gl.glDisable(self.heatmap.target)

    def output_loop(self, _):
        if self.frame%self.subframes==0:
            self.control_loop()
        alpha = float(self.frame%self.subframes)/float(self.subframes)
        for car in self.cars:
            self.anim_x[car] = (1-alpha)*self.prev_x[car]+alpha*car.x
        self.frame += 1

    def animation_loop(self, _):
        t = time.time()
        alpha = min((t-self.prev_t)/self.dt, 1.)
        for car in self.cars:
            self.anim_x[car] = (1-alpha)*self.prev_x[car]+alpha*car.x

    def draw_lane_surface(self, lane):
        gl.glColor3f(0.4, 0.4, 0.4)
        W = 1000
        graphics.draw(4, gl.GL_QUAD_STRIP, ('v2f',
            np.hstack([lane.p-lane.m*W-0.5*lane.w*lane.n, lane.p-lane.m*W+0.5*lane.w*lane.n,
                       lane.q+lane.m*W-0.5*lane.w*lane.n, lane.q+lane.m*W+0.5*lane.w*lane.n])
        ))
    def draw_lane_lines(self, lane):
        gl.glColor3f(1., 1., 1.)
        W = 1000
        graphics.draw(4, gl.GL_LINES, ('v2f',
            np.hstack([lane.p-lane.m*W-0.5*lane.w*lane.n, lane.p+lane.m*W-0.5*lane.w*lane.n,
                       lane.p-lane.m*W+0.5*lane.w*lane.n, lane.p+lane.m*W+0.5*lane.w*lane.n])
        ))
    def draw_car(self, x, color='yellow', opacity=255):
        sprite = self.sprites[color]
        sprite.x, sprite.y = x[0], x[1]
        sprite.rotation = -x[2]*180./math.pi
        sprite.opacity = opacity
        sprite.draw()
    def draw_object(self, obj):
        sprite = self.obj_sprites[obj.name]
        sprite.x, sprite.y = obj.x[0], obj.x[1]
        sprite.rotation = obj.x[2] if len(obj.x)>=3 else 0.
        sprite.draw()
    def on_draw(self):
        self.window.clear()
        gl.glMatrixMode(gl.GL_PROJECTION)
        gl.glPushMatrix()
        gl.glLoadIdentity()
        self.camera()
        gl.glEnable(self.grass.target)
        gl.glEnable(gl.GL_BLEND)
        gl.glBindTexture(self.grass.target, self.grass.id)
        W = 10000.
        graphics.draw(4, gl.GL_QUADS,
            ('v2f', (-W, -W, W, -W, W, W, -W, W)),
            ('t2f', (0., 0., W*5., 0., W*5., W*5., 0., W*5.))
        )
        gl.glDisable(self.grass.target)
        for lane in self.lanes:
            self.draw_lane_surface(lane)
        for lane in self.lanes:
            self.draw_lane_lines(lane)
        for obj in self.objects:
            self.draw_object(obj)
        for car in self.cars:
            if car!=self.main_car and car not in self.visible_cars:
                self.draw_car(self.anim_x[car], car.color)
        if self.heat is not None:
            self.draw_heatmap()
        for car in self.cars:
            if car==self.main_car or car in self.visible_cars:
                self.draw_car(self.anim_x[car], car.color)
        gl.glPopMatrix()
        if isinstance(self.main_car, Car):
            self.label.text = 'Speed: %.2f'%self.anim_x[self.main_car][3]
            self.label.draw()
        if self.output is not None:
            pyglet.image.get_buffer_manager().get_color_buffer().save(self.output.format(self.frame))

    def reset(self):
        for car in self.cars:
            car.reset()
        self.prev_t = time.time()
        for car in self.cars:
            self.prev_x[car] = car.x
            self.anim_x[car] = car.x
        self.paused = True
        self.history_x = [[] for car in self.cars]
        self.history_u = [[] for car in self.cars]

    def run(self, filename=None, pause_every=None):
        self.pause_every = pause_every
        self.reset()
        if filename is not None:
            with open(filename) as f:
                self.feed_u, self.feed_x = pickle.load(f)
        if self.output is None:
            pyglet.clock.schedule_interval(self.animation_loop, 0.02)
            pyglet.clock.schedule_interval(self.control_loop, self.dt)
        else:
            self.paused = False
            self.subframes = 6
            self.frame = 0
            self.autoquit = True
            pyglet.clock.schedule(self.output_loop)
        self.event_loop.run()

if __name__ == '__main__' and False:
    import lane
    dyn = dynamics.CarDynamics(0.1)
    vis = Visualizer(dyn.dt)
    vis.lanes.append(lane.StraightLane([0., -1.], [0., 1.], 0.13))
    vis.lanes.append(vis.lanes[0].shifted(1))
    vis.lanes.append(vis.lanes[0].shifted(-1))
    vis.cars.append(car.UserControlledCar(dyn, [0., 0., math.pi/2., .1]))
    vis.cars.append(car.SimpleOptimizerCar(dyn, [0., 0.5, math.pi/2., 0.], color='red'))
    r = -60.*vis.cars[0].linear.gaussian()
    r = r + vis.lanes[0].gaussian()
    r = r + vis.lanes[1].gaussian()
    r = r + vis.lanes[2].gaussian()
    r = r - 30.*vis.lanes[1].shifted(1).gaussian()
    r = r - 30.*vis.lanes[2].shifted(-1).gaussian()
    r = r + 30.*feature.speed(0.5)
    r = r + 10.*vis.lanes[0].gaussian(10.)
    r = r + .1*feature.control()
    vis.cars[1].reward = r
    vis.main_car = vis.cars[0]
    vis.paused = True
    vis.set_heat(r)
    #vis.set_heat(vis.lanes[0].gaussian()+vis.lanes[1].gaussian()+vis.lanes[2].gaussian())
    #vis.set_heat(-vis.cars[1].traj.gaussian()+vis.lanes[0].gaussian()+vis.lanes[1].gaussian()+vis.lanes[2].gaussian())
    vis.run()

if __name__ == '__main__' and len(sys.argv)==1:
    import world as wrld
    import car
    world = wrld.world2()
    vis = Visualizer(0.1, name='replay')
    vis.use_world(world)
    vis.main_car = world.cars[0]
    #vis.cars = []
    #vis.cars.append(car.Car(world.cars[0].dyn,[-2., -2., 0., 0.], color='yellow'))
    vis.run()

if __name__ == '__main__' and len(sys.argv)>1:
    filename = sys.argv[1]
    import world
    world_name = (filename.split('/')[-1]).split('-')[0]
    magnify = 1.
    if len(sys.argv)>3:
        magnify = float(sys.argv[3])
    vis = Visualizer(0.2, name=world_name, magnify=magnify)
    the_world = getattr(world, world_name)()
    vis.use_world(the_world)
    vis.main_car = the_world.cars[0]
    if len(sys.argv)>4:
        vis.camera_center = list(eval(sys.argv[4]))
    if len(sys.argv)>2:
        pause_every = int(sys.argv[2])
        vis.run(filename, pause_every=pause_every)
    else:
        vis.run(filename)
