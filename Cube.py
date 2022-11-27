# to represent the 3x3 Rubik's Cube environment
import pycuber as pc

class Cube(object):
    def __init__(self):
        self.pycube = pc.Cube()
        self.alg = pc.Formula()

        self.solvedstr = str(pc.Cube())

        # actions
        self.action_list = ["U", "U'", "D", "D'", "L", "L'", "R", "R'", "F", "F'", "B", "B'"]
        self.action_map = {self.action_list[i]: i for i in range(len(self.action_list))}

    

    # reset the environment by generating a scramble n moves away from the solved state
    def reset(self, n=-1):
        self.pycube = pc.Cube()
        randomalg = ""
        if (n < 1):
            randomalg = self.alg.random()
        else:
            randomalg = self.alg.random(n)
        # execute the algorithm on the cube
        cube.pycube(randomalg)
        return self.pycube, randomalg
    
    # step by executing the current action on the cube
    def step(self, act):
        self.pycube(act)
        reward = -1
        done = False
        # check for termination by comparing to the solved state
        # (this is temporary)
        if str(self.pycube) == self.solvedstr:
            reward = 1
            done = True
        return self.pycube, reward, done

    # render the cube in terminal using colors
    def render(self):
        print(repr(self.pycube))




cube = Cube()
randomalg = cube.alg.random(6)
print(randomalg)
cube.pycube(randomalg)
print(cube.pycube)
cube.render()