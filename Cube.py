# to represent the 3x3 Rubik's Cube environment
import pycuber as pc

class Cube(object):
    def __init__(self):
        self.pycube = pc.Cube()
        self.alg = pc.Formula()

        self.statesize = 20 * 24
        self.actionsize = 12

        self.solvedstr = str(pc.Cube())

        # actions
        self.action_list = ["U", "U'", "D", "D'", "L", "L'", "R", "R'", "F", "F'", "B", "B'"]
        self.action_map = {self.action_list[i]: i for i in range(len(self.action_list))}

        self.inverse = {
            "U":"U'",
            "U'":"U",
            "D":"D'",
            "D'":"D",
            "L":"L'",
            "L'":"L",
            "R":"R'",
            "R'":"R",
            "F":"F'",
            "F'":"F",
            "B":"B'",
            "B'":"B",
        }

    

    # reset the environment by generating a scramble n moves away from the solved state
    def reset(self, n=-1):
        self.pycube = pc.Cube()
        randomalg = ""
        if (n < 0):
            randomalg = self.alg.random()
        elif (n != 0):
            randomalg = self.alg.random(n)
        # execute the algorithm on the cube
        self.pycube(randomalg)
        return self.pycube, randomalg
    
    # step by executing the current action on the cube
    def step(self, act):
        self.pycube(act)
        reward = -1
        done = False
        # check for termination by comparing to the solved state
        if str(self.pycube) == self.solvedstr:
            reward = 1
            done = True
        return self.pycube, reward, done

    # render the cube in terminal using colors
    def render(self):
        print(repr(self.pycube))

    # explore each action from the current state
    def explore(self):
        res_states, res_rewards, res_dones = [], [], []
        for action in self.action_list:
            # step in the direction
            next_s, reward, done = self.step(action)
            res_states.append(str(next_s))
            res_rewards.append(reward)
            res_dones.append(int(done))
            # revert it back
            _,_,_ = self.step(self.inverse[action])
        return res_states, res_rewards, res_dones



# Uncomment to debug
# env = Cube()
# print("B2 D F R' U L")
# pcube = env.step("")
# print("")
# env.render()
# print("Solved! Sequence: L' U' R F' D' B' B'")