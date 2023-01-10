import random
import collections
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import tqdm
import time

from Cube import Cube
from encode_cube import encode
from approx_policy_iter import API_NN
import api_mcts

def solve(env, alg, net, max_seconds=60, max_steps=None, device=torch.device("cpu"), quiet=False, batch_size=1):
    print(f"Solving cube with scramble: {alg}")
    env.render()
    cube_state = env.pycube
    tree = api_mcts.MCTS(env, cube_state, alg, net)
    step_no = 0
    ts = time.time()

    while True:
        if batch_size > 1:
            solution = tree.search_batch(batch_size)
        else:
            solution = tree.search()
        if solution:
            if not quiet:
                print("On step %d we found goal state, unroll. Speed %.2f searches/s" %
                         (step_no, (step_no*batch_size) / (time.time() - ts)))
                print("Tree depths: %s" % tree.get_depth_stats())
                bfs_solution = tree.find_solution()
                print("Solutions: naive %d, bfs %d" %  (len(solution), len(bfs_solution)))
                print("BFS: %s" % bfs_solution)
                print("Naive: %s" % solution)
#                tree.dump_solution(solution)
#                tree.dump_solution(bfs_solution)
#                tree.dump_root()
#                log.info("Tree: %s", tree)
            return tree, solution
        step_no += 1
        if max_steps is not None:
            if step_no > max_steps:
                if not quiet:
                    print("Maximum amount of steps has reached, cube wasn't solved. "
                             "Did %d searches, speed %.2f searches/s" % 
                             (step_no, (step_no*batch_size) / (time.time() - ts)))
                    print("Tree depths: %s" % tree.get_depth_stats())
                return tree, None
        elif time.time() - ts > max_seconds:
            if not quiet:
                print("Time is up, cube wasn't solved. Did %d searches, speed %.2f searches/s.." % 
                         (step_no, (step_no*batch_size) / (time.time() - ts)))
                print("Tree depths: %s" % tree.get_depth_stats())
            return tree, None


def solve_scramble(n=20, maxseconds = 120):
    env = Cube()

    # net
    torch.load('api_model.pt', map_location=torch.device('cpu'))
    net = API_NN().to(torch.float).cpu()
    loadPath = "api_model.pt"
    print(f"Loading from {loadPath}")

    # random scramble a few moves away
    state, alg = env.reset(n)

    # get rid of double moves
    alg = str(alg).replace("2", "")
    _, _ = env.reset(0)
    state, _, _ = env.step(alg)
    print(alg)

    tree, sol = solve(env, alg, net, max_seconds=maxseconds)