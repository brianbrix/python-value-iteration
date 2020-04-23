import operator


def add_vector(a, b):
    return tuple(map(operator.add, a, b))


orientations = [(1, 0), (0, 1), (-1, 0), (0, -1)]


def turn_right(orientation):
    return orientations[orientations.index(orientation) - 1]


def turn_left(orientation):
    return orientations[(orientations.index(orientation) + 1) % len(orientations)]


def isnumber(x):
    return hasattr(x, '__int__')


class MDP:
    def __init__(self, init_pos, actlist, terminals, obstacles, transitions={}, states=None, gamma=0.9):
        if not (0 < gamma <= 1):
            raise ValueError("MDP should have 0 < gamma <= 1 values")
        if states:
            self.states = states
        else:
            self.states = set()
            self.init_pos = init_pos
            self.actlist = actlist
            self.terminals = terminals
            self.obstacles = obstacles
            self.transitions = transitions
            self.gamma = gamma
            self.reward = {}

    def R(self, state):
        return self.reward[state]

    def T(self, state, action):
        if self.transitions == {}:
            raise ValueError("Transition model is missing")
        else:
            return self.transitions[state][action]

    def actions(self, state):
        if state in self.terminals:
            return [None]
        if state in self.obstacles:
            return ["Obstacles"]
        else:
            return self.actlist


class GridMDP(MDP):
    def __init__(self, grid, terminals, obstacles, init_pos=(0, 0), gamma=0.9):
        # grid.reverse()
        MDP.__init__(self, init_pos, actlist=orientations,
                     terminals=terminals, obstacles=obstacles, gamma=gamma)

        self.grid = grid
        self.rows = len(grid)
        self.cols = len(grid[0])
        print(self.cols)
        print(self.rows)
        for x in range(self.cols):
            for y in range(self.rows):
                self.reward[x, y] = grid[y][x]
                if grid[y][x] is not None:
                    self.states.add((x, y))

    def T(self, state, action):

        if action is None:
            return [(0.0, state)]
        if (state[1] == 0 and action[1] == -1) or (state[0] == self.cols - 1 and action[0] == 1) or (
                state[1] == self.rows - 1 and action[1] == 1) or (state[0] == 0 and action[0] == -1):
            return [
                (0.5, self.go(state, turn_right(action))),
                (0.5, self.go(state, turn_left(action)))
            ]
        if action is "Obstacles":
            return [(0.7, self.go(state, (1, 1))),
                    (0.1, self.go(state, turn_right((1, 0)))),
                    (0.1, self.go(state, turn_left((1, 0)))),
                    (0.1, state)]
        else:
            return [(0.7, self.go(state, action)),
                    (0.1, self.go(state, turn_right(action))),
                    (0.1, self.go(state, turn_left(action))),
                    (0.1, state)]

    def go(self, state, direction):
        state1 = add_vector(state, direction)
        return state1 if state1 in self.states else state

    def to_grid(self, mapping):
        return list(([[mapping.get((x, y), None)
                       for x in range(self.cols)]
                      for y in range(self.rows)]))

    def to_arrows(self, policy):
        chars = {(1, 0): '>', (0, 1): 'v', (-1, 0): '<', (0, -1): '^', None: '.', "Obstacles": "o"}
        return self.to_grid({s: chars[a] for (s, a) in policy.items()})


def value_iteration(mdp, epsilon=0.001):
    STSN = {s: 0 for s in mdp.states}
    R, T, gamma = mdp.R, mdp.T, mdp.gamma
    while True:
        STS = STSN.copy()
        delta = 0
        for s in mdp.states:
            STSN[s] = R(s) + gamma * max([sum([p * STS[s1] for
                                               (p, s1) in T(s, a)]) for a in mdp.actions(s)])
            delta = max(delta, abs(STSN[s] - STS[s]))
        if delta < epsilon:
            # * (1 - gamma) / gamma
            return STS


def argmax(seq, fn):
    best = seq[0]
    best_score = fn(best)
    for x in seq:
        x_score = fn(x)
        if x_score > best_score:
            best, best_score = x, x_score
    return best


def best_policy(mdp, STS):
    pi = {}
    for s in mdp.states:
        pi[s] = argmax(mdp.actions(s), lambda a: expected_utility(a, s, STS, mdp))
    return pi


def expected_utility(a, s, STS, mdp):
    return sum([p * STS[s1] for (p, s1) in mdp.T(s, a)])


lines = []
with open("input.txt") as file_in:
    for line in file_in:
        lines.append(line.strip("\n").lstrip().rstrip())
grid_list = []
for line in range(int(lines[0])):
    row = [-1 for x in range(int(lines[0]))]
    grid_list.append(row)
obstacles = []
for line_item in lines[2:-1]:
    line_item = (line_item.split(","))
    obstacles.append(tuple(int(x) for x in line_item))
    grid_list[int(line_item[1])][int(line_item[0])] = -100
last_line = (lines[-1].split(","))
grid_list[int(last_line[1])][int(last_line[0])] = +100
last_line = tuple(int(x) for x in last_line)

sequential_decision_environment = GridMDP(grid_list,
                                          terminals=[last_line], obstacles=obstacles)

value_iter = best_policy(sequential_decision_environment, value_iteration(sequential_decision_environment, .01))


def print_table(table, header=None, sep=' ', numfmt='{}'):
    justs = ['rjust' if isnumber(x) else 'ljust' for x in table[0]]
    if header:
        table.insert(0, header)
    table = [[numfmt.format(x) if isnumber(x) else x for x in row]
             for row in table]
    sizes = list(map(lambda seq: max(map(len, seq)),
                     list(zip(*[map(str, row) for row in table]))))
    for row in table:
        print(sep.join(getattr(str(x), j)(size) for (j, size, x)
                       in zip(justs, sizes, row)))
    with open("output.txt", "w") as f:
        for row in table:
            f.writelines(sep.join(getattr(str(x), j)(size) for (j, size, x)
                                  in zip(justs, sizes, row)).replace(" ", "") + "\n")

# for printing on console######################################################
# print(sequential_decision_environment.to_arrows(value_iter))
# print_table(sequential_decision_environment.to_arrows(value_iter))
##for testing ratio of accuracy#################################################
# from difflib import SequenceMatcher
#
# text1 = open("output-4.txt").read()
# text2 = open("output.txt").read()
# m = SequenceMatcher(None, text1, text2)
# print(m.quick_ratio())
