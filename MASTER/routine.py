
# # JOB SCRIPTER FOR NxM GRID

import random
import copy
import operator
from typing import Callable, Literal

from sympy import Union
WAIT = {"type":"WAIT"}
ext = '.pt' 

class Evolve:
    def __init__(self, M, N, dataset, features={"learning_rate": 5e-4, "beta1": 0.5, 'beta2': 0.5}, epochs=5, mutator={'learning_rate': (-1e-4, 1e-4), 'beta1': (-0.1, 0.1), 'beta2': (-0.1, 0.1)}, limits={'learning_rate': (0, 1e-2), 'beta1': (0.1, 1), 'beta2': (0.1, 1)},
                 gen_metric='loss', disc_metric='loss',gen_compare=operator.lt, disc_compare=operator.lt, gen_tournamet = min, disc_tournamet = min):
        self.M = M
        self.N = N
        self.epochs = epochs
        self.features = features
        self.script = []
        self.gans = {}
        self.names = []
        self.dataset = dataset
        self.subspace_g = {}
        self.best=None
        self.limits = limits
        self.subspace_d = {}
        self.disc_metric = disc_metric
        self.gen_metric = gen_metric
        self.gen_tournament=gen_tournamet
        self.disc_tournament= disc_tournamet
        self.mutator = mutator
        self.gen_compare = gen_compare
        self.disc_compare = disc_compare
        self.cells = [(x, y) for x in range(self.M) for y in range(self.N)]
        self.curr_cell = -1
        self.curr_phase = 0  # [0-> Train disc, 1-> Train gens, 2-> Replace]

        for feature, value in features.items():
            assert limits[feature][0] < value < limits[feature][
                1], f"{feature} should be in range {limits[feature]} but is {value}"

        self.initialize_gans()

    def tournament_selection(self, space, metric,tournament):
        return tournament(space, key=lambda x: float(space[x]["fitness"][metric]))
        

    def initialize_gans(self, clear=True):
        if (clear):
            self.script = []

    # DEFINING GAN PARAMETERS

        for i in range(self.M):
            for j in range(self.N):
                self.names.append(f"gen_{i}_{j}")
                self.gans[f"gen_{i}_{j}"] = {
                    "features": self.features,
                    "fitness": {"loss": 100},
                    "type": "generator"
                }

                self.names.append(f"disc_{i}_{j}")
                self.gans[f"disc_{i}_{j}"] = {
                    "features": self.features,
                    "fitness": {"loss": 100},
                    "type": "discriminator"
                }

        # CREATING N GANs

        self.script += [{
            "type":"create",
            "create":self.gans[x]["type"],
            "name":x,
            "id":random.getrandbits(32)} for x in self.names]

        self.script.append(WAIT)

        # INITIAL TRAINING

        jobs = []
        for coms in range(0, len(self.names), 2):
            jobs.append({
            "type":"train",
            "generator": f"{self.names[coms]}{ext}",
            "discriminator" : f"{self.names[coms+1]}{ext}",
            "gen_lr":self.gans[self.names[coms]]["features"]['learning_rate'],
            "gen_beta1":self.gans[self.names[coms]]["features"]['beta1'],
            "gen_beta2":self.gans[self.names[coms]]["features"]['beta2'],
            "disc_lr":self.gans[self.names[coms+1]]["features"]['learning_rate'],
            "disc_beta1":self.gans[self.names[coms+1]]["features"]['beta1'],
            "disc_beta2":self.gans[self.names[coms+1]]["features"]['beta2'],
            "dataset" : self.dataset,
            "epochs":self.epochs,
            "mode":"train_original",
            "id":random.getrandbits(32)
        })

        self.script += jobs
        self.script.append(WAIT)

        # CHOOSE BEST GEN, TRAIN ALL DISC update metrics before calling

    def train_discs(self, clear=True):
        if (clear):
            self.script = []


        if (self.curr_cell+1 == len(self.cells)):
            self.curr_cell = 0
        else:
            self.curr_cell += 1

        cell = self.cells[self.curr_cell]

        neighbors = self.get_neighbours(cell[0], cell[1], self.M, self.N)
        gans = copy.deepcopy(self.gans)
        self.subspace_d = {f'disc_{x[0]}_{x[1]}': {"features": gans[f'disc_{x[0]}_{x[1]}']
                                                   ["features"], "fitness": gans[f'disc_{x[0]}_{x[1]}']["fitness"]} for x in neighbors}
        self.subspace_g = {f'gen_{x[0]}_{x[1]}': {"features": gans[f'gen_{x[0]}_{x[1]}']
                                                  ["features"], "fitness": gans[f'gen_{x[0]}_{x[1]}']["fitness"]} for x in neighbors}
        best_gen = self.tournament_selection(self.subspace_g, self.gen_metric,self.gen_tournament,)
        self.best=best_gen
        print(f"best gen is {best_gen} {self.subspace_g[best_gen]}")

        jobs = []
        for x in neighbors:

            for feature in self.features:
                mutator = self.mutator[feature]
                limits = self.limits[feature]
                mut_gen = random.uniform(
                    mutator[0], mutator[1]) if 1 == random.choice([0, 1]) else 0
                mut_disc = random.uniform(
                    mutator[0], mutator[1]) if 1 == random.choice([0, 1]) else 0
                
                if mut_gen:                
                    self.subspace_g[f'gen_{x[0]}_{x[1]}']["features"][feature] += mut_gen if limits[0] < self.subspace_g[f'gen_{x[0]}_{x[1]}']["features"][feature]+mut_gen < limits[1] else 0

                if mut_disc:
                    self.subspace_d[f'disc_{x[0]}_{x[1]}']["features"][feature] += mut_disc if limits[0] < self.subspace_d[f'disc_{x[0]}_{x[1]}']["features"][feature]+mut_disc < limits[1] else 0


            jobs.append({
            "type":"train",
            "generator": f"{best_gen}{ext}",
            "discriminator" : f"disc_{x[0]}_{x[1]}{ext}",
            "gen_lr":self.subspace_g[best_gen]["features"]["learning_rate"],
            "gen_beta1":self.subspace_g[best_gen]["features"]["beta1"],
            "gen_beta2":self.subspace_g[best_gen]["features"]["beta2"],
            "disc_lr":self.subspace_d[f'disc_{x[0]}_{x[1]}']["features"]["learning_rate"],
            "disc_beta1":self.subspace_d[f'disc_{x[0]}_{x[1]}']["features"]["beta1"],
            "disc_beta2":self.subspace_d[f'disc_{x[0]}_{x[1]}']["features"]["beta2"],
            "dataset" : self.dataset,
            "epochs":self.epochs,
            "mode":"copy",
            "focus":"discriminator",
            "id":random.getrandbits(32)
        })
        self.script += jobs

        self.script.append(WAIT)

        # CHOOSE BEST DISC, TRAIN ALL GEN and update metrics before calling

    def train_gens(self, clear=True):
        if (clear):
            self.script = []

        cell = self.cells[self.curr_cell]
        best_disc = self.tournament_selection(self.subspace_d, self.disc_metric,self.disc_tournament)

        self.best = best_disc

        neighbors = self.get_neighbours(cell[0], cell[1], self.M, self.N)
        print(f"best disc is {best_disc} {self.subspace_d[best_disc]}")
        jobs = []

        for x in neighbors:

            jobs.append({
            "type":"train",
            "generator": f"gen_{x[0]}_{x[1]}{ext}",
            "discriminator" : f"{best_disc}{ext}",
            "gen_lr":self.subspace_g[f"gen_{x[0]}_{x[1]}"]["features"]['learning_rate'],
            "gen_beta1":self.subspace_g[f"gen_{x[0]}_{x[1]}"]["features"]['beta1'],
            "gen_beta2":self.subspace_g[f"gen_{x[0]}_{x[1]}"]["features"]['beta2'],
            "disc_lr":self.subspace_d[best_disc]["features"]['learning_rate'],
            "disc_beta1":self.subspace_d[best_disc]["features"]['beta1'],
            "disc_beta2":self.subspace_d[best_disc]["features"]['beta2'],
            "dataset" : self.dataset,
            "epochs":self.epochs,
            "mode":"copy",
            "focus":"generator",
            "id":random.getrandbits(32)
        })
        self.script += jobs
        self.script.append(WAIT)

        # update metrics

    def replace_best(self, clear=True):
        if (clear):
            self.script = []
        cell = self.cells[self.curr_cell]
        best_gen = self.tournament_selection(self.subspace_g, self.gen_metric, self.gen_tournament)
        best_disc = self.tournament_selection(self.subspace_d,self.disc_metric, self.gen_tournament)

        if self.gen_compare(float(self.subspace_g[best_gen]['fitness'][self.gen_metric]) ,float(self.gans[f'gen_{cell[0]}_{cell[1]}']['fitness'][self.gen_metric])):
            print(
                f"Replacing gen_{cell[0]}_{cell[1]} : {float(self.gans[f'gen_{cell[0]}_{cell[1]}']['fitness'][self.gen_metric])} with Improvement {best_gen}: {self.subspace_g[best_gen]['fitness']['loss']}")
            # replace best gen and disc into cell[0] cell[1] pos
            job = {
                "type":"replace",
                "source":f"{best_gen}_copy",
                "destination":f"gen_{cell[0]}_{cell[1]}",
                "id":random.getrandbits(32)
            }
            self.script.append(job)
            self.gans[f'gen_{cell[0]}_{cell[1]}']['features'] = copy.deepcopy(
                self.subspace_g[best_gen]['features'])
            self.gans[f'gen_{cell[0]}_{cell[1]}']['fitness'] = copy.deepcopy(
                self.subspace_g[best_gen]['fitness'])
        else:
            print(
                f"No improvement in gen over cell {cell} from {self.gans[f'gen_{cell[0]}_{cell[1]}']['fitness'][self.gen_metric]}")

        if self.disc_compare(float(self.subspace_d[best_disc]['fitness'][self.disc_metric]) , float(self.gans[f'disc_{cell[0]}_{cell[1]}']['fitness'][self.disc_metric])):
            print(f"Replacing 'disc_{cell[0]}_{cell[1]}' : {float(self.gans[f'disc_{cell[0]}_{cell[1]}']['fitness'][self.disc_metric])} with Improvement {best_disc}: {self.subspace_d[best_disc]['fitness'][self.disc_metric]}")
            job = {
                "type":"replace",
                "source":f"{best_disc}_copy",
                "destination":f"disc_{cell[0]}_{cell[1]}",
                "id":random.getrandbits(32)
            }
            self.script.append(job)
            self.gans[f'disc_{cell[0]}_{cell[1]}']['features'] = copy.deepcopy(
                self.subspace_d[best_disc]['features'])
            self.gans[f'disc_{cell[0]}_{cell[1]}']['fitness'] = copy.deepcopy(
                self.subspace_d[best_disc]['fitness'])
        else:
            print(
                f"No improvement in disc over cell {cell} from {self.gans[f'disc_{cell[0]}_{cell[1]}']['fitness'][self.disc_metric]}")

        self.script.append(WAIT)



        return self.script

    def next(self):
        if (self.curr_phase == 0):
            self.train_discs()
            self.curr_phase += 1
        elif (self.curr_phase == 1):
            self.train_gens()
            self.curr_phase += 1
        elif (self.curr_phase == 2):
            self.replace_best()
            self.curr_phase = 0

    def get_neighbours(self, i, j, m, n):
        if i == m or j == n:
            return "Invalid"
        else:
            neighbors = []
            neighbors.append((i, j))
            if (i+1 < m):
                neighbors.append((i+1, j))
                if (j+1 < n):
                    neighbors.append((i+1, j+1))
                else:
                    neighbors.append((i+1, 0))
                if (j-1 >= 0):
                    neighbors.append((i+1, j-1))
                else:
                    neighbors.append((i+1, n-1))
            else:
                neighbors.append((0, j))
                if (j+1 < n):
                    neighbors.append((0, j+1))
                else:
                    neighbors.append((0, 0))
                if (j-1 >= 0):
                    neighbors.append((0, j-1))
                else:
                    neighbors.append((0, n-1))
            if (i-1 >= 0):
                neighbors.append((i-1, j))
                if (j+1 < n):
                    neighbors.append((i-1, j+1))
                else:
                    neighbors.append((i-1, 0))
                if (j-1 >= 0):
                    neighbors.append((i-1, j-1))
                else:
                    neighbors.append((i-1, n-1))
            else:
                neighbors.append((m-1, j))
                if (j+1 < n):
                    neighbors.append((m-1, j+1))
                else:
                    neighbors.append((m-1, 0))
                if (j-1 >= 0):
                    neighbors.append((m-1, j-1))
                else:
                    neighbors.append((m-1, n-1))

            if (j-1 >= 0):
                neighbors.append((i, j-1))
            else:
                neighbors.append((i, n-1))

            if (j+1 < n):
                neighbors.append((i, j+1))
            else:
                neighbors.append((i, 0))

            return set(neighbors)


if __name__ == "__main__":
    e1 = Evolve(3, 2, 'ds')

    # print(*e1.script, sep='\n')

    # print(getNeighbours(0,0,3,4))
