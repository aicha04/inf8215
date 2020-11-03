
from generator_problem import GeneratorProblem
from state import State

class Solve:

    def __init__(self, n_generator, n_device, seed):

        self.n_generator = n_generator
        self.n_device = n_device
        self.seed = seed

        self.instance = GeneratorProblem.generate_random_instance(self.n_generator, self.n_device, self.seed)

    def solve_naive(self):

        print("Solve with a naive algorithm")
        print("All the generators are opened, and the devices are associated to the closest one")

        opened_generators = [1 for _ in range(self.n_generator)]

        assigned_generators = [None for _ in range(self.n_device)]

        for i in range(self.n_device):
            closest_generator = min(range(self.n_generator),
                                    key=lambda j: self.instance.get_distance(self.instance.device_coordinates[i][0],
                                                                      self.instance.device_coordinates[i][1],
                                                                      self.instance.generator_coordinates[j][0],
                                                                      self.instance.generator_coordinates[j][1])
                                    )
            assigned_generators[i] = closest_generator

        self.instance.solution_checker(assigned_generators, opened_generators)
        total_cost = self.instance.get_solution_cost(assigned_generators, opened_generators)
        self.instance.plot_solution(assigned_generators, opened_generators)

        print("[ASSIGNED-GENERATOR]", assigned_generators)
        print("[OPENED-GENERATOR]", opened_generators)
        print("[SOLUTION-COST]", total_cost)

    def solve_local_search(self):
        print("Solve with a local search algorithm")
        print("initial state all generators are available, and the devices are associated to the closest one")
        maxIteration = self.n_generator -1
        # Etat initial
        available_generators = [i for i in range(self.n_generator)]
        assigned_generators = [None for _ in range(self.n_device)]
        current_state = State(available_generators,assigned_generators)
        current_state.generate_state(self.n_device, self.n_generator, self.instance)
        self.instance.solution_checker(current_state.assigned_generators, current_state.opened_generators)
        # Conditions d'arrets
        itteration = 0
        stopSearch = False
        while itteration < maxIteration and not stopSearch:
            neighborhood = []
            for generator in current_state.available_generators:
                # Voisin
                next_state = State([i for i in current_state.available_generators if i != generator], [None for _ in range(self.n_device)])
                next_state.generate_state(self.n_device, self.n_generator, self.instance)

                # Fonction de validation
                if (next_state.total_cost <= current_state.total_cost) :
                    # Voisin valide
                    neighborhood.append(next_state)
            if(len(neighborhood) == 0):
                # Solution trouvee, arret de la recherche
                stopSearch = True
            else:
                # Fonction de selection
                for neighbor in neighborhood:
                    if(neighbor.total_cost <= current_state.total_cost):
                        # Un meilleur etat a ete trouve
                        current_state = neighbor
                self.instance.solution_checker(current_state.assigned_generators, current_state.opened_generators)
            itteration += 1

        print("---------[RESULTS]-----------")
        print("[AVAILABLE-GENERATORS]", current_state.available_generators)
        print("[ASSIGNED-GENERATOR]", current_state.assigned_generators)
        print("[OPENED-GENERATOR]", current_state.opened_generators)
        print("[SOLUTION-COST]", current_state.total_cost)
        self.instance.plot_solution(current_state.assigned_generators, current_state.opened_generators)
        # print(self.instance.opening_cost)




