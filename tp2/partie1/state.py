class State:
    def __init__(self, available_generators, assigned_generators):
        self.available_generators = available_generators
        self.assigned_generators = assigned_generators
        self.opened_generators = None
        self.total_cost = None
    def generate_state(self, n_device, n_generator, instance):
        # Assigner le generateur disponible le plus pres a chaque machine
        for i in range(n_device):

            closest_generator = min(self.available_generators,
                                    key=lambda j: instance.get_distance(instance.device_coordinates[i][0],
                                                                             instance.device_coordinates[i][1],
                                                                             instance.generator_coordinates[j][0],
                                                                             instance.generator_coordinates[j][1])
                                    )
            self.assigned_generators[i] = closest_generator
        # Generateurs ouverts
        self.opened_generators = [1 if i in self.assigned_generators else 0 for i in range(n_generator)]
        # Fonction d'evaluation
        self.total_cost = instance.get_solution_cost(self.assigned_generators, self.opened_generators)