import VGsim

def setup_simulator(x_close, y_open, lock_dens, seed=0):
    sim = VGsim.Simulator(2, 3, 3, seed=seed)

    sim.set_population_size(1_000_000, population=0)
    sim.set_population_size(500_000,   population=1)
    sim.set_population_size(730_000,   population=2)

    sim.set_transmission_rate(0.25)
    sim.set_recovery_rate(0.099)
    sim.set_sampling_rate(0.001)
    sim.set_transmission_rate(0.5, haplotype="GG")

    mutation_rate = 0.00003
    substitution_weights = [1, 1, 1, 2]  # ATCG
    sim.set_mutation_rate(mutation_rate, substitution_weights)
    sim.set_mutation_rate(3 * mutation_rate, haplotype="G*", mutation=1)

    sim.set_susceptibility_type(1)
    sim.set_susceptibility_type(2, haplotype="G*")

    sim.set_susceptibility(0.1, susceptibility_type=1)
    sim.set_susceptibility(0.5, susceptibility_type=1, haplotype="G*")
    sim.set_immunity_transition(1 / 90, source=1, target=0)  # Изменение иммунитета без инфекции (например, из-за вакцинации или потери иммунитета со временем).

    sim.set_susceptibility(0.0, susceptibility_type=2)
    sim.set_immunity_transition(1 / 180, source=2, target=0)

    sim.set_sampling_multiplier(3, population=1)
    sim.set_sampling_multiplier(1, population=2)

    sim.set_npi([x_close, y_open, lock_dens])  # <-- оптимизируемые
    # sim.set_susceptibility(0.0, susceptibility_type=2)  # челы вообще не заражаются
    # sim.set_immunity_transition(vacc_rate, source=0, target=2)  # <-- оптимизируемые

    sim.set_migration_probability(10 / 365 / 2)  # 10/365 число дней заграницей, 2-число возможных популяций для посещения



    return sim