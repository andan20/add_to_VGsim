import VGsim
import matplotlib.pyplot as plt


def get_peak_infectious(simulator, num_populs, num_sites, days=100):
    num_haplo = 4 ** num_sites
    total_per_day = [0] * days

    for pop in range(num_populs):
        for haplo in range(num_haplo):
            infectious, _, _, _ = simulator.get_data_infectious(
                population=pop, haplotype=haplo, step_num=days
            )
            for day in range(days):
                total_per_day[day] += int(infectious[day])

    return total_per_day

def get_all_data_infectious(simulator, num_populs, num_sites, days = 100):
    # возвращает матрицу (число популяций)х(2^num_sites)
    # элемент матрицы - массив размера days, который обозначает число зараженных в популяции i гаплотипом j

    num_haplo = 4**num_sites

    ans = []

    for pop in range(num_populs):
        anspop = []
        for haplo in range(num_haplo):
            infectious, sampled, times, lockdowns = simulator.get_data_infectious(
                population= pop,  # какая популяция
                haplotype = haplo,  # какой гаплотип  (0, 1, 2...)
                step_num=days  # на сколько отрезков разбить время
            )

            anspop.append(list(map(int, infectious)))
        ans.append(anspop)

    return ans

def plot_all_data_infectious_all(matrix, figsize_per_cell=(4, 3)):
    n = len(matrix)
    m = len(matrix[0])

    fig, axes = plt.subplots(n, m, figsize=(figsize_per_cell[0] * m, figsize_per_cell[1] * n))
    fig.suptitle('Динамика кол-ва зараженных в популяциях по гаплотипам', fontsize=14, fontweight='bold')

    # Обработка частных случаев размерности
    if n == 1 and m == 1:
        axes = [[axes]]
    elif n == 1:
        axes = [axes]
    elif m == 1:
        axes = [[ax] for ax in axes]

    for i in range(n):

        for_lim = matrix[i][0][0]
        for j in range(m):
            for_lim = max(for_lim, max(matrix[i][j]))

        for j in range(m):
            ax = axes[i][j]

            days = list(range(1, len(matrix[i][j]) + 1))
            values = matrix[i][j]

            # Основной график
            ax.plot(days, values, 'b-', linewidth=1.5)
            ax.scatter(days, values, c='blue', s=10, zorder=5)

            # Настройка осей
            ax.set_title(f'pop {i}, sus {j}', fontsize=10)
            ax.set_xlabel('Дни')
            ax.set_ylabel(f'Кол-во людей из {i} зар-ых гап-ом {j}')
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.set_ylim(0, for_lim * 1.1)



            # Добавляем аннотацию с последним значением
            if values:
                ax.annotate(f'{values[-1]}',
                            xy=(days[-1], values[-1]),
                            xytext=(5, 5), textcoords='offset points',
                            fontsize=9, fontweight='bold')

    plt.tight_layout()
    plt.show()

def plot_all_data_infectious_certain(matrix, pops, haplos, figsize_per_cell=(4, 3)):
    n = len(pops)
    m = len(haplos)

    fig, axes = plt.subplots(n, m, figsize=(figsize_per_cell[0] * m, figsize_per_cell[1] * n))
    fig.suptitle('Динамика кол-ва зараженных в популяциях по гаплотипам', fontsize=14, fontweight='bold')

    # Обработка частных случаев размерности
    if n == 1 and m == 1:
        axes = [[axes]]
    elif n == 1:
        axes = [axes]
    elif m == 1:
        axes = [[ax] for ax in axes]

    cnt_i = 0
    for i in pops:
        cnt_j = 0

        for_lim = matrix[i][haplos[0]][0]
        for j in haplos:
            for_lim = max(for_lim, max(matrix[i][j]))

        for j in haplos:
            ax = axes[cnt_i][cnt_j]

            days = list(range(1, len(matrix[i][j]) + 1))
            values = matrix[i][j]

            # Основной график
            ax.plot(days, values, 'b-', linewidth=1.5)
            ax.scatter(days, values, c='blue', s=10, zorder=5)

            # Настройка осей
            ax.set_title(f'pop {i}, sus {j}', fontsize=10)
            ax.set_xlabel('Дни')
            ax.set_ylabel(f'Кол-во людей из {i} зар-ых гап-ом {j}')
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.set_ylim(0, for_lim * 1.1)



            # Добавляем аннотацию с последним значением
            if values:
                ax.annotate(f'{values[-1]}',
                            xy=(days[-1], values[-1]),
                            xytext=(5, 5), textcoords='offset points',
                            fontsize=9, fontweight='bold')
            cnt_j += 1
        cnt_i += 1

    plt.tight_layout()
    plt.show()

def plot_all_data_infectious_sum(matrix, figsize_per_cell=(4, 3)):
    n = len(matrix)
    m = 1

    fig, axes = plt.subplots(n, m, figsize=(figsize_per_cell[0] * m, figsize_per_cell[1] * n))
    fig.suptitle('Динамика кол-ва зараженных в популяциях суммарно по всем гаплотипам', fontsize=14, fontweight='bold')

    # Обработка частных случаев размерности
    if n == 1 and m == 1:
        axes = [[axes]]
    elif n == 1:
        axes = [axes]
    elif m == 1:
        axes = [[ax] for ax in axes]

    for i in range(n):


        ax = axes[i][0]

        days = list(range(1, len(matrix[i][0]) + 1))
        values = [sum(matrix[i][haplo][day] for haplo in range(len(matrix[i])))
                  for day in range(len(matrix[i][0]))]

        # Основной график
        ax.plot(days, values, 'b-', linewidth=1.5)
        ax.scatter(days, values, c='blue', s=10, zorder=5)

        # Настройка осей
        ax.set_title(f'pop {i}', fontsize=10)
        ax.set_xlabel('Дни')
        ax.set_ylabel(f'Кол-во людей из {i} зар-ых любым гап-ом')
        ax.grid(True, alpha=0.3, linestyle='--')

        # Добавляем аннотацию с последним значением
        if values:
            ax.annotate(f'{values[-1]}',
                        xy=(days[-1], values[-1]),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=9, fontweight='bold')

    plt.tight_layout()
    plt.show()