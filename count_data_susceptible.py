import VGsim
import matplotlib.pyplot as plt


def get_lockdown_days(simulator, num_populs, days=100):
    total = 0.0
    for pop in range(num_populs):
        _, _, lockdowns = simulator.get_data_susceptible(
            population=pop,
            susceptibility_type=0,  # тип не важен, локдауны одинаковые
            step_num=days
        )
        if not lockdowns:
            continue
        t_start = None
        for event in lockdowns:
            if event[0]:
                t_start = event[1]
            elif t_start is not None:
                total += event[1] - t_start
                t_start = None
        if t_start is not None:
            total += days - t_start
    return total


def get_all_data_susceptible(simulator, num_populs, num_susps, days = 100):
    # возвращает матрицу (число популяций)х(число типов восприимчивости)
    # элемент матрицы - массив размера days, который обозначает число людей популяции i типа восприимчвости j

    ans = []

    for pop in range(num_populs):
        anspop = []
        for sus in range(num_susps):
            sizes, times, lockdowns = simulator.get_data_susceptible(
                population= pop,  # какая популяция
                susceptibility_type= sus,  # какой тип восприимчивости (0, 1, 2...)
                step_num=days  # на сколько отрезков разбить время
            )
            anspop.append(list(map(int, sizes)))
        ans.append(anspop)

    return ans

def plot_all_data_susceptible(matrix, figsize_per_cell=(4, 3)):
    n = len(matrix)
    m = len(matrix[0])

    fig, axes = plt.subplots(n, m, figsize=(figsize_per_cell[0] * m, figsize_per_cell[1] * n))
    fig.suptitle('Динамика кол-ва людей в популяциях по sus gr', fontsize=14, fontweight='bold')

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
            ax = axes[i][j]

            days = list(range(1, len(matrix[i][j]) + 1))
            values = matrix[i][j]

            # Основной график
            ax.plot(days, values, 'b-', linewidth=1.5)
            ax.scatter(days, values, c='blue', s=10, zorder=5)

            # Настройка осей
            ax.set_title(f'pop {i}, sus {j}', fontsize=10)
            ax.set_xlabel('Дни')
            ax.set_ylabel(f'Кол-во людей из {i} в sus {j}')
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