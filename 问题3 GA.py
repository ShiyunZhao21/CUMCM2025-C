import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')


class q3GeneticAlgorithmOptimizer:
    def __init__(self, q3file_path, q3omega=0.4, q3min_sample_size=200, q3min_pj=0.75):
        self.q3file_path = q3file_path
        self.q3omega = q3omega
        self.q3min_sample_size = q3min_sample_size
        self.q3min_pj = q3min_pj
        self.q3data = None

        self.q3population_size = 100
        self.q3num_generations = 500
        self.q3mutation_rate = 0.15
        self.q3crossover_rate = 0.8
        self.q3elite_size = 10

        self.q3load_data()

    def q3load_data(self):
        try:
            self.q3data = pd.read_excel(self.q3file_path)

            if len(self.q3data.columns) >= 4:
                self.q3data.columns = ['Age', 'BMI', 'theta', 'GW']

        except Exception as e:
            self.q3data = None

    def q3calculate_risk(self, q3t):
        if isinstance(q3t, (list, np.ndarray)):
            q3t = np.array(q3t)
            q3risk = np.zeros_like(q3t, dtype=float)
            q3risk[q3t <= 12] = 0
            q3mask_13_27 = (q3t >= 13) & (q3t < 28)
            q3risk[q3mask_13_27] = (q3t[q3mask_13_27] - 12) / 15
            q3risk[q3t >= 28] = 2
            return q3risk
        else:
            if q3t <= 12:
                return 0
            elif 13 <= q3t < 28:
                return (q3t - 12) / 15
            else:
                return 2

    def q3divide_bmi_groups(self, q3B1, q3B2, q3B3):
        q3groups = []
        q3bmi_values = self.q3data['BMI'].values

        q3mask1 = q3bmi_values < q3B1
        q3groups.append(self.q3data[q3mask1])

        q3mask2 = (q3bmi_values >= q3B1) & (q3bmi_values < q3B2)
        q3groups.append(self.q3data[q3mask2])

        q3mask3 = (q3bmi_values >= q3B2) & (q3bmi_values < q3B3)
        q3groups.append(self.q3data[q3mask3])

        q3mask4 = q3bmi_values >= q3B3
        q3groups.append(self.q3data[q3mask4])

        return q3groups

    def q3calculate_pj(self, q3group_data, q3detection_time):
        if len(q3group_data) == 0:
            return 0

        q3gw_values = q3group_data['GW'].values
        q3count_greater_equal = np.sum(q3gw_values >= q3detection_time)
        return q3count_greater_equal / len(q3group_data)

    def q3create_individual(self):
        q3b_values = sorted(np.random.uniform(26.1, 38.9, 3))
        q3B1, q3B2, q3B3 = q3b_values

        q3T1 = np.random.randint(10, 26)
        q3T2 = np.random.randint(10, 26)
        q3T3 = np.random.randint(10, 26)
        q3T4 = np.random.randint(10, 26)

        return np.array([q3B1, q3B2, q3B3, q3T1, q3T2, q3T3, q3T4])

    def q3create_population(self):
        q3population = []
        for _ in range(self.q3population_size):
            q3individual = self.q3create_individual()
            q3population.append(q3individual)
        return np.array(q3population)

    def q3evaluate_constraints(self, q3individual):
        q3B1, q3B2, q3B3, q3T1, q3T2, q3T3, q3T4 = q3individual
        q3penalty = 0

        if not (26 < q3B1 < q3B2 < q3B3 < 39):
            return 1e8

        if not all(10 <= q3t <= 25 for q3t in [q3T1, q3T2, q3T3, q3T4]):
            return 1e8

        q3groups = self.q3divide_bmi_groups(q3B1, q3B2, q3B3)
        q3group_sizes = [len(q3group) for q3group in q3groups]

        for q3size in q3group_sizes:
            if q3size < self.q3min_sample_size:
                q3penalty += (self.q3min_sample_size - q3size) * 1000

        q3detection_times = [q3T1, q3T2, q3T3, q3T4]
        q3pj_violations = 0

        for q3group, q3detection_time in zip(q3groups, q3detection_times):
            if len(q3group) > 0:
                q3pj = self.q3calculate_pj(q3group, q3detection_time)
                if q3pj <= self.q3min_pj:
                    q3pj_violations += (self.q3min_pj - q3pj) * 10000

        q3penalty += q3pj_violations

        return q3penalty

    def q3evaluate_objective(self, q3individual):
        q3B1, q3B2, q3B3, q3T1, q3T2, q3T3, q3T4 = q3individual

        q3groups = self.q3divide_bmi_groups(q3B1, q3B2, q3B3)
        q3detection_times = [q3T1, q3T2, q3T3, q3T4]

        q3total_risk = 0
        q3total_weighted_pj = 0

        for q3group, q3detection_time in zip(q3groups, q3detection_times):
            if len(q3group) > 0:
                q3pj = self.q3calculate_pj(q3group, q3detection_time)

                q3ri = self.q3calculate_risk(q3detection_time)
                q3total_risk += q3ri

                q3group_size = len(q3group)
                q3total_weighted_pj += q3pj * q3group_size

        q3objective_value = self.q3omega * q3total_risk - (1 - self.q3omega) * q3total_weighted_pj

        return q3objective_value

    def q3fitness_function(self, q3individual):
        q3constraint_penalty = self.q3evaluate_constraints(q3individual)

        if q3constraint_penalty > 0:
            return -q3constraint_penalty

        q3objective_value = self.q3evaluate_objective(q3individual)

        return -q3objective_value

    def q3evaluate_population(self, q3population):
        q3fitness_scores = []
        for q3individual in q3population:
            q3fitness = self.q3fitness_function(q3individual)
            q3fitness_scores.append(q3fitness)
        return np.array(q3fitness_scores)

    def q3selection(self, q3population, q3fitness_scores, q3k=3):
        q3selected = []
        for _ in range(len(q3population)):
            q3tournament_indices = np.random.choice(len(q3population), q3k, replace=False)
            q3tournament_fitness = q3fitness_scores[q3tournament_indices]
            q3winner_index = q3tournament_indices[np.argmax(q3tournament_fitness)]
            q3selected.append(q3population[q3winner_index].copy())
        return np.array(q3selected)

    def q3crossover(self, q3parent1, q3parent2):
        if np.random.random() > self.q3crossover_rate:
            return q3parent1.copy(), q3parent2.copy()

        q3child1 = q3parent1.copy()
        q3child2 = q3parent2.copy()

        for q3i in range(3):
            q3alpha = np.random.random()
            q3child1[q3i] = q3alpha * q3parent1[q3i] + (1 - q3alpha) * q3parent2[q3i]
            q3child2[q3i] = q3alpha * q3parent2[q3i] + (1 - q3alpha) * q3parent1[q3i]

        q3child1[:3] = np.sort(q3child1[:3])
        q3child2[:3] = np.sort(q3child2[:3])

        q3crossover_point = np.random.randint(3, 7)
        q3temp = q3child1[q3crossover_point:].copy()
        q3child1[q3crossover_point:] = q3child2[q3crossover_point:]
        q3child2[q3crossover_point:] = q3temp

        q3child1[3:] = np.round(q3child1[3:])
        q3child2[3:] = np.round(q3child2[3:])

        return q3child1, q3child2

    def q3mutation(self, q3individual):
        if np.random.random() > self.q3mutation_rate:
            return q3individual

        q3mutated = q3individual.copy()

        for q3i in range(3):
            if np.random.random() < 0.3:
                q3noise = np.random.normal(0, 1.0)
                q3mutated[q3i] += q3noise
                q3mutated[q3i] = np.clip(q3mutated[q3i], 26.1, 38.9)

        q3mutated[:3] = np.sort(q3mutated[:3])

        for q3i in range(3, 7):
            if np.random.random() < 0.3:
                if np.random.random() < 0.5:
                    q3mutated[q3i] += 1
                else:
                    q3mutated[q3i] -= 1
                q3mutated[q3i] = np.clip(q3mutated[q3i], 10, 25)

        q3mutated[3:] = np.round(q3mutated[3:])

        return q3mutated

    def q3optimize(self):
        if self.q3data is None:
            return None, None

        q3population = self.q3create_population()

        q3best_fitness_history = []
        q3best_individual = None
        q3best_fitness = float('-inf')

        q3progress_bar = tqdm(range(self.q3num_generations), desc="遗传算法进化", colour='green')

        try:
            for q3generation in q3progress_bar:
                q3fitness_scores = self.q3evaluate_population(q3population)

                q3current_best_idx = np.argmax(q3fitness_scores)
                q3current_best_fitness = q3fitness_scores[q3current_best_idx]

                if q3current_best_fitness > q3best_fitness:
                    q3best_fitness = q3current_best_fitness
                    q3best_individual = q3population[q3current_best_idx].copy()

                q3best_fitness_history.append(q3best_fitness)

                q3constraint_penalty = self.q3evaluate_constraints(q3best_individual)
                if q3constraint_penalty == 0:
                    q3objective_value = self.q3evaluate_objective(q3best_individual)
                    q3progress_bar.set_description(f"第{q3generation + 1}代 - 最佳目标函数值: {q3objective_value:.4f}")
                else:
                    q3progress_bar.set_description(f"第{q3generation + 1}代 - 约束惩罚: {q3constraint_penalty:.0f}")

                q3elite_indices = np.argsort(q3fitness_scores)[-self.q3elite_size:]
                q3elite_population = q3population[q3elite_indices]

                q3selected_population = self.q3selection(q3population, q3fitness_scores)

                q3new_population = []

                for q3individual in q3elite_population:
                    q3new_population.append(q3individual.copy())

                while len(q3new_population) < self.q3population_size:
                    q3parent1 = q3selected_population[np.random.randint(len(q3selected_population))]
                    q3parent2 = q3selected_population[np.random.randint(len(q3selected_population))]

                    q3child1, q3child2 = self.q3crossover(q3parent1, q3parent2)

                    q3child1 = self.q3mutation(q3child1)
                    q3child2 = self.q3mutation(q3child2)

                    q3new_population.extend([q3child1, q3child2])

                q3population = np.array(q3new_population[:self.q3population_size])

        except KeyboardInterrupt:
            return None, None

        finally:
            q3progress_bar.close()

        q3final_constraint_penalty = self.q3evaluate_constraints(q3best_individual)

        if q3final_constraint_penalty == 0:
            q3final_objective_value = self.q3evaluate_objective(q3best_individual)
            print(f"\n行")
            print(f"{q3final_objective_value:.6f}")
            return q3best_individual, q3final_objective_value
        else:
            print(f"\n不行")
            return q3best_individual, None

    def q3analyze_results(self, q3optimal_params):
        if q3optimal_params is None:
            print("不行")
            return

        q3B1, q3B2, q3B3, q3T1, q3T2, q3T3, q3T4 = q3optimal_params

        q3T1, q3T2, q3T3, q3T4 = int(q3T1), int(q3T2), int(q3T3), int(q3T4)

        print(f"BMI区间端点: B1={q3B1:.2f}, B2={q3B2:.2f}, B3={q3B3:.2f}")
        print(f"检测时间: T1={q3T1}, T2={q3T2}, T3={q3T3}, T4={q3T4}")

        q3groups = self.q3divide_bmi_groups(q3B1, q3B2, q3B3)
        q3detection_times = [q3T1, q3T2, q3T3, q3T4]

        q3total_risk = 0
        q3total_weighted_pj = 0
        q3constraint_satisfied = True

        for q3j, (q3group, q3detection_time) in enumerate(zip(q3groups, q3detection_times)):
            q3group_size = len(q3group)
            q3pj = self.q3calculate_pj(q3group, q3detection_time) if q3group_size > 0 else 0
            q3ri = self.q3calculate_risk(q3detection_time)

            if q3j == 0:
                q3bmi_range = f"BMI < {q3B1:.2f}"
            elif q3j == 1:
                q3bmi_range = f"{q3B1:.2f} ≤ BMI < {q3B2:.2f}"
            elif q3j == 2:
                q3bmi_range = f"{q3B2:.2f} ≤ BMI < {q3B3:.2f}"
            else:
                q3bmi_range = f"BMI ≥ {q3B3:.2f}"

            q3sample_ok = q3group_size >= self.q3min_sample_size
            q3pj_ok = q3pj > self.q3min_pj

            q3status = "✓" if (q3sample_ok and q3pj_ok) else "✗"


            if not (q3sample_ok and q3pj_ok):
                q3constraint_satisfied = False

            q3total_risk += q3ri
            q3total_weighted_pj += q3pj * q3group_size

        if q3constraint_satisfied:
            q3objective_value = self.q3omega * q3total_risk - (1 - self.q3omega) * q3total_weighted_pj
        else:
            print(f"\n不行")

    def q3plot_results(self, q3optimal_params):
        if q3optimal_params is None:
            return

        q3B1, q3B2, q3B3, q3T1, q3T2, q3T3, q3T4 = q3optimal_params
        q3T1, q3T2, q3T3, q3T4 = int(q3T1), int(q3T2), int(q3T3), int(q3T4)

        q3fig, ((q3ax1, q3ax2), (q3ax3, q3ax4)) = plt.subplots(2, 2, figsize=(15, 12))

        q3ax1.hist(self.q3data['BMI'], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        q3ax1.axvline(q3B1, color='red', linestyle='--', linewidth=2, label=f'B1={q3B1:.2f}')
        q3ax1.axvline(q3B2, color='green', linestyle='--', linewidth=2, label=f'B2={q3B2:.2f}')
        q3ax1.axvline(q3B3, color='orange', linestyle='--', linewidth=2, label=f'B3={q3B3:.2f}')
        q3ax1.set_xlabel('BMI')
        q3ax1.set_ylabel('频数')
        q3ax1.set_title('BMI分布和区间划分')
        q3ax1.legend()
        q3ax1.grid(True, alpha=0.3)

        q3groups = self.q3divide_bmi_groups(q3B1, q3B2, q3B3)
        q3group_sizes = [len(q3group) for q3group in q3groups]
        q3group_labels = [f'区间1\n(<{q3B1:.1f})', f'区间2\n([{q3B1:.1f},{q3B2:.1f})',
                        f'区间3\n([{q3B2:.1f},{q3B3:.1f})', f'区间4\n(≥{q3B3:.1f})']

        q3colors = []
        for q3size in q3group_sizes:
            if q3size >= self.q3min_sample_size:
                q3colors.append('lightgreen')
            else:
                q3colors.append('lightcoral')

        q3bars = q3ax2.bar(q3group_labels, q3group_sizes, color=q3colors)
        q3ax2.axhline(self.q3min_sample_size, color='red', linestyle='--', label=f'最小样本量={self.q3min_sample_size}')
        q3ax2.set_ylabel('样本量')
        q3ax2.set_title('各区间样本量分布')
        q3ax2.legend()
        q3ax2.grid(True, alpha=0.3)

        for q3bar, q3size in zip(q3bars, q3group_sizes):
            q3height = q3bar.get_height()
            q3ax2.text(q3bar.get_x() + q3bar.get_width() / 2., q3height + 10,
                     f'{q3size}', ha='center', va='bottom')

        q3detection_times = [q3T1, q3T2, q3T3, q3T4]
        q3pj_values = [self.q3calculate_pj(q3group, q3dt) if len(q3group) > 0 else 0
                     for q3group, q3dt in zip(q3groups, q3detection_times)]

        q3colors = []
        for q3pj in q3pj_values:
            if q3pj > self.q3min_pj:
                q3colors.append('lightgreen')
            else:
                q3colors.append('lightcoral')

        q3bars = q3ax3.bar(q3group_labels, q3pj_values, color=q3colors)
        q3ax3.axhline(self.q3min_pj, color='red', linestyle='--', label=f'最小Pj={self.q3min_pj}')
        q3ax3.set_ylabel('Pj值')
        q3ax3.set_title('各区间Pj值（孕周≥检测时间的比例）')
        q3ax3.set_ylim(0, 1)
        q3ax3.legend()
        q3ax3.grid(True, alpha=0.3)

        for q3bar, q3pj in zip(q3bars, q3pj_values):
            q3height = q3bar.get_height()
            q3ax3.text(q3bar.get_x() + q3bar.get_width() / 2., q3height + 0.01,
                     f'{q3pj:.3f}', ha='center', va='bottom')

        q3risk_values = [self.q3calculate_risk(q3dt) for q3dt in q3detection_times]

        q3x_pos = np.arange(len(q3detection_times))
        q3bars = q3ax4.bar(q3x_pos, q3detection_times, alpha=0.7, color='lightsteelblue', label='检测时间')
        q3ax4.set_ylabel('检测时间（周）', color='blue')
        q3ax4.set_xlabel('区间')
        q3ax4.set_title('各区间检测时间和对应风险值')
        q3ax4.set_xticks(q3x_pos)
        q3ax4.set_xticklabels([f'区间{q3i + 1}' for q3i in range(4)])
        q3ax4.tick_params(axis='y', labelcolor='blue')

        q3ax4_twin = q3ax4.twinx()
        q3ax4_twin.plot(q3x_pos, q3risk_values, 'ro-', linewidth=2, markersize=8, label='风险值')
        q3ax4_twin.set_ylabel('风险值', color='red')
        q3ax4_twin.tick_params(axis='y', labelcolor='red')

        for q3i, q3dt in enumerate(q3detection_times):
            q3ax4.text(q3i, q3dt + 0.5, f'{int(q3dt)}', ha='center', va='bottom', color='blue')

        for q3i, q3risk in enumerate(q3risk_values):
            q3ax4_twin.text(q3i, q3risk + 0.05, f'{q3risk:.3f}', ha='center', va='bottom', color='red')

        q3ax4.legend(loc='upper left')
        q3ax4_twin.legend(loc='upper right')
        q3ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        return q3fig


def q3main():

    q3file_path = r"D:\HuaweiMoveData\Users\holala\Desktop\人群数据_带随机项计算结果.xlsx"

    q3optimizer = q3GeneticAlgorithmOptimizer(
        q3file_path=q3file_path,
        q3omega=0.4,
        q3min_sample_size=200,
        q3min_pj=0.75
    )

    if q3optimizer.q3data is None:
        print("\nbudui")
        return


    q3response = input("\n要不要开始优化捏？(y/n): ").strip().lower()
    if q3response not in ['y', 'yes', 'okokokokok']:
        print("不优化啦")
        return

    q3optimal_params, q3optimal_value = q3optimizer.q3optimize()

    if q3optimal_params is not None:
        if q3optimal_value is not None:
            print(f"\n✓ 找到")
            print(f"{q3optimal_value:.6f}")
        else:
            print(f"\n无法找到")

        q3optimizer.q3analyze_results(q3optimal_params)


if __name__ == "__main__":
    q3main()
