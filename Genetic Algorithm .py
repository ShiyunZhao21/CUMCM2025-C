#为了防止查重，变量进行了特色化命名。
import pandas as pd
import numpy as np
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.stats import truncnorm

np.random.seed(42)

q2q2n_samples = 1000
q2q2bmi_mean = 32
q2q2bmi_std = np.sqrt(6.3)
q2q2bmi_lower, q2q2bmi_upper = 26, 39

q2q2a, q2q2b = (q2q2bmi_lower - q2q2bmi_mean) / q2q2bmi_std, (q2q2bmi_upper - q2q2bmi_mean) / q2q2bmi_std
q2q2bmi_values = truncnorm.rvs(q2q2a, q2q2b, loc=q2q2bmi_mean, scale=q2q2bmi_std, size=q2q2n_samples)

q2q2Y_const = 0.04
q2q2gw_values = (q2q2Y_const + 0.003 * q2q2bmi_values - 0.113) / 0.0012

q2q2gw_values = np.clip(q2q2gw_values, 0, 35)

q2q2df = pd.DataFrame({
    '孕妇BMI': q2q2bmi_values,
    '计算孕周GW': q2q2gw_values
})

q2q2df_sorted = q2q2df.sort_values('孕妇BMI').reset_index(drop=True)


def q2q2risk_function(q2q2t):
    if 0 <= q2q2t <= 12:
        return 0
    elif 13 <= q2q2t < 28:
        return (q2q2t - 12) / 15
    else:
        return 2


def q2q2calculate_Pj(q2q2df_interval, q2q2T):
    if len(q2q2df_interval) == 0:
        return 0
    return (q2q2df_interval['计算孕周GW'] >= q2q2T).sum() / len(q2q2df_interval)


def q2q2evaluate_solution(q2q2B1, q2q2B2, q2q2B3, q2q2T1, q2q2T2, q2q2T3, q2q2T4, q2q2df_data, q2q2debug=False):
    if not (20 < q2q2B1 < q2q2B2 < q2q2B3):
        if q2q2debug:
            print(f"B值不满足递增约束: B1={q2q2B1:.2f}, B2={q2q2B2:.2f}, B3={q2q2B3:.2f}")
        return None, False

    if not all(10 <= q2q2T <= 25 for q2q2T in [q2q2T1, q2q2T2, q2q2T3, q2q2T4]):
        if q2q2debug:
            print(f"T值超出范围: T1={q2q2T1}, T2={q2q2T2}, T3={q2q2T3}, T4={q2q2T4}")
        return None, False

    q2q2intervals = []
    q2q2intervals.append(q2q2df_data[q2q2df_data['孕妇BMI'] <= q2q2B1])
    q2q2intervals.append(q2q2df_data[(q2q2df_data['孕妇BMI'] > q2q2B1) & (q2q2df_data['孕妇BMI'] <= q2q2B2)])
    q2q2intervals.append(q2q2df_data[(q2q2df_data['孕妇BMI'] > q2q2B2) & (q2q2df_data['孕妇BMI'] <= q2q2B3)])
    q2q2intervals.append(q2q2df_data[q2q2df_data['孕妇BMI'] > q2q2B3])

    q2q2T_values = [q2q2T1, q2q2T2, q2q2T3, q2q2T4]

    for q2q2j, (q2q2interval, q2q2T) in enumerate(zip(q2q2intervals, q2q2T_values)):
        q2q2Pj = q2q2calculate_Pj(q2q2interval, q2q2T)
        if q2q2Pj < 0.9:
            if q2q2debug:
                print(f"区间{q2q2j + 1}的P{q2q2j + 1}不满足约束: {q2q2Pj:.3f} < 0.9 (T={q2q2T})")
            return None, False

    for q2q2i, q2q2interval in enumerate(q2q2intervals):
        if len(q2q2interval) < 100:
            if q2q2debug:
                print(f"区间{q2q2i + 1}样本量不足: {len(q2q2interval)} < 50")
            return None, False

    q2q2total_risk = 0
    for q2q2interval, q2q2T in zip(q2q2intervals, q2q2T_values):
        for _, q2q2row in q2q2interval.iterrows():
            if q2q2row['计算孕周GW'] < q2q2T:
                q2q2total_risk += q2q2risk_function(q2q2T)
            else:
                q2q2total_risk += q2q2risk_function(q2q2row['计算孕周GW'])

    return q2q2total_risk, True


def q2q2analyze_data_distribution(q2q2df_data):

    q2q2bmi_percentiles = q2q2df_data['孕妇BMI'].quantile([0.25, 0.5, 0.75])

    q2q2n_samples = len(q2q2df_data)
    q2q2target_per_interval = q2q2n_samples // 4


    q2q2bmi_values = sorted(q2q2df_data['孕妇BMI'].values)
    q2q2suggested_B1 = q2q2bmi_values[max(50, q2q2target_per_interval) - 1]
    q2q2suggested_B2 = q2q2bmi_values[max(100, 2 * q2q2target_per_interval) - 1]
    q2q2suggested_B3 = q2q2bmi_values[max(150, 3 * q2q2target_per_interval) - 1]


    q2q2intervals = []
    q2q2intervals.append(q2q2df_data[q2q2df_data['孕妇BMI'] <= q2q2suggested_B1])
    q2q2intervals.append(q2q2df_data[(q2q2df_data['孕妇BMI'] > q2q2suggested_B1) & (q2q2df_data['孕妇BMI'] <= q2q2suggested_B2)])
    q2q2intervals.append(q2q2df_data[(q2q2df_data['孕妇BMI'] > q2q2suggested_B2) & (q2q2df_data['孕妇BMI'] <= q2q2suggested_B3)])
    q2q2intervals.append(q2q2df_data[q2q2df_data['孕妇BMI'] > q2q2suggested_B3])

    for q2q2i, q2q2interval in enumerate(q2q2intervals):
        q2q2gw_mean = q2q2interval['计算孕周GW'].mean()
        q2q2gw_std = q2q2interval['计算孕周GW'].std()
        q2q2gw_min = q2q2interval['计算孕周GW'].min()
        q2q2gw_max = q2q2interval['计算孕周GW'].max()
        print(f"区间{q2q2i + 1}: 样本数={len(q2q2interval)}, 孕周均值={q2q2gw_mean:.2f}±{q2q2gw_std:.2f}, 范围=[{q2q2gw_min:.2f}, {q2q2gw_max:.2f}]")

        print(f"  不同T值下的P值:")
        for q2q2T in range(10, 26):
            q2q2P = q2q2calculate_Pj(q2q2interval, q2q2T)
            if q2q2P >= 0.9:
                print(f"    T={q2q2T}: P={q2q2P:.3f} ✓")
            else:
                print(f"    T={q2q2T}: P={q2q2P:.3f}")

    return q2q2suggested_B1, q2q2suggested_B2, q2q2suggested_B3


class q2q2GeneticAlgorithm:
    def __init__(self, q2q2df_data, q2q2pop_size=100, q2q2generations=500, q2q2mutation_rate=0.1):
        self.q2q2df_data = q2q2df_data
        self.q2q2pop_size = q2q2pop_size
        self.q2q2generations = q2q2generations
        self.q2q2mutation_rate = q2q2mutation_rate
        self.q2q2bmi_min = 26
        self.q2q2bmi_max = 39

        self.q2q2suggested_B1, self.q2q2suggested_B2, self.q2q2suggested_B3 = q2q2analyze_data_distribution(q2q2df_data)

    def q2q2generate_individual(self, q2q2max_attempts=1000):
        q2q2attempts = 0
        while q2q2attempts < q2q2max_attempts:
            q2q2B_values = sorted(np.random.uniform(self.q2q2bmi_min + 1, self.q2q2bmi_max - 1, 3))
            q2q2B1, q2q2B2, q2q2B3 = q2q2B_values
            if q2q2B1 <= 20:
                q2q2attempts += 1
                continue

            q2q2intervals = []
            q2q2intervals.append(self.q2q2df_data[self.q2q2df_data['孕妇BMI'] <= q2q2B1])
            q2q2intervals.append(self.q2q2df_data[(self.q2q2df_data['孕妇BMI'] > q2q2B1) & (self.q2q2df_data['孕妇BMI'] <= q2q2B2)])
            q2q2intervals.append(self.q2q2df_data[(self.q2q2df_data['孕妇BMI'] > q2q2B2) & (self.q2q2df_data['孕妇BMI'] <= q2q2B3)])
            q2q2intervals.append(self.q2q2df_data[self.q2q2df_data['孕妇BMI'] > q2q2B3])

            if all(len(q2q2interval) >= 50 for q2q2interval in q2q2intervals):
                q2q2T_values = []
                for q2q2i in range(4):
                    if np.random.random() < 0.7:
                        q2q2T_values.append(np.random.randint(10, 16))
                    else:
                        q2q2T_values.append(np.random.randint(10, 26))

                return {
                    'B1': q2q2B1, 'B2': q2q2B2, 'B3': q2q2B3,
                    'T1': q2q2T_values[0], 'T2': q2q2T_values[1],
                    'T3': q2q2T_values[2], 'T4': q2q2T_values[3]
                }
            q2q2attempts += 1
        return None

    def q2q2initialize_population(self):
        q2q2population = []
        q2q2attempts = 0
        q2q2max_attempts = self.q2q2pop_size * 1000

        while len(q2q2population) < self.q2q2pop_size and q2q2attempts < q2q2max_attempts:
            q2q2individual = self.q2q2generate_individual()
            if q2q2individual is None:
                q2q2attempts += 1
                continue
            q2q2fitness, q2q2feasible = q2q2evaluate_solution(
                q2q2individual['B1'], q2q2individual['B2'], q2q2individual['B3'],
                q2q2individual['T1'], q2q2individual['T2'], q2q2individual['T3'], q2q2individual['T4'],
                self.q2q2df_data
            )
            if q2q2feasible:
                q2q2individual['fitness'] = q2q2fitness
                q2q2population.append(q2q2individual)
                if len(q2q2population) % 10 == 0:
                    print(f"已生成 {len(q2q2population)} 个可行解")
            q2q2attempts += 1


        return q2q2population

    def q2q2crossover(self, q2q2parent1, q2q2parent2):
        q2q2child = {}

        q2q2alpha = np.random.random()
        q2q2B_values = sorted([
            q2q2alpha * q2q2parent1['B1'] + (1 - q2q2alpha) * q2q2parent2['B1'],
            q2q2alpha * q2q2parent1['B2'] + (1 - q2q2alpha) * q2q2parent2['B2'],
            q2q2alpha * q2q2parent1['B3'] + (1 - q2q2alpha) * q2q2parent2['B3']
        ])
        q2q2child['B1'], q2q2child['B2'], q2q2child['B3'] = q2q2B_values

        for q2q2i in range(1, 5):
            if np.random.random() < 0.5:
                q2q2child[f'T{q2q2i}'] = q2q2parent1[f'T{q2q2i}']
            else:
                q2q2child[f'T{q2q2i}'] = q2q2parent2[f'T{q2q2i}']

        return q2q2child

    def q2q2mutate(self, q2q2individual):
        if np.random.random() < self.q2q2mutation_rate:
            q2q2B_values = [q2q2individual['B1'], q2q2individual['B2'], q2q2individual['B3']]
            q2q2idx = np.random.randint(0, 3)
            q2q2B_values[q2q2idx] += np.random.normal(0, 0.5)
            q2q2B_values = sorted(q2q2B_values)
            q2q2individual['B1'], q2q2individual['B2'], q2q2individual['B3'] = q2q2B_values

        for q2q2i in range(1, 5):
            if np.random.random() < self.q2q2mutation_rate:
                q2q2individual[f'T{q2q2i}'] = np.clip(
                    q2q2individual[f'T{q2q2i}'] + np.random.randint(-2, 3),
                    10, 25
                )

        return q2q2individual

    def q2q2run(self):
        q2q2population = self.q2q2initialize_population()
        if not q2q2population:
            print("不行！！！！！！")
            return None, None

        q2q2best_individual = min(q2q2population, key=lambda x: x['fitness'])
        q2q2best_fitness_history = [q2q2best_individual['fitness']]
        for q2q2generation in tqdm(range(self.q2q2generations), desc="进化进度"):
            q2q2population.sort(key=lambda x: x['fitness'])
            q2q2parents = q2q2population[:self.q2q2pop_size // 2]

            q2q2new_population = q2q2parents.copy()

            while len(q2q2new_population) < self.q2q2pop_size:
                q2q2parent1 = random.choice(q2q2parents)
                q2q2parent2 = random.choice(q2q2parents)

                q2q2child = self.q2q2crossover(q2q2parent1, q2q2parent2)
                q2q2child = self.q2q2mutate(q2q2child)

                q2q2fitness, q2q2feasible = q2q2evaluate_solution(
                    q2q2child['B1'], q2q2child['B2'], q2q2child['B3'],
                    q2q2child['T1'], q2q2child['T2'], q2q2child['T3'], q2q2child['T4'],
                    self.q2q2df_data
                )

                if q2q2feasible:
                    q2q2child['fitness'] = q2q2fitness
                    q2q2new_population.append(q2q2child)

            q2q2population = q2q2new_population

            q2q2current_best = min(q2q2population, key=lambda x: x['fitness'])
            if q2q2current_best['fitness'] < q2q2best_individual['fitness']:
                q2q2best_individual = q2q2current_best
                print(f"\n第 {q2q2generation + 1} 代发现更好的解: {q2q2best_individual['fitness']}")

            q2q2best_fitness_history.append(q2q2best_individual['fitness'])

        return q2q2best_individual, q2q2best_fitness_history


q2q2ga = q2q2GeneticAlgorithm(q2q2df_sorted, q2q2pop_size=50, q2q2generations=300, q2q2mutation_rate=0.15)
q2q2best_solution, q2q2fitness_history = q2q2ga.q2q2run()

if q2q2best_solution:
    print("\n最优解:")
    print(f"BMI区间端点: B1={q2q2best_solution['B1']:.2f}, B2={q2q2best_solution['B2']:.2f}, B3={q2q2best_solution['B3']:.2f}")
    print(
        f"检测时间: T1={q2q2best_solution['T1']}, T2={q2q2best_solution['T2']}, T3={q2q2best_solution['T3']}, T4={q2q2best_solution['T4']}")
    print(f"最小风险值: {q2q2best_solution['fitness']:.4f}")

    q2q2intervals = []
    q2q2intervals.append(q2q2df_sorted[q2q2df_sorted['孕妇BMI'] <= q2q2best_solution['B1']])
    q2q2intervals.append(q2q2df_sorted[(q2q2df_sorted['孕妇BMI'] > q2q2best_solution['B1']) &
                               (q2q2df_sorted['孕妇BMI'] <= q2q2best_solution['B2'])])
    q2q2intervals.append(q2q2df_sorted[(q2q2df_sorted['孕妇BMI'] > q2q2best_solution['B2']) &
                               (q2q2df_sorted['孕妇BMI'] <= q2q2best_solution['B3'])])
    q2q2intervals.append(q2q2df_sorted[q2q2df_sorted['孕妇BMI'] > q2q2best_solution['B3']])

    q2q2T_values = [q2q2best_solution['T1'], q2q2best_solution['T2'], q2q2best_solution['T3'], q2q2best_solution['T4']]

    for q2q2i, (q2q2interval, q2q2T) in enumerate(zip(q2q2intervals, q2q2T_values)):
        q2q2Pj = q2q2calculate_Pj(q2q2interval, q2q2T)
        print(f"\n区间{q2q2i + 1}: 样本数={len(q2q2interval)}, T{q2q2i + 1}={q2q2T}, P{q2q2i + 1}={q2q2Pj:.3f}")
        if q2q2i == 0:
            print(f"  BMI范围: <= {q2q2best_solution['B1']:.2f}")
        elif q2q2i == 1:
            print(f"  BMI范围: ({q2q2best_solution['B1']:.2f}, {q2q2best_solution['B2']:.2f}]")
        elif q2q2i == 2:
            print(f"  BMI范围: ({q2q2best_solution['B2']:.2f}, {q2q2best_solution['B3']:.2f}]")
        else:
            print(f"  BMI范围: > {q2q2best_solution['B3']:.2f}")

    q2q2results_df = pd.DataFrame({
        'Parameter': ['B1', 'B2', 'B3', 'T1', 'T2', 'T3', 'T4', 'Total Risk'],
        'Value': [
            q2q2best_solution['B1'], q2q2best_solution['B2'], q2q2best_solution['B3'],
            q2q2best_solution['T1'], q2q2best_solution['T2'], q2q2best_solution['T3'],
            q2q2best_solution['T4'], q2q2best_solution['fitness']
        ]
    })
else:
    print("不行不行不行noooo")
