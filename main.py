import math
from random import shuffle

import matplotlib.pyplot as plt


def calculate_quantile(data, q):
	sorted_data = sorted(data)
	n = len(sorted_data)
	index = q * (n - 1)

	lower_index = int(index)
	upper_index = lower_index + 1

	if upper_index >= n:
		return sorted_data[lower_index]

	weight = index - lower_index
	return sorted_data[lower_index] * (1 - weight) + sorted_data[upper_index] * weight


def transponate(X):
	return [[X[j][i] for j in range(len(X))] for i in range(len(X[0]))]


def multiply(X, Y):
	XY = []
	for i in range(len(X)):
		row = []
		for j in range(len(Y[0])):
			row.append(sum(X[i][k] * Y[k][j] for k in range(len(Y))))
		XY.append(row)
	return XY


def obrat(X):
	... # надо вспомнить


dataset = list()
header: list[str]
invalid_cnt = 0
with open("california_housing_train.csv") as f:
	s = f.readlines()
	header = s.pop(0).replace('"', '').replace('\n', '').split(';')
	for i in range(len(s)):
		try:
			dataset.append(list(map(float, s[i].split(';'))))
		except:
			invalid_cnt += 1

if invalid_cnt > 0:
	print(f'нашли {invalid_cnt} невалидных строк')
else:
	print('все строки валидны')

table = {}
for i in range(len(header)):
	table[header[i]] = []
	for j in range(len(dataset)):
		table[header[i]].append(dataset[j][i])

# 1 grafiki
number = len(dataset)
avg = sum(table['median_house_value']) / number
standart_otkl = math.sqrt(sum((x - avg) ** 2 for x in table['median_house_value']) / (number - 1))
minimum = min(table['median_house_value'])
maximum = max(table['median_house_value'])

quantile5 = calculate_quantile(table['median_house_value'], 0.05)
quantile25 = calculate_quantile(table['median_house_value'], 0.25)
quantile95 = calculate_quantile(table['median_house_value'], 0.95)

print(f"{number=}, {avg=}, {standart_otkl=}, {minimum=}, {maximum=}, {quantile5=}, {quantile25=}")

plt.hist(table['median_house_value'], bins=50)
plt.axvline(avg, color='red', label='среднее')
plt.axvline(quantile5, color='green', label='квантиль 5%')
plt.axvline(quantile25, color='orange', label='квантиль 25%')
plt.axvline(quantile95, color='yellow', label='квантиль 95%')
plt.legend()
plt.show()

# 3 razdeleniye
shuffle(s)
learning_s = [s[i] for i in range(int(0.3 * len(s)), len(s))]
shuffle(learning_s)
testing_s = [s[i] for i in range(int(0.3 * len(s)))]
shuffle(testing_s)
learning = {}
testing = {}
for i in range(len(header)):
	learning[header[i]] = []
	testing[header[i]] = []
	for j in range(len(testing_s)):
		testing[header[i]].append(testing_s[j][i])
	for j in range(len(learning_s)):
		learning[header[i]].append(learning_s[j][i])

# 1 - зависимость от местоположения (longitude, latitude)
S = table['median_house_value']  # вектор наблюдений зависимой переменной
X = [
	[1] * len(table['median_house_value']),
	table['longitude'],
	table['latitude'],
]  # матрица значений независимых переменных
X_shtrih = transponate(X)  # транспонированная
XX_shtrih = multiply(X, X_shtrih)  # перемножили
XX_shtrih_minus1 = obrat(XX_shtrih)  # нашли обратную
B = multiply(XX_shtrih_minus1, XX_shtrih)  # опять перемножили получили B
