import math
from random import shuffle

import matplotlib.pyplot as plt


def predict(X, B):
	res = []
	for i in range(len(X[0])):
		res.append(B[0] + sum(X[j][i] * B[j + 1] for j in range(len(X))))
	return res


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


def det(X):
	if len(X) == 1:
		return X[0][0]
	if len(X) == 2:
		return X[0][0] * X[1][1] - X[0][1] * X[1][0]
	res = 0
	for i in range(len(X)):
		X_s = list()
		for j in range(len(X)):
			if i != j:
				X_s.append(X[j][:])
				X_s[-1].pop(0)
		res += (-1) ** i * X[i][0] * det(X_s)
	return res


def alg_dops(X):
	X_dops = [[0] * len(X) for _ in range(len(X))]
	for i in range(len(X)):
		for j in range(len(X)):
			minor = []
			for k in range(len(X)):
				if k != i:
					row = []
					for m in range(len(X)):
						if m != j:
							row.append(X[k][m])
					minor.append(row)
			minor_det = det(minor)
			X_dops[i][j] = (-1) ** (i + j) * minor_det
	return X_dops


def obrat(X):  # assert matrix is квадратная
	X_det = det(X)
	X_dops = alg_dops(X)
	X_dops_T = transponate(X_dops)
	res = list()
	for i in range(len(X_dops_T)):
		res.append([])
		for j in range(len(X_dops_T)):
			res[i].append(X_dops_T[i][j] / X_det)
	return res


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
if len(dataset) == 0:
	print('Кривой файл, проверьте валидность и возвращайтесь')
	exit(-1)
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
s = [list(map(float, s[i].split(';'))) for i in range(len(s))]
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
S = learning['median_house_value']  # вектор наблюдений зависимой переменной
X = [
	[1] * len(S),
	learning['longitude'],
	learning['latitude'],
]  # матрица значений независимых переменных
X_shtrih = transponate(X)  # транспонированная
XX_shtrih = multiply(X, X_shtrih)  # перемножили
XX_shtrih_minus1 = obrat(XX_shtrih)  # нашли обратную
XS = list(sum(X[j][i] * S[i] for i in range(len(S))) for j in range(len(X)))  # умножаем на вектор
B = list(sum(XX_shtrih_minus1[j][i] * XS[i] for i in range(len(XS))) for j in range(len(XX_shtrih_minus1)))  # сейм
# B = (XX')^-1 XS
predicted = predict([testing['longitude'], testing['latitude']], B)
R2 = 1 - sum((predicted[i] - testing['median_house_value'][i]) ** 2 for i in range(len(predicted))) / sum(
	(testing['median_house_value'][i] - sum(testing['median_house_value']) / len(testing['median_house_value'])) ** 2
	for i in range(len(testing['median_house_value'])))
print(f'R^2 для зависимости от локации: {R2}')