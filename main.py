import math
from random import shuffle

s = ""
header = []
with open("california_housing_train.csv") as f:
	s = f.readlines()
	header = s.pop(0).replace('"', '').replace('\n', '').split(';')
	for i in range(len(s)):
		s[i] = list(map(float, s[i].split(';')))

table = {}
for i in range(len(header)):
	table[header[i]] = []
	for j in range(len(s)):
		table[header[i]].append(s[j][i])

# 1 grafiki
number = len(s)
avg = sum(table['median_house_value']) / number
standart_otkl = math.sqrt(sum((x-avg)**2 for x in table['median_house_value']) / (number - 1))
minimum = min(table['median_house_value'])
maximum = max(table['median_house_value'])

# TODO: kvantili

print(number, avg, standart_otkl, minimum, maximum)

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