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

number = len(s)
avg = sum(table['median_house_value']) / number

standart_otkl = ...
minimum = min(table['median_house_value'])
maximum = max(table['median_house_value'])

print(number, avg, minimum, maximum)