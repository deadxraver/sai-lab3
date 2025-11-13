s = ""
global header
with open("california_housing_train.csv") as f:
	s = f.readlines()
	header = s.pop(0).split(';')
	for i in range(len(s)):
		s[i] = list(map(float, s[i].split(';')))

number = len(s)
avg = sum(row[-1] for row in s) / number

standart_otkl = ...
minimum = min(row[-1] for row in s)
maximum = max(row[-1] for row in s)

print(number, avg, minimum, maximum)