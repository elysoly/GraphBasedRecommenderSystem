import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import csv


def plot_cnf_matrix(matrix):
    [nx, ny] = matrix.shape
    plt.figure()
    tb = plt.table(cellText=matrix, loc=(0, 0), cellLoc='center')

    tc = tb.properties()['child_artists']
    for cell in tc:
        cell.set_height(1 / ny)
        cell.set_width(1 / nx)

    ax = plt.gca()
    ax.set_xticks([])
    ax.set_yticks([])


i = 0
data = np.zeros([100005, 15])
with open('my_data.csv', 'r') as f:
    reader = csv.reader(f)
    for k in reader:
        data[i, :] = [float(j) for j in k]
        i = i+1


np.random.shuffle(data)
tr_percent = 0.76
test_percent = 0.24
n_tr = int (len(data) * 0.6)
tr_data = data[:n_tr]
ts_data = data[n_tr+1:]
target = ts_data[:, 14]
rates = [0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]

# n = 50
estimator = ExtraTreesRegressor(n_estimators=50, criterion='mse')
estimator.fit(tr_data[:, 0:14], tr_data[:, 14])
res = estimator.predict(ts_data[:, 0:14])
result_50 = [round(r * 2) / 2 for r in res]
mae_50 = mean_absolute_error(result_50, target)
diff = abs(target - result_50)
plt.hist(diff, bins=rates, color='green')
plt.title('n = 50 - difference histogram')
plt.grid()

t1 = target * 2
tt1 = np.ndarray.astype(t1, int)
tt2 = [int(2*j) for j in result_50]
cnf_matrix_50 = confusion_matrix(tt1, tt2)


# n = 100
plt.figure()
estimator = ExtraTreesRegressor(n_estimators=100, criterion='mse')
estimator.fit(tr_data[:, 0:14], tr_data[:, 14])
res = estimator.predict(ts_data[:, 0:14])
result_100 = [round(r * 2) / 2 for r in res]
mae_100 = mean_absolute_error(result_100, target)
diff = abs( target - result_100)
plt.hist(diff, bins=rates, color='red')
plt.title('n = 100 - difference histogram')
plt.grid()
tt2 = [int(2*j) for j in result_100]
cnf_matrix_100 = confusion_matrix(tt1, tt2)


# n = 200
plt.figure()
estimator = ExtraTreesRegressor(n_estimators=200, criterion='mse')
estimator.fit(tr_data[:, 0:14], tr_data[:, 14])
res = estimator.predict(ts_data[:, 0:14])
result_200 = [round(r * 2) / 2 for r in res]
mae_200 = mean_absolute_error(result_200, target)
diff = abs(target - result_200)
plt.hist(diff, bins=rates, color='blue')
plt.title('n = 200 - difference histogram')
plt.grid()
tt2 = [int(2*j) for j in result_200]
cnf_matrix_200 = confusion_matrix(tt1, tt2)

f, axarr = plt.subplots(nrows=1,ncols=3)
plt.figure()
diff = abs(target - result_50)
axarr[0].hist(diff, bins=rates, color='green')
axarr[0].set_title('n=50')
axarr[0].grid()
print('n = 50')
for i in rates:
    population = np.count_nonzero(diff == i)
    print('bin '+ str(i)+': '+ str(population))


diff = abs(target - result_100)
axarr[1].hist(diff, bins=rates, color='red')
axarr[1].set_title('n=100')
axarr[1].grid()
print('n = 100')
for i in rates:
    population = np.count_nonzero(diff == i)
    print('bin ' + str(i) +': ' + str(population))


diff = abs(target - result_200)
axarr[2].hist(diff, bins=rates, color='blue')
axarr[2].set_title('n=200')
axarr[2].grid()
print('n = 200')
for i in rates:
    population = np.count_nonzero( diff == i )
    print('bin '+ str(i)+': '+ str(population))


plot_cnf_matrix(cnf_matrix_50)
plt.title('confusion matrix 50')
plot_cnf_matrix(cnf_matrix_100)
plt.title('confusion matrix 100')
plot_cnf_matrix(cnf_matrix_200)
plt.title('confusion matrix 200')


plt.show()
