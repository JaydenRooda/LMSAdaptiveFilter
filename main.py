import numpy as np
import xlsxwriter
import matplotlib.pyplot as plt

N = 100                     # total number of data points
n = np.arange(0, N, 1)      # [0,..., N-1] vector

x = np.full(N, 5.0)         # measured protein x concentration in uM (vector), constant measured x protein concentration (= 5 uM)
y = np.full(N, 0.0)         # measured protein y concentration in uM (vector)
a1 = np.full(N, 0.2)        # constant a1 parameter (vector) of 0.2
b1 = np.full(N, 0.9)        # constant b1 parameter (vector) of 0.9
m = 0                       # mean of Gaussian noise
sd = 0.2                    # standard deviation of Gaussian noise
x[0] = x[0] + np.random.normal(m, sd)       # initial measured x protein concentration (simulated)
y[0] = 0 + np.random.normal(m, sd)          # initial measured y protein concentration (simulated)

yHat = np.full(N, 0.0)      # estimated protein y concentration
a1Hat = np.full(N, 0.0)     # estimated a1 parameter (vector), initial estimated a1 parameter (= 0 uM)
b1Hat = np.full(N, 0.0)     # estimated b1 parameter (vector), initial estimated b1 parameter (= 0 uM)
e = np.full(N, 0.0)         # y protein estimation error in uM (vector) (= y - yHat)
u = 0.001                   # step size (mu)

def simulate_estimate():
    for i in range(1, N - 1):

        # simulated experiment
        x[i] = x[i - 1] + np.random.normal(m, sd)                               # measured x protein concentration (simulated)
        y[i] = a1[i] * x[i - 1] + b1[i] * y[i - 1] + np.random.normal(m, sd)    # measured y protein concentration (simulated)

        # adaptive system identification (LMS)
        yHat[i] = a1Hat[i] * x[i - 1] + b1Hat[i] * y[i - 1]     # estimated protein y concentration
        e[i] = y[i] - yHat[i]                                   # y protein estimation error
        a1Hat[i + 1] = a1Hat[i] + u * x[i - 1] * e[i]           # estimated a1 parameter
        b1Hat[i + 1] = b1Hat[i] + u * y[i - 1] * e[i]           # estimated b1 parameter

    x[N - 1] = x[N - 2] + np.random.normal(m, sd)                                       # last measured y protein concentration (simulated)
    y[N - 1] = a1[N - 1] * x[N - 2] + b1[N - 1] * y[N - 2] + np.random.normal(m, sd)    # last measured y protein concentration (simulated)
    yHat[N - 1] = a1Hat[N - 1] * x[N - 2] + b1Hat[N - 1] * y[N - 2]                     # last estimated protein y concentration
    e[N - 1] = y[N - 1] - yHat[N - 1]                                                   # last y protein estimation error


def save_data(filename):
    # Save the data into an Excel spreadsheet
    workbook = xlsxwriter.Workbook(filename)
    worksheet = workbook.add_worksheet()

    worksheet.write('A1', 'time')
    worksheet.write('B1', 'x protein')
    worksheet.write('C1', 'y protein')
    worksheet.write('D1', 'estimated y protein')
    worksheet.write('E1', 'estimated a1')
    worksheet.write('F1', 'estimated b1')
    worksheet.write('G1', 'y protein estimation error')
    for i in range(0, N):
        worksheet.write('A'+str(i+2), i+1)
        worksheet.write('B'+str(i+2), x[i])
        worksheet.write('C'+str(i+2), y[i])
        worksheet.write('D'+str(i+2), yHat[i])
        worksheet.write('E'+str(i+2), a1Hat[i])
        worksheet.write('F'+str(i+2), b1Hat[i])
        worksheet.write('G'+str(i+2), e[i])

    workbook.close()

def plot_data(plotname):
    # Print data on a plot
    plt.plot(n, x, 'tab:blue', label='x protein')
    plt.plot(n, y, 'tab:red', label='y protein')
    plt.plot(n, yHat, 'tab:green', label='estimated y protein')
    plt.plot(n, a1Hat, 'tab:purple', label='estimated a1')
    plt.plot(n, b1Hat, 'tab:cyan', label='estimated b1')
    plt.plot(n, e, 'tab:orange', label='y protein estimation error')
    plt.xlabel('time (i)')
    plt.ylabel('protein concentration (uM)')
    plt.legend(loc='upper left')
    plt.axis([0, N, -3, 25])
    plt.grid(True)
    plt.savefig(plotname)

def main():
    simulate_estimate()
    save_data('LMSAdaptiveFilter.xlsx')
    plot_data('LMSAdaptiveFilter.png')


if __name__ == "__main__":
    main()