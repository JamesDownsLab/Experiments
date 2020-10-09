import numpy as np
import matplotlib.pyplot as plt
import filehandling

direc = "/media/data/Data/January2020/RecordFluctuatingInterface/Quick/first_frames/"

x = np.loadtxt(direc+'x.txt')
hs = np.loadtxt(direc+'hs.txt')
plt.plot(x, hs[0])
print("dx = ", x[1]-x[0])


L = x[-1]
h = hs[0]
N = len(h)
print("N = ", N)

hk = np.fft.rfft(h)
group = 10
k = np.fft.rfftfreq(len(h), x[1]-x[0])
# plt.loglog(freq, sp, '.')


hks = [np.abs(np.fft.rfft(h))**2 for h in hs]
hk_mean = np.mean(hks, axis=0)[1:]
hk_err = np.std(hks, axis=0)[1:]
freq = np.fft.rfftfreq(len(x), x[1]-x[0])
k = 2*np.pi * freq[1:]
wavelengths = 2*np.pi / k
fig, ax = plt.subplots()
plt.plot(wavelengths, hk_mean)
ax.set_xscale('log')
ax.set_yscale('log')

plt.figure()
k_log = np.log(k)
hk_mean_log = np.log(hk_mean)
hk_err_log = np.log(hk_err)




p, cov = np.polyfit(k_log, hk_mean_log, 1, cov=True, w=hk_err_log)



p1 = np.poly1d(p)
plt.plot(k_log, hk_mean_log, 'x')
plt.plot(k_log, p1(k_log))
plt.legend(['Data', 'Fit with gradient ${:.2f} \pm {:.2f}$'.format(p[0],
                                                        cov[0][0] ** 0.5)])
plt.xlabel('log($k [$mm$^{-1}]$)')
plt.ylabel('$log( < |\delta h_k|^2 > L [$mm$^3] $)')
plt.show()

np.savetxt(direc + 'k_data.txt', k_log)
np.savetxt(direc + 'hk_mean.txt', hk_mean_log)
np.savetxt(direc + 'hk_fit.txt', p1(k_log))
with open(direc + 'fit_results.txt', 'w') as f:
    f.writelines(['m, dm', str(p[0]) + ', ' + str(cov[0][0]**0.5)])


