import numpy as np
import matplotlib.pyplot as plt
import sys

if len(sys.argv) <= 1:
    print("python3 ShowHistory.py test87.npy")
    sys.exit()

a = np.load(sys.argv[1])
for i in range(len(a)):
    if len(a[i]):
        start_from = i
        break

x = range(start_from, len(a))
y = a[start_from:]
plt.subplot(1, 3, 1)
plt.title('acc')
plt.plot(x, [i[-2]['acc'] for i in y], 'o-', label='train')
plt.plot(x, [i[-1]['acc'] for i in y], 'o-', label='valid')
plt.legend()

plt.subplot(1, 3, 2)
plt.title('f1')
plt.plot(x, [i[-2]['f1'] for i in y], 'o-', label='train')
plt.plot(x, [i[-1]['f1'] for i in y], 'o-', label='valid')
plt.legend()

plt.subplot(1, 3, 3)
plt.title('loss')
plt.plot(x, [i[-2]['loss'] for i in y], 'o-', label='train')
plt.plot(x, [i[-1]['loss'] for i in y], 'o-', label='valid')
plt.legend()

plt.show()
