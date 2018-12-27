import numpy as np
import matplotlib.pyplot as plt
import sys

if len(sys.argv) <= 1:
    print("python3 ShowHistory.py test87.npy")
    sys.exit()

a = np.load(sys.argv[1])
start_from = 3

plt.subplot(1, 2, 1)
plt.title('acc')
plt.plot([i[-2]['acc'] for i in a[start_from:]], label='train')
plt.plot([i[-1]['acc'] for i in a[start_from:]], label='valid')

plt.subplot(1, 2, 2)
plt.title('f1')
plt.plot([i[-2]['f1'] for i in a[start_from:]], label='train')
plt.plot([i[-1]['f1'] for i in a[start_from:]], label='valid')

plt.show()
