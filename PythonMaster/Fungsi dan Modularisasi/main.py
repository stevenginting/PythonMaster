import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('mnist_train.csv')

# pisahkan label dan fitur
labels = df.iloc[:, 0].values
data = df.iloc[:, 1:].values.astype(np.float32)



fig, axs = plt.subplots(3, 4, figsize=(10, 6))

for ax in axs.flatten():
    randimg2show = np.random.randint(0, high=data.shape[0])

    # create the image (reshape inget)
    img = np.reshape(data[randimg2show, :], (28, 28))
    ax.imshow(img, cmap='gray')

    # title
    ax.set_title('The number %i'%labels[randimg2show])

plt.suptitle('How we see the data', fontsize=20)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

# show a few random digits
fig,axs = plt.subplots(3,4,figsize=(10,6))

for ax in axs.flatten():
  # pick a random image
  randimg2show = np.random.randint(0,high=data.shape[0])

  # create the image
  ax.plot(data[randimg2show,:],'ko')

  # title
  ax.set_title('The number %i'%labels[randimg2show])

plt.suptitle('How the FFN model sees the data',fontsize=20)
plt.tight_layout(rect=[0,0,1,.95])
plt.show()
