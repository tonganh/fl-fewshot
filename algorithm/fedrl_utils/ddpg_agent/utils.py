
from IPython.display import clear_output
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import torch
import os


def plot(frame_idx, rewards):
    now = datetime.now()
    date_time = now.strftime("%m/%d/%Y, %H:%M:%S")
    clear_output(True)
    plt.figure(figsize=(20,5))
    plt.subplot(131)
    plt.title('frame %s. reward: %s' % (frame_idx, rewards[-1]))
    plt.plot(rewards)
    if not os.path.exists('./log/images/'):
        os.makedirs('./log/images/')
    plt.savefig('./log/images/'+date_time)
    plt.show()


def get_state(losses, n_samples, n_epochs):
    # print("Losses: ", len(losses), losses)
    # print("N_samples: ", len(n_samples), n_samples)
    # print("N_epochs: ", len(n_epochs), n_epochs)

    retval = torch.Tensor(losses + n_samples + n_epochs).double()
    return retval.flatten()


def get_reward(losses, M_matrix, beta=0.45):
    # beta = 0.45
    losses = np.asarray(losses)
    # return - beta * np.mean(losses) - (1 - beta) * np.std(losses)
    return - np.mean(losses) - (losses.max() - losses.min()) + 0.05 * np.sum(M_matrix)/2




