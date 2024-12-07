import torch
import numpy as np

class GANMonitor:
    """
    GANMonitor class for early stopping based on generator loss.

    Attributes:
    -----------
    patience : int
        Number of epochs with no improvement after which training will be stopped.
    best_g_loss : float
        The best generator loss observed during training.
    wait : int
        Counter for the number of epochs with no improvement.
    stopped_epoch : int
        The epoch at which training was stopped.
    """

    def __init__(self, patience=10):
        """
        Initializes the GANMonitor with the specified patience.

        Parameters:
        -----------
        patience : int
            Number of epochs with no improvement after which training will be stopped.
        """
        self.patience = patience
        self.best_g_loss = np.inf
        self.wait = 0
        self.stopped_epoch = 0
        self.early_stop = False

    def on_epoch_end(self, epoch, g_loss):
        """
        Callback function called at the end of each epoch to check for improvement in generator loss.

        Parameters:
        -----------
        epoch : int
            The current epoch number.
        g_loss : float
            The current generator loss for the epoch.
        """
        if g_loss < self.best_g_loss:
            self.best_g_loss = g_loss
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.early_stop = True
                self.stopped_epoch = epoch
                print(f"\nEarly stopping triggered. Generator loss did not improve for {self.patience} epochs.")

    def reset(self):
        """
        Resets the monitor for a new training session.
        """
        self.best_g_loss = np.inf
        self.wait = 0
        self.stopped_epoch = 0
        self.early_stop = False
