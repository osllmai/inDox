from tensorflow import keras
import numpy as np


class GANMonitor(keras.callbacks.Callback):
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
    """

    def __init__(self, patience=10):
        """
        Initializes the GANMonitor with the specified patience.

        Parameters:
        -----------
        patience : int
            Number of epochs with no improvement after which training will be stopped.
        """
        super(GANMonitor, self).__init__()
        self.patience = patience
        self.best_g_loss = np.inf
        self.wait = 0

    def on_epoch_end(self, epoch, logs=None):
        """
        Callback function called at the end of each epoch to check for improvement in generator loss.

        Parameters:
        -----------
        epoch : int
            The current epoch number.
        logs : dict, optional
            Dictionary of logs containing training metrics, including generator loss.
        """
        current_g_loss = logs.get('g_loss')
        if current_g_loss is None:
            return

        if current_g_loss < self.best_g_loss:
            self.best_g_loss = current_g_loss
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.model.stop_training = True
                print(f"\nEarly stopping triggered. Generator loss did not improve for {self.patience} epochs.")
