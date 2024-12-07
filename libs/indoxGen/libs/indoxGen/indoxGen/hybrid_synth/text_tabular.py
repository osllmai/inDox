import pandas as pd
from typing import List, Dict, Any, Optional
import warnings

from ..synthCore import GenerativeDataSynth

warnings.filterwarnings("ignore", category=RuntimeWarning)


def get_gan_module(prefer_tensor: bool = True):
    """
    Attempt to import the preferred GAN module, falling back to the other if necessary.

    Parameters:
    -----------
    prefer_tensor : bool, optional
        If True, try to import indoxGen_tensor first; otherwise, try indoxGen_torch first.

    Returns:
    --------
    module, bool
        The imported module and a boolean indicating if it's the tensor version.
    """
    first_choice, second_choice = (
        ("indoxGen_tensor", "indoxGen_torch") if prefer_tensor else ("indoxGen_torch", "indoxGen_tensor")
    )

    try:
        module = __import__(first_choice)
        return module, first_choice == "indoxGen_tensor"
    except ImportError:
        try:
            module = __import__(second_choice)
            return module, second_choice == "indoxGen_tensor"
        except ImportError:
            raise ImportError(
                f"Neither `{first_choice}` nor `{second_choice}` is installed. "
                "Please install one of these packages to proceed."
            )


# Define the LLM initialization function
def initialize_llm_synth(
        generator_llm,
        columns: List[str],
        example_data: List[Dict[str, Any]],
        user_instruction: str,
        judge_llm: Optional[Any] = None,
        real_data: Optional[List[Dict[str, Any]]] = None,
        diversity_threshold: float = 0.7,
        max_diversity_failures: int = 20,
        verbose: int = 0
):
    """
    Initializes the LLM-based synthetic text generator setup.

    Parameters:
    -----------
    generator_llm : YourLLMGeneratorClass
        The LLM model used to generate synthetic text.
    judge_llm : YourLLMJudgeClass
        The LLM model used to judge the quality of generated text.
    columns : List[str]
        List of text column names to generate.
    example_data : List[Dict[str, Any]]
        A list of example data records for reference during generation.
    user_instruction : str
        Instructions for the LLM on how to generate the text data.
    diversity_threshold : float, optional
        Threshold for diversity of generated text.
    max_diversity_failures : int, optional
        Maximum allowed diversity failures.
    verbose : int, optional
        Verbosity level for logging and output.

    Returns:
    --------
    SyntheticDataGenerator
        Instance of the initialized synthetic data generator.
    """
    return GenerativeDataSynth(
        generator_llm=generator_llm,
        judge_llm=judge_llm,
        columns=columns,
        example_data=example_data,
        user_instruction=user_instruction,
        diversity_threshold=diversity_threshold,
        max_diversity_failures=max_diversity_failures,
        real_data=real_data,
        verbose=verbose
    )


# Define the GAN initialization function
def initialize_gan_synth(
        input_dim: int,
        generator_layers: List[int],
        discriminator_layers: List[int],
        learning_rate: float,
        beta_1: float,
        beta_2: float,
        batch_size: int,
        epochs: int,
        n_critic: int,
        categorical_columns: List[str],
        mixed_columns: Dict[str, Any],
        integer_columns: List[str],
        data: pd.DataFrame,
        device: str = 'cpu',
        prefer_tensor: bool = True,
        patience: int = 15,
        verbose: int = 1
):
    """
    Initializes the GAN setup for generating numerical data.

    Parameters:
    -----------
    input_dim : int
        Dimension of the input data.
    generator_layers : List[int]
        Sizes of layers in the generator network.
    discriminator_layers : List[int]
        Sizes of layers in the discriminator network.
    learning_rate : float
        Learning rate for training the GAN.
    beta_1 : float
        Beta1 hyperparameter for the Adam optimizer.
    beta_2 : float
        Beta2 hyperparameter for the Adam optimizer.
    batch_size : int
        Batch size for GAN training.
    epochs : int
        Number of epochs to train the GAN.
    n_critic : int
        Number of discriminator updates per generator update.
    categorical_columns : List[str]
        List of categorical columns (if any) in the numerical data.
    mixed_columns : Dict[str, Any]
        Dictionary of mixed column types (if any).
    integer_columns : List[str]
        List of integer columns in the numerical data.
    data : pd.DataFrame
        The dataset containing numerical columns to train the GAN.
    device : str, optional
        Device to run the GAN ('cpu' or 'cuda'). Only used for indoxGen_torch.
    prefer_tensor : bool, optional
        If True, prefer using indoxGen_tensor; if False, prefer indoxGen_torch.
    patience : int, optional
        Number of epochs with no improvement after which training will be stopped.
    verbose : int, optional
        Verbosity mode. 0 = silent, 1 = progress bar, 2 = one line per epoch.

    Returns:
    --------
    TabularGANTrainer
        Instance of the initialized GAN setup.
    """
    gan_module, using_tensor = get_gan_module(prefer_tensor)
    TabularGANConfig = gan_module.TabularGANConfig
    TabularGANTrainer = gan_module.TabularGANTrainer

    # Prepare the configuration
    config = TabularGANConfig(
        input_dim=input_dim,
        generator_layers=generator_layers,
        discriminator_layers=discriminator_layers,
        learning_rate=learning_rate,
        beta_1=beta_1,
        beta_2=beta_2,
        batch_size=batch_size,
        epochs=epochs,
        n_critic=n_critic
    )

    # Initialize the trainer with or without the device parameter
    if using_tensor:
        trainer = TabularGANTrainer(
            config=config,
            categorical_columns=categorical_columns,
            mixed_columns=mixed_columns,
            integer_columns=integer_columns
        )
    else:  # using indoxGen_torch
        trainer = TabularGANTrainer(
            config=config,
            categorical_columns=categorical_columns,
            mixed_columns=mixed_columns,
            integer_columns=integer_columns,
            device=device
        )

    # Train the model with the specified patience and verbosity
    trainer.train(data, patience=patience, verbose=verbose)
    return trainer


# Define the main pipeline class that integrates both the LLM and GAN setups
class TextTabularSynth:
    """
    A class to generate synthetic data combining GAN for numerical data
    and LLM for text data.
    """

    def __init__(self, tabular, text: GenerativeDataSynth):
        """
        Initializes the TextTabularSynth pipeline.

        Parameters:
        -----------
        tabular :
            Instance of the initialized GAN trainer for numerical data.
        text : TextDataGenerator
            Instance of the initialized synthetic text generator.
        """
        self.tabular = tabular
        self.text = text

    def generate(self, num_samples: int) -> pd.DataFrame:
        """
        Generate synthetic data for both numerical and text columns.

        Parameters:
        -----------
        num_samples : int
            Number of synthetic data samples to generate.

        Returns:
        --------
        pd.DataFrame
            A DataFrame containing the combined synthetic data.
        """
        # 1. Generate synthetic numerical data using the GAN
        synthetic_numerical_data = self.tabular.generate_samples(num_samples)

        # 2. Generate synthetic text data using the LLM with numerical context
        synthetic_text_data = self.text.generate_data(num_samples=num_samples)

        # Convert the list of generated text data to a DataFrame
        # synthetic_text_data = pd.DataFrame(synthetic_text_data_list)

        # 3. Combine the numerical and text data into a single DataFrame
        synthetic_numerical_data.reset_index(drop=True, inplace=True)
        synthetic_text_data.reset_index(drop=True, inplace=True)
        synthetic_data = pd.concat([synthetic_numerical_data, synthetic_text_data], axis=1)

        return synthetic_data
