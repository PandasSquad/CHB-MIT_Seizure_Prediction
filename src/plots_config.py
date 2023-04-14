"""This module contains the configuration for the plots."""


from types import ModuleType as Module


def plots_config(plt) -> None:
    """Configure the plots.

    :param plt: the matplotlib module
    :return: None
    """

    plt.style.use("ggplot")
    plt.rcParams["text.color"] = "black"
    plt.rcParams["axes.labelcolor"] = "black"
    plt.rcParams["xtick.color"] = "black"
    plt.rcParams["ytick.color"] = "black"
    plt.rcParams["figure.figsize"] = (10, 6)
    plt.rcParams["font.size"] = 12
    plt.rcParams["axes.grid"] = True

    return plt
