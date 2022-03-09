import torch


def derivative_magnitude(d_tensor: torch.Tensor) -> torch.Tensor:

    """ Returns the magnitude of a vector of derivatives, taking a tensor of derivative vectors as input.

    Parameters
    ----------
    d_tensor: torch.Tensor
        Tensor containing differenct components of derivative as columns and data points as rows.

    Returns
    -------
    d_mag: torch.Tensor
        Single column tensor containing magnitudes of derivative vectors.

    """

    d_mag = torch.zeros((d_tensor.shape[0], 1))

    for i in range(d_tensor.shape[0]):
        d_mag[i] = torch.linalg.vector_norm(d_tensor[i], 2)

    return d_mag


def hessian_eigenvalues(second_derivatives: torch.Tensor) -> torch.Tensor:

    """ For input of second derivatives, assumes it is convertible into a square matrix and returns the eigenvalues.

    Parameters
    ----------
    second_derivatives: torch.Tensor
        Second derivatives in a vector. For a 2D problem this should be of shape [m, 4].

    Returns
    -------
    h_eigs: torch.Tensor
        Eigenvalues of the hessian.
    """

    # initialise 2x2 hessian - will have 2 eigenvalues
    h_eigs = torch.zeros((second_derivatives.shape[0], 2))

    for i in range(second_derivatives.shape[0]):
        hessian = second_derivatives[i].resize_(2, 2)
        h_eigs[i] = torch.linalg.eigvals(hessian)

    return h_eigs
