from typing import Tuple

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


# def hessian_invariants(second_derivatives: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
# 
#     """For input of second derivatives, gives the non-zero (first and second) tensor invariants invar_1 and invar_2
# 
#     Parameters
#     ----------
#     second_derivatives: torch.Tensor
#         Second derivatives in a vector. For a 2D problem this should be of shape [m, 4].
# 
#     Returns
#     -------
#     (invar_1, invar_2): Tuple[torch.Tensor, torch.Tensor]
#         First and second tensor invariants for the second derivatives.
#     """
# 
#     T_xx = second_derivatives[:, 0]
#     T_yy = second_derivatives[:, 3]
#     T_xy = second_derivatives[:, 1]
# 
#     invar_1 = T_xx + T_yy
#     invar_2 = (T_xx * T_yy) - (T_xy * T_xy)
# 
#     return invar_1, invar_2

def hessian_invariants(eigs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

    """For input of second derivatives, gives the non-zero (first and second) tensor invariants invar_1 and invar_2

    Parameters
    ----------
    eigs: torch.Tensor
        eigenvalues from which to calculate invariants.

    Returns
    -------
    (invar_1, invar_2): Tuple[torch.Tensor, torch.Tensor]
        First and second tensor invariants for the second derivatives.
    """

    lambda_1 = eigs[:, 0]
    lambda_2 = eigs[:, 1]

    invar_1 = lambda_1 + lambda_2
    invar_2 = (lambda_1 ** 2) + (lambda_2 ** 2) - (2 * lambda_1 * lambda_2)

    return invar_1, invar_2
