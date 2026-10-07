
import torch.fft
import numpy as np
import torch.nn.functional as F

def _low_pass_real_signal_mask(size: int, bandwidth: int) -> torch.Tensor:
    """
    Create a low pass filter mask to be applied on a real signal.
    Since the Discrete Fourier Transform of a real signal is Hermitian-symmetric, only returns the
    fft_length / 2 + 1 unique components of the transform.

    Parameters
    ----------
    size
        Size of the mask.
    bandwidth
        Bandwidth of the low pass filter. The higher the bandwidth, the more frequencies are kept.

    Returns
    -------
    mask
        Low pass filter mask of shape (height, width).
    """
    center = (size // 2, size // 2)

    y_grid, x_grid = np.ogrid[:size, :size]
    dist_from_center = np.sqrt((x_grid - center[0])**2 + (y_grid - center[1])**2)

    mask = dist_from_center <= bandwidth
    # un-center the mask
    mask = torch.fft.fftshift(torch.tensor(mask, dtype=torch.float32))
    # keep only the unique components
    mask = mask[:, :size // 2 + 1]

    return mask


def fg(explanations: torch.Tensor, sigma: int = 10) -> torch.Tensor:
    """
    ForGRAD is a method that enhances any attributions explanations (particularly useful on
    gradients based attribution method) by eliminating high frequencies in the explanations.

    Ref. Gradient strikes back: How filtering out high frequencies improves explanations (2023).
         https://arxiv.org/pdf/2307.09591.pdf

    Parameters
    ----------
    explanations
        List of explanations to filter. Explanation should be at least 3D (batch, height, width)
        and should have the same height and width.
    sigma
        Bandwidth of the low pass filter. The higher the sigma, the more frequencies are kept.
        Sigma should be positive and less than image size.
        Default to paper recommendation, 15 for image size 224.

    Returns
    -------
    filtered_explanations
        Explanations low-pass filtered.
    """
    image_size = explanations.shape[1]

    assert image_size == explanations.shape[2], "Explanations should be square."
    assert len(explanations.shape) > 2, "Explanations should be at least 3D (batch, height, width)."
    assert 0 < sigma <= image_size, "Sigma should be positive and less than image size."

    if len(explanations.shape) == 4:
        explanations = explanations.mean(dim=-1)
    explanations = torch.abs(explanations)

    spectrums = torch.fft.rfft2(explanations)

    filter_mask = _low_pass_real_signal_mask(explanations.shape[1], sigma).to(explanations.device)
    filter_mask = filter_mask.to(torch.complex64)
    filtered_spectrums = spectrums * filter_mask

    filtered_explanations = torch.fft.irfft2(filtered_spectrums)
    filtered_explanations = torch.abs(filtered_explanations)

    # resize if odd dimension (cut-off pixel)
    if image_size % 2 == 1:
        filtered_explanations = F.interpolate(filtered_explanations.unsqueeze(1), size=(image_size, image_size), mode="bicubic")

    if len(explanations.shape) == 4 and len(filtered_explanations.shape) == 3:
        filtered_explanations = filtered_explanations.unsqueeze(-1)

    return filtered_explanations
