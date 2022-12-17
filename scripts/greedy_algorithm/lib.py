import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F


def log_reconstruction_loss(data, output, sigma, mask, device):
    """Expectation term in the VAE loss function (see above).
    :param data: dataset
    :param output: output of the decoder
    :param sigma: diagonal of the decoder distribution covariance matrix
    :param mask: mask to apply to data and reconstructions
    :returns: (torch tensor)
    """
    recon = output * mask
    test = data.to(device) * mask
    test_minus_mu = test - recon
    log_p_x_given_z = - (test_minus_mu * test_minus_mu / (2 * sigma * sigma) ).sum(1)
    return -1 * log_p_x_given_z


def average_VAE_loss(mu_code1, log_sigma_code1, data, output, sigma, mask, device):
    """Compute average VAE loss over a dataset.
    :param mu_code1: latent mean embeddings
    :param log_sigma_code1: latent covariance matrix diagonal embeddings
    :param data: images
    :param output: output of net on the dataset
    :param sigma: diagonal of the decoder distribution covariance matrix
    :param mask: mask to apply to data and reconstructions
    :param device: device to compute on
    :returns: (torch tensor) average VAE loss over images
    """
    # negative KL term in loss function (see explanation above)
    negative_KL = (torch.ones_like(mu_code1) + 2 * log_sigma_code1 - mu_code1 * mu_code1 - torch.exp(
      2 * log_sigma_code1)).sum(1) / 2

    # reconstruction loss term
    log_reconst_loss = log_reconstruction_loss(data, output, sigma, mask, device)
    # optimize average loss value over data
    nkl= negative_KL.mean()
    lgr = log_reconst_loss.mean()
    average_loss = lgr - nkl

    return average_loss, lgr


def find_code(data, net, shift_transform, mu_code1, log_sigma_code1, shift_codes, cls_label, mask, sigma, device, iternum=50, lr=0.01, shifts=False):
    """Optimize the latent space embedding for data of a given class.  The
    parameters of the model are fixed and we run gradient descent through the
    embeddings to optimize the VAE loss.
    :param data: training data, all of the same class
    :param net: model to use, with fixed parameters
    :param shift_transform: shift transformer to apply to images after decoding
    :param mu_code1: latent space embedding of the mean
    :param log_sigma_code1: latent space log square root of the diagonal of the
        covariance matrix
    :param shift_codes: parameters that shift images after they have been decoded
    :param cls_label: integer - class label of the training data being supplied
    :param mask: mask to apply to the reconstructions when computing loss
    :param sigma: diagonal of the decoder distribution covariance matrix
    :param device: device to execute on
    :param iternum: number of epochs used in optimization
    :param lr: learning rate to use for optimizing mu and sigma values
    :param shifts: whether to do shifting of images after decoding and
        backpropogate through shift codes
    :returns: optimized mu and sigma embeddings
    """

    # require gradients so we can backpropogate through these
    mu_code1.requires_grad_(True)
    log_sigma_code1.requires_grad_(True)

    # optimizers
    optimizer_mu = torch.optim.Adam(
        [mu_code1],
        lr=lr
    )
    optimizer_log_sigma = torch.optim.Adam(
        [log_sigma_code1],
        lr=lr
    )

    if shifts:
        # learning the shifts of images
        shift_codes.requires_grad_(True)
        optimizer_shifts = torch.optim.Adam(
            [shift_codes],
            lr=lr
        )

    for j in range(iternum):
        # zero out gradients
        optimizer_mu.zero_grad()
        optimizer_log_sigma.zero_grad()

        if shifts:
            optimizer_shifts.zero_grad()

        # sample from latent space
        eps = torch.randn_like(mu_code1)
        z = mu_code1 + torch.exp(log_sigma_code1) * eps
        # decode using pretrained decoder - possibly doing shifting
        output, _ = net(z, cls_label)

        # TODO this was rendered obsolete
        # if shifts:
        #     # apply shift to decoded images and reflatten
        #     output = output.view(-1, 50, 50, 3)
        #     output = output.permute(0, 3, 1, 2)
        #     output = shift_transform(output, shift_codes)
        #     # invert the reshaping
        #     output = output.permute(0, 2, 3, 1)
        #     output = torch.flatten(output, start_dim=1)

        # compute loss - shift only effects reconstruction, is not regularized
        # like the other latent codes
        average_loss, mse = average_VAE_loss(
            mu_code1,
            log_sigma_code1,
            data,
            output,
            sigma,
            mask,
            device
        )
        average_loss.backward(retain_graph=True)
        optimizer_mu.step()
        optimizer_log_sigma.step()

        if shifts:
            optimizer_shifts.step()

    return mu_code1, log_sigma_code1, shift_codes



