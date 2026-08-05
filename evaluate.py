import torch
import torch.nn.functional as F
from tqdm import tqdm


def rmse(y_true, y_pred):
    mse = F.mse_loss(y_true.expand_as(y_pred), y_pred, reduction="mean")
    rmse = torch.sqrt(mse)
    return rmse


def crps(y_true, y_pred):
    num_samples = y_pred.shape[0]
    absolute_error = torch.mean(torch.abs(y_pred - y_true), dim=0)

    if num_samples == 1:
        return torch.mean(absolute_error)

    y_pred, _ = torch.sort(y_pred, dim=0)
    empirical_cdf = (
        torch.arange(num_samples, device=y_pred.device).view(-1, 1, 1) / num_samples
    )
    b0 = torch.mean(y_pred, dim=0)
    b1 = torch.mean(y_pred * empirical_cdf, dim=0)

    crps = absolute_error + b0 - 2 * b1
    crps = torch.mean(crps)
    return crps


def min_ade(y_true, y_pred):
    distances = torch.norm(y_pred - y_true.expand_as(y_pred), dim=-1)
    min_distances = torch.min(distances, dim=0).values
    return torch.mean(min_distances)


def min_fde(y_true, y_pred):
    fde = torch.norm(y_pred[:, -1, :] - y_true[:, -1, :], dim=-1)
    return fde.min()


def evaluate(observation_site, model, num_samples, device):
    model.eval()

    with torch.no_grad():
        nll_sum = 0
        rmse_sum = 0
        crps_sum = 0
        min_ade_sum = 0
        min_fde_sum = 0
        count = 0

        for batch in tqdm(observation_site.test_loader, desc="Evaluating"):
            test_input, test_feature, test_type, test_target = (
                batch["input"],
                batch["feature"],
                batch["type"],
                batch["target"],
            )

            # Move data to device
            test_input = test_input.to(device)  # [batch_size, max_num_cars, seq_len, 2]
            test_feature = test_feature.to(
                device
            )  # [batch_size, max_num_cars, seq_len, feature_dim]
            test_type = test_type.to(device)  # [batch_size, max_num_cars]
            test_target = test_target.to(device)  # [batch_size, pred_len, 2]

            # Forward pass
            z_t0, det, embedding = model(test_input, test_target, test_feature, test_type)
            logpz_t0, logpz_t1 = model.log_prob(z_t0, det, embedding)
            nll_sum += -torch.mean(logpz_t1)

            # Sampling
            _, samples, _ = model.sample(
                test_input, test_feature, test_target.shape[1], test_type, num_samples
            )

            # Denormalize targets and samples using THIS sample's location box.
            # test_batch_size==1, so the batch is a single sample/location.
            if "locationId" in batch:
                loc = int(batch["locationId"].view(-1)[0].item())
                denorm = lambda a: observation_site.denormalize_loc(a, loc)
            else:
                denorm = observation_site.denormalize
            test_target = torch.tensor(
                denorm(test_target.cpu().numpy())
            ).to(device)
            samples = torch.tensor(
                denorm(samples.cpu().numpy())
            ).to(device)

            # Compute metrics
            rmse_sum += rmse(test_target, samples)
            crps_sum += crps(test_target, samples)
            min_ade_sum += min_ade(test_target, samples)
            min_fde_sum += min_fde(test_target, samples)
            count += 1

        # Average metrics
        rmse_score = rmse_sum / count
        crps_score = crps_sum / count
        min_ade_score = min_ade_sum / count
        min_fde_score = min_fde_sum / count
        nll_score = nll_sum / count

    return rmse_score, crps_score, min_ade_score, min_fde_score, nll_score
