import torch
import torch.nn as nn
import torch.nn.functional as F
from torchdiffeq import odeint_adjoint
from model.layers.MovingBatchNorm import MovingBatchNorm1d
from model.layers.SquashLinear import ConcatSquashLinear


class ODEFunc(nn.Module):
    def __init__(self, input_dim, condition_dim, hidden_dims):
        super(ODEFunc, self).__init__()

        self.sampling_frequency = 1
        self.epsilon = None
        self.frame_indices = None

        temporal_context_dim = 2
        dim_list = [input_dim] + list(hidden_dims) + [input_dim]

        layers = []
        for i in range(len(dim_list) - 1):
            layers.append(
                ConcatSquashLinear(
                    dim_list[i], dim_list[i + 1], condition_dim + temporal_context_dim
                )
            )
        self.layers = nn.ModuleList(layers)

    def _z_dot(self, t, z, condition):
        condition = condition.unsqueeze(1).expand(-1, z.shape[1], -1)

        # Handle both scaler t and batch t
        if t.dim() == 0:
            time_encoding = t.expand(z.shape[0], z.shape[1], 1)
        else:
            # (batch, seq_len, 1)
            time_encoding = t.view(-1, 1, 1).expand(z.shape[0], z.shape[1], 1)

        if self.frame_indices is None:
            positional_encoding = torch.cumsum(
                torch.ones_like(z)[:, :, 0], 1
            ).unsqueeze(-1)
            positional_encoding = positional_encoding / self.sampling_frequency
        else:
            # frame_indices: shape [batch, num_frames]
            positional_encoding = self.frame_indices.unsqueeze(-1)

        context = torch.cat([positional_encoding, time_encoding, condition], dim=-1)

        z_dot = z
        for l, layer in enumerate(self.layers):
            z_dot = layer(context, z_dot)
            if l < len(self.layers) - 1:
                z_dot = F.tanh(z_dot)
        return z_dot

    def _gaussian_noise(self, z):
        noise = torch.randn_like(z).to(z)
        self.epsilon = noise

    def _rademacher_noise(self, z):
        random_bits = torch.randint(0, 2, z.shape, device=z.device, dtype=z.dtype)
        noise = 2 * random_bits - 1
        self.epsilon = noise

    def _hutchinson_estimator(self, z_dot, z):
        e = self.epsilon
        z_dot_e = torch.autograd.grad(z_dot, z, grad_outputs=e, create_graph=True)[0]
        trace_estimate = torch.sum(z_dot_e * e, dim=-1)
        return trace_estimate

    def _jacobian_trace_joint(self, z_dot, z):
        return self._hutchinson_estimator(z_dot, z)

    def _jacobian_trace(self, z_dot, z):
        batch_size, seq_len, dim = z.shape
        trace = torch.zeros(batch_size, seq_len, device=z.device)
        for i in range(dim):
            trace += torch.autograd.grad(z_dot[:, :, i].sum(), z, create_graph=True)[0][
                :, :, i
            ]
        return trace

    def forward(self, t, states):
        z = states[0]
        condition = states[2]

        with torch.set_grad_enabled(True):
            t.requires_grad_(True)
            for state in states:
                state.requires_grad_(True)
            z_dot = self._z_dot(t, z, condition)
            divergence = self._jacobian_trace(z_dot, z)

        return z_dot, -divergence, torch.zeros_like(condition).requires_grad_(True)


class CNF(torch.nn.Module):
    def __init__(self, input_dim, condition_dim, hidden_dims):
        super(CNF, self).__init__()
        self.time_derivative = ODEFunc(input_dim, condition_dim, hidden_dims)
        self.condition_norm = nn.LayerNorm(condition_dim)
        self.n1 = MovingBatchNorm1d(input_dim)
        self.n2 = MovingBatchNorm1d(input_dim)

    def forward(
        self,
        z,
        condition,
        delta_logpz=None,
        integration_times=None,
        reverse=False,
        sampling_frequency=1,
        frame_indices=None,
    ):
        if delta_logpz is None:
            delta_logpz = torch.zeros(z.shape[0], z.shape[1], 1).to(z)
        if integration_times is None:
            integration_times = torch.tensor([0.0, 1.0]).to(z)
        if reverse:
            integration_times = torch.flip(integration_times, [0])

        self.time_derivative.sampling_frequency = sampling_frequency
        self.time_derivative.frame_indices = frame_indices
        condition = self.condition_norm(condition)

        z, delta_logpz = (
            self.n1(z, delta_logpz, reverse)
            if not reverse
            else self.n2(z, delta_logpz, reverse)
        )
        state = odeint_adjoint(
            self.time_derivative,
            (z, delta_logpz, condition),
            integration_times,
            method="dopri5",
            # method="rk4",
            atol=1e-5,
            rtol=1e-5,
        )
        z, delta_logpz, condition = (
            tuple(s[1] for s in state) if len(integration_times) == 2 else state
        )
        z, delta_logpz = (
            self.n2(z, delta_logpz, reverse)
            if not reverse
            else self.n1(z, delta_logpz, reverse)
        )
        return z, delta_logpz.squeeze(-1)
