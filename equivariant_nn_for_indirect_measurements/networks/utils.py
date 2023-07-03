import torch

class AvgPool(torch.nn.Module):
    def forward(self, values_in):
        # values_in (..., n_points_in, channels_in)
        return torch.mean(values_in, dim=-2) # (..., channels_in)

class MaxPool(torch.nn.Module):
    def forward(self, values_in):
        # values_in (..., n_points_in, channels_in)
        return torch.max(values_in, dim=-2) # (..., channels_in)

class BatchNorm(torch.nn.BatchNorm1d):
    """
    Based on https://github.com/ptrblck/pytorch_misc/blob/master/batch_norm_manual.py
    """
    def forward(self, input):
        exponential_average_factor = 0.0

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        # calculate running estimates
        dims = list(range(len(input.shape)-1))
        if self.training:
            mean = input.mean(dim=dims)
            # use biased var in train
            var = input.var(dims, unbiased=False)
            n = input.numel() / input.shape[-1]
            with torch.no_grad():
                self.running_mean = exponential_average_factor * mean\
                    + (1 - exponential_average_factor) * self.running_mean
                # update running_var with unbiased var
                self.running_var = exponential_average_factor * var * n / (n - 1)\
                    + (1 - exponential_average_factor) * self.running_var
        else:
            mean = self.running_mean
            var = self.running_var

        input = (input - mean) / (torch.sqrt(var + self.eps))
        if self.affine:
            input = input * self.weight + self.bias

        return input