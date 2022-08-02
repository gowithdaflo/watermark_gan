import torch

class MeanMetric:
    def __init__(self):
        self.reset_states()

    def update_state(self, values, sample_weight):
        self.values += sample_weight * values
        self.sample_weights += sample_weight

    def result(self):
        return self.values / self.sample_weights

    def reset_states(self):
        self.sample_weights = 0
        self.values = 0


class Metrics:
    """Class for saving the metrics.

    Parameters
    ----------
        keys: list
            Name of the different metrics to watch (e.g. 'loss', 'mae' etc)
    """

    def __init__(self, keys):
        self.keys = keys

        self.mean_metrics = {}
        for key in self.keys:
            self.mean_metrics[key] = MeanMetric()

    def update_state(self, nsamples, **updates):
        """Update the metrics.

        Parameters
        ----------
            nsamples: int
                Number of samples for which the updates where calculated on.
            updates: dict
                Contains metric updates.
        """
        assert set(updates.keys()) == set(self.keys)
        with torch.no_grad():
            for key, update in updates.items():
                self.mean_metrics[key].update_state(
                    update.cpu(), sample_weight=nsamples
                )

    def write(self, summary_writer, step):
        """Write metrics to summary_writer."""
        for key, val in self.result().items():
            summary_writer.add_scalar(key, val, global_step=step)

    def reset_states(self):
        for key in self.keys:
            self.mean_metrics[key].reset_states()

    def result(self):
        """
        Returns
        -------
            result_dict: dict
                Contains the numpy values of the metrics.
        """
        return {key: self.mean_metrics[key].result().numpy().item() for key in self.keys}

    @property
    def loss(self):
        return self.mean_metrics["loss"].result().numpy().item()
