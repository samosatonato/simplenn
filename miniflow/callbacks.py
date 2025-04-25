# TODO

import numpy as np

class Callback:

    pass


class EarlyStopping(Callback):
    def __init__(self, monitor='val_loss', patience=5, min_delta=0, mode='min', baseline=None, restore_best_weights=False):
        super().__init__()
        self.monitor = monitor
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.baseline = baseline
        self.restore_best_weights = restore_best_weights

        self.wait = 0
        self.stopped_epoch = 0
        self.best_score = np.Inf if mode == 'min' else -np.Inf
        self.best_weights = None
        self.model = None

    def on_train_begin(self, model):
        self.model = model
        self.wait = 0
        self.stopped_epoch = 0
        self.best_score = np.Inf if self.mode == 'min' else -np.Inf
        self.best_weights = None

    def on_epoch_end(self, epoch, current_score):
        if self.model is None:
             raise RuntimeError("Callback needs model reference.")

        if self.mode == 'min':
            score_check = self.best_score - self.min_delta
            improved = current_score < score_check
        else:
            score_check = self.best_score + self.min_delta
            improved = current_score > score_check

        if self.baseline is not None:
             if self.mode == 'min' and current_score > self.baseline: return False # Not stopping
             if self.mode == 'max' and current_score < self.baseline: return False # Not stopping


        if improved:
            self.best_score = current_score
            self.wait = 0
            if self.restore_best_weights:

                self.best_weights = [np.copy(l.get_weights()) for l in self.model.layers], \
                                   [np.copy(l.get_biases()) for l in self.model.layers]
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch

                print(f"\nEpoch {epoch+1}: Early stopping triggered.")

                if self.restore_best_weights and self.best_weights is not None:

                    print("Restoring model weights from the end of the best epoch.")

                    weights, biases = self.best_weights
                    for i, layer in enumerate(self.model.layers):
                        layer.set_weights(weights[i])
                        layer.set_biases(biases[i])

                return True
            
        return False

class LearningRateScheduler(Callback):

    pass


class ModelCheckpoint(Callback):

    pass