
import torch
import pytorch_lightning as pl
from pytorch_lightning.utilities import DeviceType, parsing, rank_zero_warn


#%%
class MyTrainer(pl.Trainer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def run_evaluation(self, on_epoch: bool = False):
        if not (self.evaluating or self.sanity_checking):
            rank_zero_warn(
                f"`trainer.run_evaluation()` was called but the running stage is set to {self.state.stage}."
                " This should not happen normally. Setting it to `RunningStage.VALIDATING`", RuntimeWarning
            )
            self.validating = True

        # prepare dataloaders
        dataloaders, max_batches = self.evaluation_loop.get_evaluation_dataloaders()

        # check if we want to skip this evaluation
        if self.evaluation_loop.should_skip_evaluation(max_batches):
            return [], []

        # enable eval mode + no grads
        self.evaluation_loop.on_evaluation_model_eval()
        # ref model
        model = self.lightning_module
        model.zero_grad()
        torch.set_grad_enabled(False)

        # hook
        self.evaluation_loop.on_evaluation_start()

        # set up the eval loop
        self.evaluation_loop.setup(max_batches, dataloaders)

        # hook
        self.evaluation_loop.on_evaluation_epoch_start()

        # run validation/testing
        for dataloader_idx, dataloader in enumerate(dataloaders):
            # bookkeeping
            dl_outputs = []
            dataloader = self.accelerator.process_dataloader(dataloader)
            dl_max_batches = self.evaluation_loop.max_batches[dataloader_idx]

            for batch_idx, batch in enumerate(dataloader):
                if batch is None:
                    continue

                # stop short when running on limited batches
                if batch_idx >= dl_max_batches:
                    break

                # hook
                self.evaluation_loop.on_evaluation_batch_start(batch, batch_idx, dataloader_idx)

                # lightning module methods
                with self.profiler.profile("evaluation_step_and_end"):
                    output = self.evaluation_loop.evaluation_step(batch, batch_idx, dataloader_idx)
                    output = self.evaluation_loop.evaluation_step_end(output)

                # hook + store predictions
                self.evaluation_loop.on_evaluation_batch_end(output, batch, batch_idx, dataloader_idx)

                # log batch metrics
                self.evaluation_loop.log_evaluation_step_metrics(batch_idx)

                # track epoch level outputs
                dl_outputs = self.track_output_for_epoch_end(dl_outputs, output)

            # store batch level output per dataloader
            if self.evaluation_loop.should_track_batch_outputs_for_epoch_end:
                self.evaluation_loop.outputs.append(dl_outputs)

        outputs = self.evaluation_loop.outputs

        # reset outputs
        self.evaluation_loop.outputs = []

        # with a single dataloader don't pass a 2D list
        if len(outputs) > 0 and self.evaluation_loop.num_dataloaders == 1:
            outputs = outputs[0]

        # lightning module method
        self.evaluation_loop.evaluation_epoch_end(outputs)

        # hook
        self.evaluation_loop.on_evaluation_epoch_end()

        # update epoch-level lr_schedulers
        if on_epoch:
            self.optimizer_connector.update_learning_rates(
                interval='epoch',
                opt_indices=[
                    opt_idx
                    for opt_idx, _ in self.train_loop.get_optimizers_iterable(batch_idx=(
                        self.total_batch_idx - 1
                    ))  # Select the optimizers which were used in the last batch of the epoch
                ],
            )

        # hook
        self.evaluation_loop.on_evaluation_end()

        # log epoch metrics
        eval_loop_results = self.logger_connector.get_evaluate_epoch_results()

        # save predictions to disk
        self.evaluation_loop.predictions.to_disk()

        # enable train mode again
        self.evaluation_loop.on_evaluation_model_train()

        # reset cached results
        self.logger_connector.reset()

        torch.set_grad_enabled(True)

        return outputs

    def run_evaluate(self):
        if not self.is_global_zero and self.progress_bar_callback is not None:
            self.progress_bar_callback.disable()

        assert self.evaluating
        with self.profiler.profile(f"run_{self.state.stage}_evaluation"):
            eval_loop_results = self.run_evaluation()

        outputs = {k:[] for k in eval_loop_results[0].keys()}
        # remove the tensors from the eval results
        for i, result in enumerate(eval_loop_results):
            if isinstance(result, dict):
                for k, v in result.items():
                    if isinstance(v, torch.Tensor):
                        outputs[k].append(v.cpu())
        outputs = {k:torch.vstack(v).detach().float() for k,v in outputs.items()}
        return outputs