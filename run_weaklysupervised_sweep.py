import torch
from torch import optim
from torch.utils import data as torch_data

import timeit

import wandb
import numpy as np

from utils import networks, datasets, loss_functions, evaluation, experiment_manager, parsers


if __name__ == '__main__':
    args = parsers.sweep_argument_parser().parse_known_args()[0]
    cfg = experiment_manager.setup_cfg(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('=== Runnning on device: p', device)

    def run_training(sweep_cfg=None):

        torch.manual_seed(cfg.SEED)
        np.random.seed(cfg.SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        with wandb.init(config=sweep_cfg):
            sweep_cfg = wandb.config

            cfg.MODEL.OUTPUT_PATH = cfg.PATHS.OUTPUT
            net = networks.PopulationDualTaskNet(cfg.MODEL)
            net.to(device)

            pretraining = cfg.CHANGE_DETECTION.PRETRAINING
            if pretraining.ENABLED:
                net.load_pretrained_encoder(pretraining.CFG_NAME, cfg.PATHS.OUTPUT, device, verbose=True)
                if pretraining.FREEZE_ENCODER:
                    net.freeze_encoder()

            optimizer = optim.AdamW(net.parameters(), lr=sweep_cfg.lr, weight_decay=0.01)
            criterion = loss_functions.get_criterion(cfg.MODEL.LOSS_TYPE)

            # unpacking cfg
            epochs = cfg.TRAINER.EPOCHS
            train_units = datasets.get_units(cfg.PATHS.DATASET, 'train')
            steps_per_epoch = len(train_units)

            # tracking variables
            global_step = epoch_float = 0

            # early stopping
            best_rmse_change_val, trigger_times = None, 0
            stop_training = False

            _ = evaluation.model_change_evaluation_units(net, cfg, 'train', epoch_float)
            _ = evaluation.model_change_evaluation_units(net, cfg, 'val', epoch_float)

            for epoch in range(1, epochs + 1):
                print(f'Starting epoch {epoch}/{epochs}.')

                start = timeit.default_timer()
                loss_set = []

                np.random.shuffle(train_units)
                for i_unit, train_unit in enumerate(train_units):
                    dataset = datasets.BitemporalCensusUnitDataset(cfg=cfg, unit_nr=train_unit)
                    dataloader_kwargs = {
                        'batch_size': len(dataset),
                        'num_workers': 0 if cfg.DEBUG else cfg.DATALOADER.NUM_WORKER,
                        'shuffle': cfg.DATALOADER.SHUFFLE,
                        'drop_last': False,
                        'pin_memory': True,
                    }
                    dataloader = torch_data.DataLoader(dataset, **dataloader_kwargs)
                    batch = next(iter(dataloader))

                    net.train()
                    optimizer.zero_grad()

                    x_t1 = batch['x_t1'].to(device)
                    x_t2 = batch['x_t2'].to(device)
                    pred_change, pred_pop_t1, pred_pop_t2 = net(x_t1, x_t2)
                    # need to be detached for backprop to work
                    pred_pop_t1.detach(), pred_pop_t2.detach()
                    pred_change = torch.sum(pred_change, dim=0)

                    y_change, *_ = dataset.get_unit_labels()
                    loss = criterion(pred_change, y_change.to(device).float())
                    loss.backward()
                    optimizer.step()

                    loss_set.append(loss.item())

                    global_step += 1
                    epoch_float = global_step / steps_per_epoch

                # logging loss
                time = timeit.default_timer() - start
                wandb.log({
                    'loss': np.mean(loss_set),
                    'time': time,
                    'step': global_step,
                    'epoch': epoch_float,
                })

                # logging at the end of each epoch
                _ = evaluation.model_change_evaluation_units(net, cfg, 'train', epoch_float)
                rmse_change_val = evaluation.model_change_evaluation_units(net, cfg, 'val', epoch_float)

                if best_rmse_change_val is None or rmse_change_val < best_rmse_change_val:
                    best_rmse_change_val = rmse_change_val
                    wandb.log({
                        'best val change rmse': best_rmse_change_val,
                        'step': global_step,
                        'epoch': epoch_float,
                    })
                    print(f'saving network (RMSE change {rmse_change_val:.3f})', flush=True)
                    networks.save_checkpoint(net, optimizer, epoch, cfg)
                    trigger_times = 0
                else:
                    trigger_times += 1
                    if trigger_times >= cfg.TRAINER.PATIENCE:
                        stop_training = True

                if stop_training:
                    break  # end of training by early stopping

            net, *_ = networks.load_checkpoint(cfg, device)
            _ = evaluation.model_change_evaluation_units(net, cfg, 'test', epoch_float)

    if args.sweep_id is None:
        # Start running a new seep
        sweep_config = {
            'method': 'grid',
            'name': cfg.NAME,
            'metric': {'goal': 'maximize', 'name': 'best val change rmse'},
            'parameters':
                {
                    'lr': {'values': [0.001, 0.0001, 0.00001]},
                }
        }
        sweep_id = wandb.sweep(sweep=sweep_config, project=args.project, entity=args.entity)
        wandb.agent(sweep_id, function=run_training)
    else:
        # Or resume existing sweep via its id
        sweep_id = args.sweep_id
        wandb.agent(sweep_id, project=args.project, function=run_training)