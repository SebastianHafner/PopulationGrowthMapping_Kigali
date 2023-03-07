import torch
from pathlib import Path
import numpy as np
from utils import experiment_manager, networks, datasets, parsers, geofiles

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def produce_population_grids(cfg: experiment_manager.CfgNode):
    net, *_ = networks.load_checkpoint(cfg, device)
    net.eval()
    for year in [2016, 2020]:
        ds = datasets.PopInferenceDataset(cfg, year)
        arr = ds.get_pop_grid()
        tracker = 0
        with torch.no_grad():
            for item in ds:
                x = item['x'].to(device)
                pred = net(x.unsqueeze(0)).cpu().item()
                gt = item['y']
                i, j = item['i'], item['j']
                arr[i, j, 0] = float(pred)
                arr[i, j, 1] = gt
                tracker += 1
                if tracker % 10_000 == 0:
                    print(tracker)

        transform, crs = ds.get_pop_grid_geo()
        file = Path(cfg.PATHS.OUTPUT) / 'inference' / 'population_grids' / f'pop_kigali_{year}_{cfg.NAME}.tif'
        geofiles.write_tif(file, arr, transform, crs)


def produce_population_grids_endtoend(cfg: experiment_manager.CfgNode):
    net, *_ = networks.load_checkpoint(cfg, device)
    net.eval()

    t1, t2 = 2016, 2020
    ds_t1 = datasets.PopInferenceDataset(cfg, t1, nonans=True)
    ds_t2 = datasets.PopInferenceDataset(cfg, t2, nonans=True)
    assert (len(ds_t1) == len(ds_t2))

    tracker = 0
    arr_t1, arr_t2 = ds_t1.get_pop_grid(), ds_t2.get_pop_grid()
    with torch.no_grad():
        for index in range(len(ds_t1)):
            item_t1 = ds_t1.__getitem__(index)
            x_t1 = item_t1['x'].to(device)
            item_t2 = ds_t2.__getitem__(index)
            x_t2 = item_t2['x'].to(device)
            pred_change, pred_t1, pred_t2 = net(x_t1.unsqueeze(0), x_t2.unsqueeze(0))
            pred_change, pred_t1, pred_t2 = pred_change.cpu().item(), pred_t1.cpu().item(), pred_t2.cpu().item()

            for t, item, pred, arr in zip([t1, t2], [item_t1, item_t2], [pred_t1, pred_t2], [arr_t1, arr_t2]):
                gt = item['y']
                i, j = item['i'], item['j']
                arr[i, j, 0] = float(pred)
                arr[i, j, 1] = gt
            tracker += 1
            if tracker % 10_000 == 0:
                print(tracker)

        for t, ds, arr in zip([t1, t2], [ds_t1, ds_t2], [arr_t1, arr_t2]):
            transform, crs = ds.get_pop_grid_geo()
            file = Path(cfg.PATHS.OUTPUT) / 'inference' / 'population_grids' / f'pop_kigali_{t}_{cfg.NAME}.tif'
            geofiles.write_tif(file, arr, transform, crs)


# quantitative population predictions on a census level unit
def produce_unit_stats(cfg: experiment_manager.CfgNode):
    net, *_ = networks.load_checkpoint(cfg, device)
    net.eval()

    data = {}
    for year in [2016, 2020]:
        ds = datasets.PopInferenceDataset(cfg, year, nonans=True)
        tracker = 0
        with torch.no_grad():
            for item in ds:
                x = item['x'].to(device)
                pred = net(x.unsqueeze(0)).cpu().item()
                gt = item['y']
                unit = int(item['unit'])
                if str(unit) not in data.keys():
                    data[str(unit)] = {}
                unit_data = data[str(unit)]
                if f'pred_pop{year}' in unit_data.keys():
                    unit_data[f'pred_pop{year}'] += pred
                    if not np.isnan(gt):
                        unit_data[f'gt_pop{year}'] += gt
                else:
                    unit_data[f'pred_pop{year}'] = pred
                    if not np.isnan(gt):
                        unit_data[f'gt_pop{year}'] = gt
                tracker += 1
                if tracker % 10_000 == 0:
                    print(tracker)

    out_file = Path(cfg.PATHS.OUTPUT) / 'inference' / 'unit_stats' / f'pop_kigali_{cfg.NAME}.json'
    geofiles.write_json(out_file, data)


def produce_unit_stats_endtoend(cfg: experiment_manager.CfgNode):
    net, *_ = networks.load_checkpoint(cfg, device)
    net.eval()

    data = {}
    t1, t2 = 2016, 2020
    ds_t1 = datasets.PopInferenceDataset(cfg, t1, nonans=True)
    ds_t2 = datasets.PopInferenceDataset(cfg, t2, nonans=True)
    assert(len(ds_t1) == len(ds_t2))

    tracker = 0
    with torch.no_grad():
        for index in range(len(ds_t1)):
            item_t1 = ds_t1.__getitem__(index)
            x_t1 = item_t1['x'].to(device)
            item_t2 = ds_t2.__getitem__(index)
            x_t2 = item_t2['x'].to(device)
            pred_change, pred_t1, pred_t2 = net(x_t1.unsqueeze(0), x_t2.unsqueeze(0))
            pred_change, pred_t1, pred_t2 = pred_change.cpu().item(), pred_t1.cpu().item(), pred_t2.cpu().item()

            gt_t1, gt_t2 = item_t1['y'], item_t2['y']
            unit_t1, unit_t2 = int(item_t1['unit']), int(item_t2['unit'])
            assert(unit_t1 == unit_t2)
            unit = unit_t1

            if str(unit) not in data.keys():
                data[str(unit)] = {}
            unit_data = data[str(unit)]

            for t, pred, gt in zip([t1, t2], [pred_t1, pred_t2], [gt_t1, gt_t2]):
                if f'pred_pop{t}' in unit_data.keys():
                    unit_data[f'pred_pop{t}'] += pred
                    if not np.isnan(gt):
                        unit_data[f'gt_pop{t}'] += gt
                else:
                    unit_data[f'pred_pop{t}'] = pred
                    if not np.isnan(gt):
                        unit_data[f'gt_pop{t}'] = gt

            if 'pred_change' in unit_data.keys():
                unit_data['pred_change'] += pred_change
            else:
                unit_data[f'pred_change'] = pred_change

            tracker += 1
            if tracker % 10_000 == 0:
                print(tracker)

    out_file = Path(cfg.PATHS.OUTPUT) / 'inference' / 'unit_stats' / f'pop_kigali_{cfg.NAME}.json'
    geofiles.write_json(out_file, data)


if __name__ == '__main__':
    args = parsers.deployment_argument_parser().parse_known_args()[0]
    cfg = experiment_manager.setup_cfg(args)
    if not cfg.CHANGE_DETECTION.ENDTOEND:
        produce_unit_stats(cfg)
        produce_population_grids(cfg)
    else:
        produce_unit_stats_endtoend(cfg)
        produce_population_grids_endtoend(cfg)
