import torch
import matplotlib.pyplot as plt
from matplotlib import lines
from scipy import stats
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
from pathlib import Path
from utils import experiment_manager, networks, datasets, parsers, geofiles
import geopandas as gpd


def total_population(cfg: experiment_manager.CfgNode, run_type: str = 'all'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net, *_ = networks.load_checkpoint(cfg.INFERENCE_CHECKPOINT, cfg, device)
    net.eval()
    ds = datasets.PopDataset(cfg, run_type, no_augmentations=True)
    y_gt, y_pred = 0, 0

    with torch.no_grad():
        for item in ds:
            x = item['x'].to(device)
            y = item['y']
            pred = net(x.unsqueeze(0))
            y_gt += y.item()
            y_pred += pred.cpu().item()

    print(y_gt, y_pred)


def unit_stats_bitemporal(cfg: experiment_manager.CfgNode):
    file = Path(cfg.PATHS.OUTPUT) / 'inference' / 'unit_stats' / f'pop_kigali_{cfg.NAME}.json'
    assert(file.exists())
    pred_data = geofiles.load_json(file)

    metadata_file = Path(cfg.PATHS.DATASET) / f'metadata.json'
    metadata = geofiles.load_json(metadata_file)

    census = metadata['census']
    units = list(census.keys())
    fig_data = {}
    for split in ['training', 'test']:
        fig_data[split] = {}
        for variable in ['pred', 'gt']:
            fig_data[split][variable] = {'2016': [], '2020': []}
            fig_data[split][f'{variable}_change'] = []

    for unit in units:
        split = census[str(unit)]['split']
        for year in [2016, 2020]:
            # ground truth
            fig_data[split]['gt'][str(year)].append(census[str(unit)][f'pop{year}'])
            # prediction
            fig_data[split]['pred'][str(year)].append(pred_data[str(unit)][f'pred_pop{year}'])
        if cfg.CHANGE_DETECTION.ENDTOEND:
            fig_data[split]['pred_change'].append(pred_data[str(unit)]['pred_change'])

    print(fig_data)
    fig, axs = plt.subplots(2, 3, figsize=(15, 10), constrained_layout=True)
    plt.tight_layout()
    for i, split in enumerate(['training', 'test']):
        for j, year in enumerate([2016, 2020]):
            gt = fig_data[split]['gt'][str(year)]
            pred = fig_data[split]['pred'][str(year)]
            axs[i, j].scatter(gt, pred)
            _, _, r_value, *_ = stats.linregress(gt, pred)
            textstr = r'$R^2 = {r_value:.2f}$'.format(r_value=r_value)
            axs[i, j].text(0.05, 0.95, textstr, transform=axs[i, j].transAxes,verticalalignment='top')


        # difference
        gt_diff = np.array(fig_data[split]['gt']['2020']) - np.array(fig_data[split]['gt']['2016'])
        if cfg.CHANGE_DETECTION.ENDTOEND:
            pred_diff = np.array(fig_data[split]['pred_change'])
        else:
            pred_diff = np.array(fig_data[split]['pred']['2020']) - np.array(fig_data[split]['pred']['2016'])

        axs[i, 2].scatter(gt_diff, pred_diff)
        _, _, r_value, *_ = stats.linregress(gt_diff, pred_diff)
        textstr = r'$R^2 = {r_value:.2f}$'.format(r_value=r_value)
        axs[i, 2].text(0.05, 0.95, textstr, transform=axs[i, 2].transAxes, verticalalignment='top')

    for index, ax in np.ndenumerate(axs):
        i, j = index
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlabel('Ground Truth')
        ax.set_ylabel('Predicted')
        if i < 2 and j < 2:
            axs[i, j].set_xlim(0, 50_000)
            axs[i, j].set_ylim(0, 50_000)
            ax.plot([0, 50_000], [0, 50_000], c='r', zorder=-1, label='1:1 line')
        else:
            if cfg.CHANGE_DETECTION.ENDTOEND:
                min_diff = -1_000
                max_diff = 2_000
            else:
                min_diff = -1_000
                max_diff = 10_000
            ax.plot([min_diff, max_diff], [min_diff, max_diff], c='r', zorder=-1, label='1:1 line')
            axs[i, j].set_xlim(min_diff, max_diff)
            axs[i, j].set_ylim(min_diff, max_diff)

    legend_elements = [
        lines.Line2D([0], [0], color='r', lw=1, label='1:1 Line'),
        lines.Line2D([0], [0], marker='.', color='w', markerfacecolor='#1f77b4', label='Census Unit', markersize=15),
    ]
    axs[0, 0].legend(handles=legend_elements, frameon=False, loc='upper center')


    cols = ['Population 2016', 'Population 2020', 'Difference']
    rows = ['Training', 'Test']

    pad = 5  # in points

    for ax, col in zip(axs[0], cols):
        ax.annotate(col, xy=(0.5, 1), xytext=(0, pad),
                    xycoords='axes fraction', textcoords='offset points',
                    size='large', ha='center', va='baseline')

    for ax, row in zip(axs[:, 0], rows):
        ax.annotate(row, xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - pad, 0),
                    xycoords=ax.yaxis.label, textcoords='offset points',
                    size='large', ha='right', va='center')

    fig.tight_layout()
    fig.subplots_adjust(left=0.15, top=0.95)

    file = Path(cfg.PATHS.OUTPUT) / 'plots' / f'{cfg.NAME}.png'
    plt.savefig(file, dpi=300, bbox_inches='tight')
    plt.show()


def unit_stats_change(cfg: experiment_manager.CfgNode, split: str = 'test'):
    file = Path(cfg.PATHS.OUTPUT) / 'inference' / 'unit_stats' / f'pop_kigali_{cfg.NAME}.json'
    assert (file.exists())
    pred_data = geofiles.load_json(file)

    metadata_file = Path(cfg.PATHS.DATASET) / f'metadata.json'
    metadata = geofiles.load_json(metadata_file)

    census = metadata['census']
    units = list(census.keys())
    fig_data = {'test': {}}

    for variable in ['pred', 'gt']:
        fig_data[split][variable] = {'2016': [], '2020': []}
        fig_data[split][f'{variable}_change'] = []

    for unit in units:
        unit_split = census[str(unit)]['split']
        if unit_split == split:
            for year in [2016, 2020]:
                # ground truth
                fig_data[unit_split]['gt'][str(year)].append(census[str(unit)][f'pop{year}'])
                # prediction
                fig_data[unit_split]['pred'][str(year)].append(pred_data[str(unit)][f'pred_pop{year}'])
            if cfg.CHANGE_DETECTION.ENDTOEND:
                fig_data[unit_split]['pred_change'].append(pred_data[str(unit)]['pred_change'])

    print(fig_data)
    plt.rcParams.update({'font.size': 14})
    fig, ax = plt.subplots(1, 1, figsize=(5, 5), constrained_layout=True)
    plt.tight_layout()

    gt_diff = np.array(fig_data[split]['gt']['2020']) - np.array(fig_data[split]['gt']['2016'])
    if cfg.CHANGE_DETECTION.ENDTOEND:
        pred_diff = np.array(fig_data[split]['pred_change'])
    else:
        pred_diff = np.array(fig_data[split]['pred']['2020']) - np.array(fig_data[split]['pred']['2016'])

    ax.scatter(gt_diff, pred_diff, s=10)
    rmse = mean_squared_error(gt_diff, pred_diff, squared=False)
    ax.text(0.05, 0.95, rf'RMSE $= {rmse:.0f}$', transform=ax.transAxes, verticalalignment='top')
    mae = mean_absolute_error(gt_diff, pred_diff)
    ax.text(0.05, 0.88, rf'MAE $= {mae:.0f}$', transform=ax.transAxes, verticalalignment='top')
    _, _, r_value, *_ = stats.linregress(gt_diff, pred_diff)
    ax.text(0.05, 0.81, rf'R$^2 = {r_value:.2f}$', transform=ax.transAxes, verticalalignment='top')

    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel('Ground Truth')
    ax.set_ylabel('Predicted')

    if cfg.CHANGE_DETECTION.ENDTOEND:
        min_diff = -250
        max_diff = 1_000
    else:
        min_diff = -2_500
        max_diff = 7_500
    ticks = np.linspace(min_diff, max_diff, num=6, endpoint=True)
    ax.plot([min_diff, max_diff], [min_diff, max_diff], c='r', zorder=-1, label='1:1 line')
    ax.set_xlim(min_diff, max_diff)
    ax.set_xticks(ticks)
    ax.set_ylim(min_diff, max_diff)
    ax.set_yticks(ticks)


    legend_elements = [
        lines.Line2D([0], [0], color='r', lw=1, label='1:1 Line'),
        lines.Line2D([0], [0], marker='.', color='w', markerfacecolor='#1f77b4', label='Census Unit', markersize=10),
    ]
    ax.legend(handles=legend_elements, frameon=False, loc='lower right', handlelength=1)

    file = Path(cfg.PATHS.OUTPUT) / 'plots' / f'change_{cfg.NAME}.png'
    plt.savefig(file, dpi=300, bbox_inches='tight')
    plt.show()


def print_quantitative_results(cfg: experiment_manager.CfgNode, split: str = 'test'):
    file = Path(cfg.PATHS.OUTPUT) / 'inference' / 'unit_stats' / f'pop_kigali_{cfg.NAME}.json'
    assert (file.exists())
    pred_data = geofiles.load_json(file)

    metadata_file = Path(cfg.PATHS.DATASET) / f'metadata.json'
    metadata = geofiles.load_json(metadata_file)

    census = metadata['census']
    units = list(census.keys())
    fig_data = {split: {}}

    for variable in ['pred', 'gt']:
        fig_data[split][variable] = {'2016': [], '2020': []}
        fig_data[split][f'{variable}_change'] = []

    for unit in units:
        unit_split = census[str(unit)]['split']
        if unit_split == split:
            for year in [2016, 2020]:
                # ground truth
                fig_data[split]['gt'][str(year)].append(census[str(unit)][f'pop{year}'])
                # prediction
                fig_data[split]['pred'][str(year)].append(pred_data[str(unit)][f'pred_pop{year}'])

    for year in [2016, 2020]:
        gt = fig_data[split]['gt'][str(year)]
        pred = fig_data[split]['pred'][str(year)]
        rmse = mean_squared_error(gt, pred, squared=False)
        mae = mean_absolute_error(gt, pred)
        _, _, r_value, *_ = stats.linregress(gt, pred)
        print(f'{split} {year}: RMSE = {rmse:.0f}; MAE = {mae:.0f}; R2 = {r_value:.2f}.')


def print_quantitative_results_grid(cfg: experiment_manager.CfgNode, split: str = 'test', year: int = 2020):
    file = Path(cfg.PATHS.OUTPUT) / 'inference' / 'population_grids' / f'pop_kigali_{year}_{cfg.NAME}.tif'
    assert (file.exists())
    arr, *_ = geofiles.read_tif(file)
    pred, gt = arr[:, :, 0].flatten(), arr[:, :, 1].flatten()
    not_nan = ~np.isnan(gt)
    pred, gt = pred[not_nan], gt[not_nan]

    rmse = mean_squared_error(gt, pred, squared=False)
    mae = mean_absolute_error(gt, pred)
    _, _, r_value, *_ = stats.linregress(gt, pred)
    print(f'{split} {year}: RMSE = {rmse:.0f}; MAE = {mae:.0f}; R2 = {r_value:.2f}.')


def produce_census_maps(cfg: experiment_manager.CfgNode):
    pred_file = Path(cfg.PATHS.OUTPUT) / 'inference' / 'unit_stats' / f'pop_kigali_{cfg.NAME}.json'
    assert (pred_file.exists())
    pred_data = geofiles.load_json(pred_file)

    metadata_file = Path(cfg.PATHS.DATASET) / f'metadata.json'
    metadata = geofiles.load_json(metadata_file)
    census = metadata['census']

    file = Path(cfg.PATHS.DATASET) / 'ancillary_data' / f'census_kigali.shp'
    assert (file.exists())
    gdf = gpd.read_file(file)

    years = (2016, 2020)
    gdf['Pred2016'] = 0
    gdf['Pred2020'] = 0
    gdf['PredGrowth'] = 0
    gdf['Split'] = ''

    for index, row in gdf.iterrows():
        unit_nr = str(row['id'])
        gdf.iat[index, gdf.columns.get_loc('Split')] = census[unit_nr]['split']
        for year in years:
            gt = census[unit_nr][f'pop{year}']
            gdf.iat[index, gdf.columns.get_loc(f'Pred{year}')] = pred_data[unit_nr][f'pred_pop{year}']
            # print(row[f'Pop{year}'], pred_data[unit_nr][f'gt_pop{year}'])

        if cfg.CHANGE_DETECTION.ENDTOEND:
            pred_change = pred_data[unit_nr]['pred_change']
        else:
            pred_change = pred_data[unit_nr]['pred_pop2020'] - pred_data[unit_nr]['pred_pop2016']
        gdf.iat[index, gdf.columns.get_loc('PredGrowth')] = pred_change

    file = Path(cfg.PATHS.OUTPUT) / 'maps' / f'{cfg.NAME}.geojson'
    gdf.to_file(file, driver='GeoJSON')


if __name__ == '__main__':
    args = parsers.deployment_argument_parser().parse_known_args()[0]
    cfg = experiment_manager.setup_cfg(args)
    # unit_stats_change(cfg)
    # print_quantitative_results(cfg)
    # print_quantitative_results_grid(cfg)
    produce_census_maps(cfg)
    # total_population(cfg, run_type=args.run_type)
