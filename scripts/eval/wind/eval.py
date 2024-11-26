from syscaps.evaluation import metrics
from syscaps.evaluation.managers import MetricsManager
from syscaps.models import model_factory
from syscaps.data.wind import WindDataset
from syscaps import utils
import torch 
from tqdm import tqdm
import tomli
import argparse
from pathlib import Path 
import os
import wandb
import pandas as pd


SCRIPT_PATH = Path(os.path.realpath(__file__)).parent
SYSCAPS_PATH = os.environ.get('SYSCAPS', '')
if SYSCAPS_PATH == '':
    raise ValueError('SYSCAPS environment variable not set')

# eval one model on one dataset
def evaluate(model, dataloader, caption_split, args):
    model.eval()
    metrics_manager = MetricsManager(
    metrics=[metrics.ErrorMetric('nrmse', 'simrun', 'hourly', metrics.squared_error, normalize=True, sqrt=True),
            metrics.ErrorMetric('nmbe',  'simrun', 'hourly', metrics.bias_error, normalize=True, sqrt=False)]) 

    with torch.no_grad():
        for batch in tqdm(dataloader):   
            
            for k,v in batch.items():
                if torch.is_tensor(v):
                    batch[k] = v.to(args.device)

            targets = batch['y']

            with torch.cuda.amp.autocast():
                preds = model.predict(batch)
                batch_loss = model.loss(preds, targets)

            targets = targets.to(torch.float32)
            preds = preds.to(torch.float32)

            # unscale loads 
            targets = inverse_normalization_for_qoi_predictions(targets)
            preds = inverse_normalization_for_qoi_predictions(preds)

            if targets.dim() == 2:
                targets = targets.unsqueeze(1)
                preds = preds.unsqueeze(1)
                
            metrics_manager(
                y_true=targets,
                y_pred=preds,
                loss=batch_loss
            )

    summary = metrics_manager.summary(with_loss=True)
    return summary

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval_name', type=str, required=True,
                        help='Use the same eval_name each time you execute '
                        'this script, for each model you want to compare in '
                         'the same wandb table.')
    parser.add_argument('--model_fnames', type=str, required=True, 
                        help="model file names seperated by \",\"")
    parser.add_argument("--ckpt_dir", type=str, required=True, 
                        help="directory of saved checkpoints")
    parser.add_argument('--index_file', type=str, default='floris_test_seed=42.idx')
    parser.add_argument('--model', type=str, required=True,
                        help='Ex: {onehot/keyvalue/medium}/ResNet')
    parser.add_argument('--device', type=str, default="cuda:0")
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--caption_splits', type=str, default='onehot',
                        help='caption splits, separate by \",\", default = onehot')    
    parser.add_argument('--wandb_project', type=str, default='',
                        help='wandb project name for these eval runs')
    args = parser.parse_args()

    args.model_fnames = [model_fname.strip() for model_fname in args.model_fnames.split(",") if model_fname != ""]

    args.caption_splits = [caption_split.strip() for caption_split in args.caption_splits.split(",") if caption_split != ""]

    print("model files", args.model_fnames)
    print("caption splits", args.caption_splits)

    config_path = SCRIPT_PATH  / '..' / '..' / '..' / \
          'syscaps'  / 'configs'/ 'wind'

    if (config_path / f'{args.model}.toml').exists():
        toml_args = tomli.load(( config_path / f'{args.model}.toml' ).open('rb'))
    else:
        raise ValueError()

    # grab the custom model args as set in the config file
    model_args = toml_args['model']
    model = model_factory(toml_args['experiment']['module_name'], 'wind',
                          model_args)
    model = model.to(args.device)

    # checkpoint directory
    ckpt_dir = Path(args.ckpt_dir)

    run = wandb.init(
        project=args.wandb_project,
        notes=args.model,
        config={**model_args, **vars(args)})

    column_names = ['model', 'dataset', 'caption']
    row_data = []
    
    i = 0
    for caption_split in args.caption_splits:
        for model_fname in args.model_fnames:
            model.load_from_checkpoint(ckpt_dir / model_fname, device=args.device)

            #for index_file in args.index_files:
            print(f"model fname = {model_fname}")
            #print(f"index file = {index_file}")
            print(f"caption split = {caption_split}")

            dataset =  WindDataset(
                data_path=Path(SYSCAPS_PATH),
                index_file=args.index_file,
                syscaps_split=caption_split if caption_split != 'onehot' else 'keyvalue',
                use_random_caption_augmentation=False, # turn off augmentation for testing
                caption_augmentation_style=1, # style = with an objective tone
                include_text = True
            )
            inverse_normalization_for_qoi_predictions = dataset.undo_transform

            dataloader = torch.utils.data.DataLoader(
                dataset, batch_size=args.batch_size,
                drop_last=False,
                num_workers = 8,
                worker_init_fn=utils.worker_init_fn,
                collate_fn=dataset.collate_fn())

            summary = evaluate(model, dataloader, caption_split, args)

            data = [model_fname, args.index_file, caption_split]
            for k,v in summary.items():
                if k == 'loss':
                    print(f'loss: {summary[k]}')
                    data += [summary[k]]
                else:
                    print(f'{summary[k].name}: {summary[k].value}')
                    data += [summary[k].value]
                
                if i == 0:
                    column_names += [k]
            print()
            row_data += [data]
            i += 1
    table = wandb.Table(columns=column_names, data=row_data)
    wandb.log({args.eval_name: table})
    # create a pandas dataframe for the table
    for row in row_data:
        for i in range(len(row)):
            if isinstance(row[i], torch.Tensor):
                row[i] = row[i].cpu().item()
    #row_data = [e.cpu().item() if isinstance(e, torch.Tensor) else e for e in row_data]
    df = pd.DataFrame(row_data, columns=column_names)
    # save to csv
    df.to_csv(f'{args.eval_name}.csv')
    
    run.finish()