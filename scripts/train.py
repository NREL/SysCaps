from copy import deepcopy
import torch
from pathlib import Path
import argparse 
import wandb
import os
import tomli
import itertools as it
from timeit import default_timer as timer
from socket import gethostname
import transformers

from syscaps import utils
from syscaps.data.energyplus import EnergyPlusDataset
from syscaps.data.wind import WindDataset
from syscaps.evaluation.managers import MetricsManager
from syscaps.models import model_factory
from syscaps.evaluation import metrics

SCRIPT_PATH = Path(os.path.realpath(__file__)).parent

@torch.no_grad() 
def validation(model, val_dataloader, inverse_transform):
    """ Each batch contains B buildings each with 1 year of hours (8,760) of data.
        We will compute "national-scale" metrics by summing energy consumption over all
        buildings in the validation set, but we will average over hours.
    """
    model.eval()
    step = 0
    
    metrics_manager = MetricsManager(
        metrics=[metrics.ErrorMetric('nrmse', 'simrun', 'hourly', metrics.squared_error, normalize=True, sqrt=True),
                metrics.ErrorMetric('nmbe',  'simrun', 'hourly', metrics.bias_error, normalize=True, sqrt=False)]) 
        
    for batch in val_dataloader:   

        for k,v in batch.items():
            if torch.is_tensor(v):
                batch[k] = v.to(model.device)

        targets = batch['y']
        with torch.cuda.amp.autocast():
            preds = model.module.predict(batch) 
            batch_loss = model.module.loss(preds, targets)

        # make sure values are no longer float16
        targets = targets.to(torch.float32)
        preds = preds.to(torch.float32)

        # unscale loads 
        preds = inverse_transform(preds)
        targets = inverse_transform(targets)

        if targets.dim() == 2:
            targets = targets.unsqueeze(1)
            preds = preds.unsqueeze(1)

        metrics_manager(
            y_true=targets,
            y_pred=preds,
            loss=batch_loss
        )    

        step += 1

    model.train()

    # aggregate by summing over buildings
    summary = metrics_manager.summary(with_loss=True)
    return summary


def main(args):
    """ Main training loop
    
    Args:
        args (dict): Keys = 'model' (value is type Dict) 
                        and 'experiment' (value is type Namespace)
    """
    model_args_ = args['model']  # Dict
    experiment_args_ = args['experiment'] # Namespace

    utils.set_seed(experiment_args_.random_seed)

    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    # Optimize for fixed input sizes
    torch.backends.cudnn.benchmark = False

    ######################### DDP setup  #########################
    # SLURM_LOCALID: gpu local rank (=0 as the first gpu of the node)
    # SLURM_PROCID: gpu global rank (=4 as the fifth gpu among the 8)
    # MASTER_ADDR and MASTER_PORT env variables should be set when calling this script
    gpus_per_node = torch.cuda.device_count()    
    experiment_args_.world_size    = int(os.environ["WORLD_SIZE"])
    if experiment_args_.disable_slurm:
        local_rank     = int(os.environ["LOCAL_RANK"])
        experiment_args_.rank      = local_rank
    else:
        experiment_args_.rank      = int(os.environ["SLURM_PROCID"])
        print(f"Hello from rank {experiment_args_.rank} of {experiment_args_.world_size} on {gethostname()} where there are" \
            f" {gpus_per_node} allocated GPUs per node.", flush=True)

        local_rank = experiment_args_.rank - gpus_per_node * (experiment_args_.rank // gpus_per_node)

    print(f'About to call init_process_group on rank {experiment_args_.rank} with local rank {local_rank}', flush=True)
    torch.distributed.init_process_group(backend=experiment_args_.dist_backend, 
                                        init_method=experiment_args_.dist_url,
                                        world_size=experiment_args_.world_size,
                                        rank=experiment_args_.rank)
    if experiment_args_.rank == 0: print(f"Group initialized? {torch.distributed.is_initialized()}", flush=True)
    torch.cuda.set_device(local_rank)

    print(f'rank {experiment_args_.rank} torch cuda available = ', torch.cuda.is_available(), flush=True)
    print(f'rank {experiment_args_.rank} torch cuda device count = ', torch.cuda.device_count(), flush=True)
    print(f'rank {experiment_args_.rank} torch cuda current device = ', torch.cuda.current_device(), flush=True)
    print(f'rank {experiment_args_.rank} torch cuda get_device_name = ', torch.cuda.get_device_name(0), flush=True)
    print(f'rank {experiment_args_.rank} torch threads = ', torch.get_num_threads(), flush=True)
    
    config_path = SCRIPT_PATH  / '..' / 'syscaps'  / 'configs'/ experiment_args_.dataset
    
    model_args_list, experiment_args_list, hypparams_list = [], [], []
    if (config_path / f'{experiment_args_.hyperopt_file}.toml').exists():
        toml_args = tomli.load(( config_path / f'{experiment_args_.hyperopt_file}.toml').open('rb'))
        parameters = {}

        for arg_category in ['model', 'experiment']:
            if arg_category in toml_args:
                for k,v in toml_args[arg_category].items():
                    parameters[f'{arg_category}:{k}'] = v

        # create table whose rows are all possible combinations of list values of keys in parameters
        varNames = sorted(parameters)
        #print(f"starting hyperparameter grid search for {varNames}")
        grid = [dict(zip(varNames, prod)) for prod in it.product(*(parameters[varName] for varName in varNames))]
        
        for params in grid:
            hypparams_list += [params]
            model_args_copy = deepcopy(model_args_)
            experiment_args_copy = deepcopy(experiment_args_)

            note = '-'.join([f'{k}:{v}' for k,v in params.items()])
            note = experiment_args_copy.note + f'_{note}'
            setattr(experiment_args_copy, 'note', note)

            for k,v in params.items():
                arg_category, arg = k.split(':')

                if arg_category == 'model':
                    model_args_copy[arg] = v
                elif arg_category == 'experiment':
                    setattr(experiment_args_copy, arg, v)

            model_args_list += [model_args_copy]
            experiment_args_list += [experiment_args_copy]
    else:
        model_args_list = [model_args_]
        experiment_args_list = [experiment_args_]

    # For loop for grid search
    for params_config_idx, (model_args, experiment_args) in enumerate(zip(model_args_list, experiment_args_list)):
        if experiment_args.rank == 0 and experiment_args.hyperopt_file != '':
            the_params = hypparams_list[params_config_idx]
            print(f"starting hyperparameter grid search training run for {the_params}")
        # else, we are just doing regular training...
            
        # check environment variables
        SYSCAPS_PATH = os.environ.get('SYSCAPS', '')
        if SYSCAPS_PATH == '':
            raise ValueError('SYSCAPS environment variable not set')


        # TODO: Maybe fix
        # !THIS WILL AUTO LOAD/SAVE to your own checkpoints subdir.
        username = os.getenv("USER")
        checkpoint_dir = SCRIPT_PATH / '..' / 'checkpoints' / username
        encoder_type, model_name = experiment_args.model.split('/')
        checkpoint_name = f'{model_name}_{experiment_args.dataset}_{encoder_type}_seed={experiment_args_.random_seed}'
        # Make sure you update the experiment_args.note
        if experiment_args.rank == 0:
           
            if not checkpoint_dir.exists():
                os.makedirs(checkpoint_dir)
            
            wandb_project = os.environ.get('WANDB_PROJECT', '')
            if wandb_project == '':
                print('WANDB_PROJECT environment variable not set, disabling wandb')
                experiment_args.disable_wandb = True
            
            if experiment_args.disable_wandb:
                run = wandb.init(
                    project=wandb_project,
                    mode="disabled",
                    config={**model_args, **vars(experiment_args)})
            elif experiment_args.resume_from_checkpoint != '':
                run = wandb.init(
                    id=experiment_args.wandb_run_id,
                    project=wandb_project,
                    notes=experiment_args.note,
                    resume="allow",
                    config={**model_args, **vars(experiment_args)})
            else:
                run = wandb.init(
                    project=wandb_project,
                    notes=experiment_args.note,
                    config={**model_args, **vars(experiment_args)})
        
        global_batch_size = experiment_args.world_size * experiment_args.batch_size

        #################### Model setup ####################

        model = model_factory(experiment_args.module_name, experiment_args.dataset, model_args)
        model = model.to(local_rank)

        #################### Dataset setup ####################
        inverse_normalization_for_qoi_predictions = lambda x: x

        if experiment_args.dataset == 'energyplus_comstock' or experiment_args.dataset == 'energyplus_resstock':
            train_dataset = EnergyPlusDataset(
                buildings_bench_path=Path(SYSCAPS_PATH),
                index_file=experiment_args.train_idx_file,
                resstock_comstock='comstock' if experiment_args.dataset == 'energyplus_comstock' else 'resstock',
                syscaps_split=experiment_args.caption_dataset_split,
                return_full_year=model.is_sequential,
                tokenizer=model_args["text_encoder_name"]
            )
            val_dataset = EnergyPlusDataset(
                buildings_bench_path=Path(SYSCAPS_PATH),
                index_file=experiment_args.val_idx_file,
                resstock_comstock='comstock' if experiment_args.dataset == 'energyplus_comstock' else 'resstock',
                syscaps_split=experiment_args.caption_dataset_split,
                return_full_year=True,
                include_text = True,
                tokenizer=model_args["text_encoder_name"]
            )
            inverse_normalization_for_qoi_predictions = train_dataset.load_transform.undo_transform

        elif experiment_args.dataset == 'wind':
            train_dataset = WindDataset(
                data_path=Path(SYSCAPS_PATH),
                index_file=experiment_args.train_idx_file,
                syscaps_split=experiment_args.caption_dataset_split,
                use_random_caption_augmentation=(not experiment_args.disable_random_caption_augmentation),
                caption_augmentation_style=experiment_args.caption_augmentation_style,
            )
            val_dataset = WindDataset(
                data_path=Path(SYSCAPS_PATH),
                index_file=experiment_args.val_idx_file,
                syscaps_split=experiment_args.caption_dataset_split,
                use_random_caption_augmentation=(not experiment_args.disable_random_caption_augmentation),
                caption_augmentation_style=experiment_args.caption_augmentation_style,
                include_text = True
            )
            inverse_normalization_for_qoi_predictions = train_dataset.undo_transform

        train_sampler = torch.utils.data.distributed.DistributedSampler(
                                        dataset=train_dataset,
                                        num_replicas=experiment_args.world_size,
                                        rank=experiment_args.rank, shuffle=True)
        # val_sampler = torch.utils.data.distributed.DistributedSampler(
        #                                 dataset=val_dataset,
        #                                 num_replicas=experiment_args.world_size,
        #                                 rank=experiment_args.rank, shuffle=True)

        train_dataloader = torch.utils.data.DataLoader(
            train_dataset, 
            batch_size=experiment_args.batch_size,
            sampler=train_sampler,
            drop_last=False, 
            worker_init_fn=utils.worker_init_fn,
            collate_fn=train_dataset.collate_fn(),
            shuffle=(train_sampler is None), num_workers=experiment_args.num_workers, pin_memory=True)
        
        val_dataloader = torch.utils.data.DataLoader(
            val_dataset, batch_size=experiment_args.batch_size, 
            #sampler=val_sampler,
            drop_last=False,
            worker_init_fn=utils.worker_init_fn,
            collate_fn=val_dataset.collate_fn(),
            shuffle=False,#(val_sampler is None),
            num_workers=experiment_args.num_workers, pin_memory=True)
        

        #################### Optimizer setup ##########################

        # wrap model with DistributedDataParallel
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)

        print(f'rank {experiment_args.rank} wrapped model in DDP', flush=True)
        optimizer = torch.optim.AdamW(model.parameters(), lr=experiment_args.lr) #BB defaults: betas=(0.9, 0.98), eps=1e-9, weight_decay=0.01)
        
        #training_data_points = len(train_dataset)
        max_train_steps = experiment_args.max_train_steps 

        scheduler = transformers.get_cosine_schedule_with_warmup(
                                optimizer,
                                num_warmup_steps=100, # use a fairly small warmup, don't really want to tune this...
                                num_training_steps=max_train_steps)
            
        scaler = torch.cuda.amp.GradScaler()

        #################### Resume from checkpoint ####################

        if experiment_args.resume_from_checkpoint != '':
            model, optimizer, scheduler, step , best_val_loss, patience_counter = utils.load_model_checkpoint(
                checkpoint_dir / experiment_args.resume_from_checkpoint, model, local_rank, optimizer, scheduler)
            print(f'successfully loaded model checkpoint {username}/{experiment_args.resume_from_checkpoint}...')
            epoch = max_train_steps // len(train_dataset) # how many times have we passed over all buidings?
        else:
            step = 0
            epoch = 0
            best_val_loss = 1e9
            patience_counter = 0
            

        #################### Training loop ##############################
        step_wandb_log = 50
        early_stopping_flag = False

        print(f'rank {experiment_args.rank} step {step} epoch {epoch} max_train_steps = {max_train_steps}', flush=True)
            
        model.train()
        start_time = timer()

        while step < max_train_steps and not early_stopping_flag:
            # fix sampling seed such that each gpu gets different part of dataset 
            # and each epoch has a different shuffled order
            train_sampler.set_epoch(epoch)
            #val_sampler.set_epoch(epoch)

            for batch in train_dataloader:
                optimizer.zero_grad()

                for k,v in batch.items():
                    if torch.is_tensor(v):
                        batch[k] = v.to(model.device)
                
                targets = batch['y']  
                with torch.cuda.amp.autocast():
                    preds = model(batch) 
                    batch_loss = model.module.loss(preds, targets)
            
                # Scale Gradients
                scaler.scale(batch_loss).backward()
                
                # Update Optimizer
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()

                step += 1

                if experiment_args.rank == 0 and step % step_wandb_log == 0:
                    end_time = timer()
                    secs_per_step = (end_time - start_time) / step_wandb_log
                    start_time = timer()

                    print(f'train/loss: {batch_loss:.5f}, train/secs_per_step: {secs_per_step:.4f}, '
                          f'train/lr: {optimizer.param_groups[0]["lr"]:.5f}')

                    wandb.log({
                        'train/loss': batch_loss,
                        'train/secs_per_step': secs_per_step,
                        'train/lr': optimizer.param_groups[0]['lr']
                    }, step=step)

                if experiment_args.rank == 0 and step % 500 == 0:
                    print(f'started validation at step {step}...')

                    val_summary = validation(model, val_dataloader, 
                                            inverse_normalization_for_qoi_predictions)
                    val_loss = val_summary['loss']
                    # only rank 0 needs to save model
                    if val_loss < best_val_loss:
                        patience_counter = 0

                        best_val_loss = val_loss

                        model_name = checkpoint_name + '_best.pt'
      
                        # delete previous
                        if (checkpoint_dir / model_name).exists():
                            Path(checkpoint_dir / model_name).unlink()
                        utils.save_model_checkpoint(model, optimizer, scheduler, step, best_val_loss, 
                                              patience_counter, checkpoint_dir / model_name)
                    else:
                        # patience counter for early stopping
                        patience_counter += 1
                        if patience_counter >= experiment_args.early_stopping_patience:
                            print(f'Early stopping at step {step} with val loss {val_loss}')
                            early_stopping_flag = True
                    
                    # we always save the last val model
                    last_model_name = checkpoint_name + '_last.pt' #f'ckpt-{experiment_args.note}_last.pt'
                    utils.save_model_checkpoint(model, optimizer, scheduler, step, best_val_loss, 
                                              patience_counter, checkpoint_dir / last_model_name)

                    # N.b. these are aggregated by summing over all val buildings.
                    # nrmse is average hourly energy consumption normalized over all val buildings
                    print_string = ''
                    for metric_name in ['nrmse', 'nmbe']:
                        print_string += f'val/{metric_name}: {val_summary[metric_name].value:.5f}, '
                        wandb.log({
                                f'val/{metric_name}' : val_summary[metric_name].value 
                        }, step=step)
                    print_string += f'val/loss: {val_loss:.5f}, val/best_loss: {best_val_loss:.5f}'
                        
                    wandb.log({
                        'val/loss': val_loss,
                        'val/best_loss': best_val_loss
                    }, step=step)
                    print(print_string)
                    print(f'finished validation at step {step}...')
            
                if step >= max_train_steps or early_stopping_flag:
                    # stop training after this many steps/train_tokens
                    break
            # for batch in train_dataloader
            else:
                if experiment_args.rank == 0:
                    print(f'\nepoch {epoch} best val loss {best_val_loss}...')
                epoch += 1
                continue # continue if step < train_steps
            break   
    
        run.finish()
    torch.distributed.destroy_process_group()        


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Experiment args. If provided in config file, these will be overridden.
    # Use arg `hyper_opt` to avoid overriding the argparse args with the config file.
    parser.add_argument('--random_seed', type=int, default=1234)
    parser.add_argument('--early_stopping_patience', type=int, default=5,
                        help='Number of validation rounds without validation loss improvement'
                             'before early stopping.')
    #parser.add_argument('--ignore_scoring_rules', action='store_true',
    #                    help='Do not compute a scoring rule for this model.')
    
    parser.add_argument('--resume_from_checkpoint', type=str, default='')
    parser.add_argument('--wandb_run_id', type=str, default='')

    # Wandb
    parser.add_argument('--disable_wandb', action='store_true')
    parser.add_argument('--note', type=str, default='',
                        help='Note to append to model checkpoint name. '
                        'Also used for wandb notes.')    

    # DDP
    parser.add_argument('--disable_slurm', action='store_true')
    parser.add_argument('--world-size', default=-1, type=int, 
                        help='number of nodes for distributed training')
    parser.add_argument('--rank', default=-1, type=int, 
                        help='node rank for distributed training')
    parser.add_argument('--dist-url', default='env://', type=str, 
                        help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str, 
                        help='distributed backend')
    parser.add_argument('--num_workers', type=int, default=1)

    # Model training
    parser.add_argument('--model', type=str, default='', required=True,
                        help='Name of your model. Should match your models config'
                             ' filename without .toml extension.'
                             ' Example: "LSTM".')
    parser.add_argument('--hyperopt_file', type=str, default='',
                        help='Hyperparameter optimization. '
                             'Name of your model + "_hyperopt.toml".')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=0.00001)
    parser.add_argument('--max_train_steps', type=int, default=100000)

    ### ignore
    parser.add_argument('--module_name', type=str, default='',
                        help='Name of syscaps model module to import. '
                             'Overridden by config file.')
    
    # Data
    parser.add_argument('--dataset', type=str, default='energyplus_comstock', required=True,
                        choices=['energyplus_comstock', 'energyplus_resstock', 'wind'])    
    parser.add_argument('--train_idx_file', type=str, default='',
                        help='Name of index files for training')
    parser.add_argument('--val_idx_file', type=str, default='',
                        help='Name of index files for validation')
    parser.add_argument('--caption_dataset_split', type=str, default='short',
                        choices=['keyvalue', 'short', 'medium', 'long'])
    
    ## wind only
    parser.add_argument('--caption_augmentation_style', type=int, default=1,
                        help='1: with an objective tone.')
    parser.add_argument('--disable_random_caption_augmentation', action='store_true',
                        help='disable random caption augmentation.')
    
    if not torch.cuda.is_available():
        raise ValueError('CUDA is not available for training!')
    
    experiment_args = parser.parse_args()
    model_args = {}
    config_path = SCRIPT_PATH  / '..' / 'syscaps'  / 'configs'/ experiment_args.dataset
    if (config_path / f'{experiment_args.model}.toml').exists():
        toml_args = tomli.load(( config_path / f'{experiment_args.model}.toml').open('rb'))
        # grab the custom model args as set in the config file
        model_args = toml_args['model']

        # grab inputs to override the argparse defaults above from the config file
        if 'experiment' in toml_args:
            for k,v in toml_args['experiment'].items():
                #if not k in experiment_args.hyper_opt:
                if hasattr(experiment_args, k):
                    print(f'Overriding argparse default for {k} with {v}')
                # Just set the argparse value to the value in the config file
                # even if there is no default
                setattr(experiment_args, k, v)
        
    else:
        raise ValueError(f'config {experiment_args.model}.toml not found.')

    all_args = {
        'experiment': experiment_args,
        'model': model_args
    }

    print("start training...")
    main(all_args)
