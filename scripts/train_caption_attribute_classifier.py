
from syscaps.data.energyplus import EnergyPlusDataset
from syscaps import utils
from syscaps.models.modules import TextEncoder

from sklearn.preprocessing import OneHotEncoder
import os
from pathlib import Path 
import torch 
from tqdm import tqdm
import numpy as np
import pandas as pd
import argparse 
import wandb 
import torch
import torch.nn as nn

## Env variables
SCRIPT_PATH = Path(os.path.realpath(__file__)).parent
SYSCAPS_PATH = os.environ.get('SYSCAPS', '')
if SYSCAPS_PATH == '':
    raise ValueError('SYSCAPS environment variable not set')

SYSCAPS_PATH = Path(SYSCAPS_PATH)


class MultiLabelClassifier(nn.Module):
    """
    Test the quality of captions by mapping the output of
    DistilBERT back to each one.
    """
    def __init__(self, classes_per_task, text_encoder_name, freeze=True):
        super().__init__()

        self.attr_text_encoder = TextEncoder(
            model_name = text_encoder_name,
            freeze = freeze,
            finetune=False
        )
        self.heads = nn.ModuleList()
        for t in range(len(classes_per_task)):
            self.heads += [
                nn.Linear(self.attr_text_encoder.output_dim, classes_per_task[t])
            ]

    def forward(self, batch):
        """
        """
        y_text = {'input_ids': batch["attributes_input_ids"],
                'attention_mask': batch["attributes_attention_mask"]}
        g_text = self.attr_text_encoder(y_text['input_ids'], y_text['attention_mask'])

        logits = []
        for h in self.heads:
            logits += [ h(g_text) ]
        return logits  
    
def evaluate_on_dataset(model, dataloader):
    model.eval()
    avg_val_loss = []
    attribute_accuracy = {}
    for a in attributes:
        attribute_accuracy[a] = []
        
    with torch.no_grad():
        for batch in tqdm(dataloader):   
            for k,v in batch.items():
                if torch.is_tensor(v):
                    batch[k] = v.to(args.device)
            # convert this into labels
            onehot = batch['attributes_onehot']
            labels = []
            i = 0
            for c in classes_per_task:
                labels += [
                    onehot[:,0,i:i+c]
                ]
                i += c

            with torch.cuda.amp.autocast():
                logits_per_head = model(batch)
                val_loss = 0
                for logits,label in zip(logits_per_head, labels):
                    val_loss += nn.functional.cross_entropy(logits, label)
                val_loss = val_loss / len(classes_per_task) # average task loss
            avg_val_loss += [val_loss.item()]

            for logits,label,att in zip(logits_per_head, labels, attributes):
                attribute_accuracy[att] += [ ((nn.functional.one_hot(torch.argmax(logits,-1), \
                                                                     label.shape[1]) & label.long()).sum() / logits.shape[0]).item() ]
    
    acc_means = {}
    for k,v in attribute_accuracy.items():
        acc_means[k] = np.mean(v)

    return np.mean(avg_val_loss), acc_means

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, required=True,
                        choices=['energyplus_comstock', 'energyplus_resstock'])    
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--caption_split', type=str, default='medium')
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--max_epochs', type=int, default=100)
    parser.add_argument('--early_stopping_patience', type=int, default=5)
    parser.add_argument('--random_seed', type=int, default=111)
    parser.add_argument('--text_encoder_name', type=str, default='distilbert-base-uncased')
    parser.add_argument('--freeze_text_encoder', action='store_true', default=False)
    parser.add_argument('--resume_from_checkpoint', type=str, default='')
    parser.add_argument('--wandb_run_id', type=str, default='')
    parser.add_argument('--batch_step', type=int, default=0,
                        help='TODO delete')
    parser.add_argument('--evaluate', action='store_true', default=False)
    parser.add_argument('--device', type=str, default="cuda:0")
    parser.add_argument('--disable_wandb', action='store_true', default=False)

    args = parser.parse_args()

    utils.set_seed(args.random_seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    # Optimize for fixed input sizes
    torch.backends.cudnn.benchmark = False

    # !THIS WILL AUTO LOAD/SAVE to your own checkpoints subdir.
    username = os.getenv("USER")
    checkpoint_dir = SCRIPT_PATH / '..' / 'checkpoints' / username
    if not checkpoint_dir.exists():
        os.makedirs(checkpoint_dir)

    resstock_comstock = 'comstock' if args.dataset == 'energyplus_comstock' else 'resstock'
    attributes = open(SYSCAPS_PATH / 'metadata' / f'attributes_{resstock_comstock}.txt', 'r').read().split('\n')
    df1 = pd.read_parquet(SYSCAPS_PATH / 'metadata' / f"{resstock_comstock}_amy2018.parquet", engine="pyarrow")
    df2 = pd.read_parquet(SYSCAPS_PATH / 'metadata' / f"{resstock_comstock}_tmy3.parquet", engine="pyarrow")
    attributes = [x.strip('"') for x in attributes]
    attributes = [x for x in attributes if x != '']
    
    # if resstock, remove _ft_2 attributes
    if resstock_comstock == 'resstock':
        attributes = [x for x in attributes if '_ft_2' not in x]
    
    num_tasks = len(attributes)

    df = df1.loc[ df1.index.intersection(df2.index).values ]

    classes_per_task = []    
    for a in attributes:
        attribute_onehot_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
        attribute_onehot_encoder.fit(df[[a]].values)
        classes_per_task += [attribute_onehot_encoder.transform([[None]]).shape[1]]

    print(f'classes_per_task: {classes_per_task}')

    # set up WANDB
    note = f'classifier-caption-split-{args.caption_split}-seed={args.random_seed}'
    if args.disable_wandb:
        run = wandb.init(
                project=f'{resstock_comstock}-eval',
                mode="disabled",
                config=args)
    elif args.resume_from_checkpoint != '' and not args.evaluate:
        run = wandb.init(
            id=args.wandb_run_id,
            project=f'{resstock_comstock}-eval',
            notes=note,
            resume="allow",
            config=args)
    else:
        run = wandb.init(
            project=f'{resstock_comstock}-eval',
            notes=note,
            config=args)
    
    model = MultiLabelClassifier(classes_per_task, 
                                 args.text_encoder_name, 
                                 freeze=args.freeze_text_encoder)
    model=model.to(args.device)

    train_dataset = EnergyPlusDataset(
        buildings_bench_path=Path(SYSCAPS_PATH),
        index_file=f'{resstock_comstock}_train_seed=42.idx',
        resstock_comstock=resstock_comstock,
        syscaps_split=args.caption_split,
        tokenizer = args.text_encoder_name,
        return_full_year=True,
        include_text = False
    )

    val_dataset = EnergyPlusDataset(
        buildings_bench_path=Path(SYSCAPS_PATH),
        index_file=f'{resstock_comstock}_hyperparam_val_seed=42.idx',
        resstock_comstock=resstock_comstock,
        syscaps_split=args.caption_split,
        tokenizer = args.text_encoder_name,
        return_full_year=True,
        include_text = True
    )

   
    train_dataloader = torch.utils.data.DataLoader(
            train_dataset, 
            batch_size=args.batch_size,
            drop_last=False, 
            worker_init_fn=utils.worker_init_fn,
            collate_fn=train_dataset.collate_fn(),
            shuffle=True,
            num_workers=8,
            pin_memory=True)
        

    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, 
        batch_size=args.batch_size,
        drop_last=False,
        num_workers = 8,
        shuffle=False,
        worker_init_fn=utils.worker_init_fn,
        collate_fn=val_dataset.collate_fn())
    
  
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    
    early_stopping_flag = False
    if args.resume_from_checkpoint != '':
        model, optimizer, _, batch_step , best_val_loss, patience_counter = utils.load_model_checkpoint(
            checkpoint_dir / args.resume_from_checkpoint, model, int(args.device.split(':')[1]), optimizer)
        print(f'successfully loaded model checkpoint {args.resume_from_checkpoint}...')
        batch_step = args.batch_step
        starting_epoch = batch_step // len(train_dataset) # how many times have we passed over all buidings?
    else:
        best_val_loss = 1e9
        patience_counter = 0
        batch_step = 0
        starting_epoch = 0
    
    if not args.evaluate:
        for epoch in range(starting_epoch, args.max_epochs):
           # train_sampler.set_epoch(epoch)
            if early_stopping_flag:
                break
            model.train()

            for batch in tqdm(train_dataloader):
                
                optimizer.zero_grad()

                for k,v in batch.items():
                    if torch.is_tensor(v):
                        batch[k] = v.to(args.device)
                # convert this into labels
                onehot = batch['attributes_onehot']
                labels = []
                i = 0
                for c in classes_per_task:
                    labels += [
                        onehot[:,0,i:i+c]
                    ]
                    i += c
                with torch.cuda.amp.autocast():
                    logits_per_head = model(batch)
                    loss = 0
                    for logits,label in zip(logits_per_head, labels):
                        loss += nn.functional.cross_entropy(logits, label)
                    loss = loss / len(classes_per_task) # average task loss

                loss.backward()
                optimizer.step()

                if batch_step % 50 == 0:
                    #print(f'epoch {epoch} step {batch_step}, average task xent loss {loss.item()}')

                    wandb.log({
                        'train/loss': loss,
                        'train/epoch': epoch,   
                    }, step=batch_step)

                if batch_step % 500 == 0:
                    
                    val_loss, attribute_accuracy = evaluate_on_dataset(model, val_dataloader)
                    # compute average attribute_accuracy
                    mean_acc = 0
                    for k,v in attribute_accuracy.items():
                        mean_acc += v
                    mean_acc /= len(attribute_accuracy)
                    
                    #wandb_acc = {}
                    #wandb_acc['val/mean_attribute_accuracy'] = 100*mean_acc
                    column_data = []
                    row_data = []
                    #if resstock_comstock == 'comstock':
                    for k,v in attribute_accuracy.items():
                        print(f'attribute {k} pred. accuracy (%) :: {100*v:.3f}')
                        #wandb_acc[f'val/attribute_accuracy/{k}'] = 100*v
                        column_data += [k]
                        row_data += [100*v]
                
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        patience_counter = 0
                        print('saving model...')
                        model_fname = f'classifier-caption-split-{args.caption_split}-seed={args.random_seed}'
                        utils.save_model_checkpoint(model,
                                                    optimizer,
                                                    optimizer,
                                                    batch_step,
                                                    best_val_loss,
                                                    patience_counter,
                                                    checkpoint_dir / f'{model_fname}.pt')
                    else:
                        patience_counter += 1
                        if patience_counter >= args.early_stopping_patience:
                            print('saving model and early stopping...')
                            model_fname = f'classifier-caption-split-{args.caption_split}-seed={args.random_seed}'
                            utils.save_model_checkpoint(model,
                                                        optimizer,
                                                        optimizer,
                                                        batch_step,
                                                        best_val_loss,
                                                        patience_counter,
                                                        checkpoint_dir / f'{model_fname}.pt')
                            break

                    wandb.log({
                        'val/loss': val_loss,
                        'val/best_val_loss': best_val_loss,
                        'val/epoch': epoch,
                        'val/mean_acc': 100*mean_acc,
                    }, step=batch_step)
                    table = wandb.Table(columns=column_data, data=[row_data])
                    wandb.log({'val/attribute_accuracy': table}, step=batch_step)
                    model.train()

                batch_step += 1 


    print('testing...')

    if resstock_comstock == 'comstock':
        test_files = ["comstock_buildings900k_test_seed=42.idx", "comstock_attribute_combos_seed=42.idx"]
    else:
        test_files = ["resstock_buildings900k_test_seed=42.idx"]

    avg_loss = 0
    attribute_results = {}
    
    for test_idx in test_files:
        test_dataset = EnergyPlusDataset(
            buildings_bench_path=Path(SYSCAPS_PATH),
            index_file=test_idx,
            resstock_comstock=resstock_comstock,
            syscaps_split=args.caption_split,
            tokenizer = args.text_encoder_name,
            return_full_year=True,
            include_text = True
        )

        test_dataloader = torch.utils.data.DataLoader(
            test_dataset, 
            batch_size=args.batch_size,
            drop_last=False,
            num_workers = 8,
            worker_init_fn=utils.worker_init_fn,
            collate_fn=val_dataset.collate_fn())
        

        test_loss, test_attribute_accuracy = evaluate_on_dataset(model, test_dataloader)
        
        avg_loss+= test_loss
        for k,v in test_attribute_accuracy.items():
            if k not in attribute_results:
                attribute_results[k] = []
            attribute_results[k] += [v]
            
            

    column_data = []
    row_data = []
    
    mean_acc = 0
    for k,v in attribute_results.items():
        v_ = np.mean(v)
        print(f'attribute {k} pred. accuracy (%) :: {100*v_:.3f}, ({v})')
        #wandb_acc[f'test/{test_idx}/attribute_accuracy/{k}'] = 100*v
        column_data += [k]
        row_data += [100*v_]
        mean_acc += 100*v_
        
    mean_acc /= len(attribute_results)
    
    wandb.log({
        'test/loss': avg_loss / len(test_files),
        'test/mean_acc': mean_acc,
    })
    table = wandb.Table(columns=column_data, data=[row_data])
    wandb.log({'test/attribute_accuracy': table})
