from pathlib import Path 
import argparse
import torch

if __name__ == '__main__':
    """
    Merge the weights of N PyTorch checkpoints of the same model.

    Usage: python merge_weights.py --model_fnames "model1.ckpt,model2.ckpt" --output "merged.ckpt" --ckpt_dir "path/to/checkpoints"
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_fnames', type=str, required=True, 
                        help="model file names seperated by \",\"")
    parser.add_argument('--output', type=str, required=True,
                        help='output filename')
    parser.add_argument('--ckpt_dir', type=str, required=True, 
                        help="directory of saved checkpoints")
    parser.add_argument('--map_location', type=str, default='cuda:0',
                        help='map location argument for torch.load')
    
    args = parser.parse_args()

    args.model_fnames = [model_fname.strip() for model_fname in args.model_fnames.split(",") if model_fname != ""]

    ckpt_dir = Path(args.ckpt_dir)

    all_ckpts = []
    for f in args.model_fnames:
        all_ckpts += [ torch.load(ckpt_dir / f, map_location=args.map_location)['model'] ]

    new_state_dict = {'model': {}}
    # for each parameter
    for k in all_ckpts[0].keys():
        agg= []
        # aggregate across all checkpoints
        for ckpt in all_ckpts:
            agg += [ckpt[k]]
        
        #print(agg)
        try:
            new_state_dict['model'][k] = torch.mean(torch.stack(agg,0), dim=0)
        except RuntimeError:
            if 'l_kernel' in k:
                new_state_dict['model'][k] = agg[0]
                print(k, agg)
        #print(new_state_dict['model'][k])

    torch.save(new_state_dict, ckpt_dir / args.output)