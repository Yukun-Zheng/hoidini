import gc
import os
import argparse
from functools import partial
from tqdm import tqdm
import numpy as np
import json
from hoidini.datasets.grab.grab_utils import parse_npz
from hoidini.datasets.smpldata_preparation import MAPPING_INTENT, intent_list, one_hot_vectors
from hoidini.inference_hoi_model import HoiResult
from hoidini.datasets.smpldata import slice_smpldata
from hoidini.datasets.smpldata import SmplData
from hoidini.inference_hoi_model import HoiResult
from hoidini.object_contact_prediction.cpdm_dno_conds import find_contiguous_static_blocks
import torch
import torch.nn.functional as F
from typing import Optional


# os.environ['CUDA_LAUNCH_BLOCKING']="1"
# os.environ['TORCH_USE_CUDA_DSA'] = "1"

from hoidini.datasets.dataset_smplrifke import HoiGrabDataset, collate_smplrifke_mdm

from torch.utils.data import DataLoader
from hoidini.eval.action_rec import LSTM_Action_Classifier_Hoidini, Conv_Action_Classifier_Hoidini, Transformer_Action_Classifier_Hoidini  

import wandb
from hoidini.eval.evaluate_statistical_metrics import calculate_accuracy

from hoidini.amasstools.smplrifke_feats import axis_angle_to_cont6d

import torch
normal_repr = torch.Tensor.__repr__
torch.Tensor.__repr__ = lambda self: f"{self.shape}_{normal_repr(self)}"

# Architecture dictionary for classifier selection
arch_dict = {
    'legasy': LSTM_Action_Classifier_Hoidini,
    'conv': Conv_Action_Classifier_Hoidini,
    'transformer': Transformer_Action_Classifier_Hoidini,
}


data_rep_to_n_features = {
    'loc_ang': 162,
    'loc_6d': 165,
    'loc': 159,
    'no_obj': 156,
    'hands_obj': 99,
    'hands_only': 90,
}

def process_smpldata_to_classification_batch(smpldata_lst, crop_size=60, contact_only=False, crop_start=False,  data_rep='loc_ang', method='random'):
    motion_features = []
    smpl_data_ret = []
    assert method in ['random', 'random_start', 'opt_1', 'opt_2'], f"Invalid method: {method}"

    for smpl_data in smpldata_lst:
        # Randomly crop the sequence
        min_start_idx = 0
        max_start_idx = smpl_data.joints.shape[0] - crop_size
        if max_start_idx == min_start_idx and min_start_idx == 0:
                start_idx=0
                end_idx = start_idx + crop_size
        elif 'random'in  method:
            if contact_only and smpl_data.contact is not None:
                any_contact = smpl_data.contact.any(dim=1)
                first_contact_idx = torch.where(any_contact)[0][0]
                last_contact_idx = torch.where(any_contact)[0][-1]
                min_start_idx = first_contact_idx.item()
                last_contact_idx = max(min_start_idx+1, last_contact_idx - crop_size)
                max_start_idx = min(last_contact_idx, max_start_idx).item()
                assert min_start_idx < max_start_idx, f"No enough contact in the sequence"
            if method == 'random_start':
                start_idx = min_start_idx
            else:
                start_idx = np.random.randint(min_start_idx, max_start_idx) 
            end_idx = start_idx + crop_size
            
        elif method == 'opt_1':
            start_idx, end_idx = get_longest_contact_range(smpl_data)

        elif method == 'opt_2':
            start, end = get_longest_contact_range(smpl_data)
            mid = (start + end) // 2
            start_idx = mid - 15
            end_idx = mid + 15
            
        smpl_data_cropped = smpl_data.cut(start_idx, end_idx)
        
        if method == 'opt_1':
            smpl_data_cropped  = resample(smpl_data_cropped, 30)

        smpl_data_ret.append(smpl_data_cropped)
        joints = smpl_data_cropped.joints.clone()  # [seqlen, 52, 3]
        obj_trans = smpl_data_cropped.trans_obj.clone()  # [seqlen, 3]
        obj_ang = smpl_data_cropped.poses_obj.clone()  # [seqlen, 3]
        obj_6d = axis_angle_to_cont6d(obj_ang).clone()  # [seqlen, 6]
        init_root_joint = joints[:1, :1].clone()

        # Bring everithing to origin
        joints -= init_root_joint
        obj_trans -= init_root_joint[0]

        # Flatten and concatenate
        if data_rep == 'loc_ang':
            features = torch.cat([joints.reshape(crop_size, -1), obj_trans, obj_ang], dim=-1)[None]
        elif data_rep == 'loc_6d':
            features = torch.cat([joints.reshape(crop_size, -1), obj_trans, obj_6d], dim=-1)[None]
        elif data_rep == 'loc':
            features = torch.cat([joints.reshape(crop_size, -1), obj_trans], dim=-1)[None]
        elif data_rep == 'no_obj':
            features = joints.reshape(crop_size, -1)[None]
        elif data_rep == 'hands_obj':
            hands = joints[:, -30:, :]
            features = torch.cat([hands.reshape(crop_size, -1), obj_trans, obj_6d], dim=-1)[None]
        elif data_rep == 'hands_only':
            hands = joints[:, -30:, :]
            features = hands.reshape(crop_size, -1)[None]
        else:
            raise ValueError(f"Invalid data representation: {data_rep}")
        motion_features.append(features)

    motion_features = torch.cat(motion_features, dim=0)

    return motion_features, smpl_data_ret

def load_hoi_results(exp_name, seq_len, data_rep='loc_ang'):

    results_json = '/home/dcor/roeyron/trumans_utils/results/cphoi_paper_experiments_fixed_keep_object_static/result_paths_per_experiment.json'
    with open(results_json, 'r') as f:
        results_dict = json.load(f)
    our_dataset_paths = results_dict[exp_name]

    hoi_results = [HoiResult.load(p) for p in our_dataset_paths]
    intent_vecs = [one_hot_vectors(parse_npz(hoi_result.grab_seq_path)['motion_intent'], intent_list, MAPPING_INTENT) for hoi_result in hoi_results]
    smpl_data_gen = [hoi_result.smpldata for hoi_result in hoi_results]
    smpl_data_gen = [slice_smpldata(s, 15, None) for s in smpl_data_gen]
    gen_features, _ = process_smpldata_to_classification_batch(smpl_data_gen, crop_size=seq_len, data_rep=data_rep)

    return gen_features, intent_vecs


def evaluate_accuracy_on_loader(data_loader, model, args, device, contact_only=False):
    joints_list = []
    gt_labels_list = []
    with torch.no_grad():
        for motion, kwargs in data_loader:
            x, _ = process_smpldata_to_classification_batch(kwargs['y']['smpldata_lst'], crop_size=args.seq_len, contact_only=contact_only, data_rep=args.data_rep)
            x = x.to(device)
            y = kwargs['y']['intent_vec'].to(device)
            for i in range(x.shape[0]):
                joints_list.append(x[i][None].detach().to(device))
                gt_labels_list.append(y[i][None].detach().to(device))
    acc = calculate_accuracy(joints_list, 29, model, gt_labels_list)
    return acc

def evaluate_accuracy_on_features(features, intent_vecs, model, args, device, contact_only=False):
    joints_list = []
    gt_labels_list = []
    with torch.no_grad():
        for i in range(len(features)):
            x = features[i]
            joints_list.append(x[None].detach().to(device))
    acc = calculate_accuracy(joints_list, 29, model, [torch.from_numpy(e).to(device)[None] for e in intent_vecs])
    return acc

def load_classifier(classifier_dir):
    """
    Load a trained classifier from a directory
    Args:
        classifier_dir: Path to directory containing args.json and best.pt
    Returns:
        model: The loaded model
    """
    # Load args
    with open(os.path.join(classifier_dir, 'args.json'), 'r') as f:
        args = json.load(f)
        # print("\nLoaded args:")
        # print(json.dumps(args, indent=2))

    # Create model instance
    model = arch_dict[args['arch']](
        n_features=data_rep_to_n_features[args['data_rep']],  # This should match training, or could be args['n_features'] if saved
        hidden_dim=args.get('hidden_dim', 128),
        label_size=29,   # This should match training, or could be args['label_size'] if saved
        num_layers=args.get('num_layers', 2)
    )
    model.load_state_dict(torch.load(os.path.join(classifier_dir, 'best.pt'), map_location='cpu'))
    model.eval()
    return model, args

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--is_debug', action='store_true', help='If set, use lim=10 for debug mode, else lim=None')
    parser.add_argument('--seq_len', type=int, default=30, help='Sequence length for dataset')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training and evaluation')
    parser.add_argument('--exp_name', type=str, default='hoi_lstm_classifier', help='Experiment name for wandb and checkpoints')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    parser.add_argument('--n_epochs', type=int, default=300, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--save_interval', type=int, default=10, help='Save checkpoint every N epochs')
    parser.add_argument('--no_wandb', action='store_true', help='If set, disables wandb logging')
    parser.add_argument('--hidden_dim', type=int, default=128, help='Hidden dimension for LSTM')
    parser.add_argument('--num_layers', type=int, default=2, help='Number of LSTM layers')
    parser.add_argument('--arch', type=str, default='legasy', choices=list(arch_dict.keys()), help='Model architecture to use')
    parser.add_argument('--train_contact_only', action='store_true', help='If set, only use contact features for training')
    parser.add_argument('--eval_contact_only', action='store_true', help='If set, only use contact features for testing')
    parser.add_argument('--use_lr_annealing', action='store_true', help='If set, enables learning rate annealing (StepLR)')
    parser.add_argument('--data_rep', type=str, default='loc_ang', choices=list(data_rep_to_n_features.keys()), help='Data representation to use')
    args = parser.parse_args()

    # Save args to JSON file for reproducibility and loading
    save_dir = f'./save_classifier/{args.exp_name}'
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)

    use_wandb = not args.no_wandb
    if use_wandb:
        wandb.init(project='hoi_action_classifier3', name=args.exp_name, config=vars(args))

    dataset_path = "/home/dcor/roeyron/trumans_utils/DATASETS/DATA_GRAB_RETARGETED"
    lim = 10 if args.is_debug else None
    batch_size = 3 if args.is_debug else args.batch_size
    train_set = HoiGrabDataset(dataset_path, seq_len=args.seq_len, lim=lim, 
                               use_cache=True, features_string=None, grab_seq_paths=None, grab_split='train')   
    test_set = HoiGrabDataset(dataset_path, seq_len=args.seq_len, lim=lim, 
                              use_cache=True, features_string=None, grab_seq_paths=None, grab_split='test')

    gen_features, gen_intent_vecs = load_hoi_results('1_ours_0', args.seq_len, data_rep=args.data_rep)
    gen_infer_features, gen_infer_intent_vecs = load_hoi_results('1b_our_inference_only_0', args.seq_len, data_rep=args.data_rep)   

    collate_fn = partial(collate_smplrifke_mdm, pred_len=args.seq_len, context_len=15)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    model = arch_dict[args.arch](n_features=data_rep_to_n_features[args.data_rep], hidden_dim=args.hidden_dim, label_size=29, num_layers=args.num_layers).to(device)
    # print(f'ARGS: \n{args}')
    # print(f'Number of model parameters: {sum(p.numel() for p in model.parameters())}')
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    if args.use_lr_annealing:
        from torch.optim.lr_scheduler import StepLR
        scheduler = StepLR(optimizer, step_size=50, gamma=0.5)
    if use_wandb:
        wandb.watch(model, log='all', log_freq=100)

    best_acc = 0
    os.makedirs(f'./save_classifier/{args.exp_name}', exist_ok=True)

    for epoch in tqdm(range(args.n_epochs)):
        model.train()
        running_loss = 0
        for motion, kwargs in train_loader:
            x, _ = process_smpldata_to_classification_batch(kwargs['y']['smpldata_lst'], crop_size=args.seq_len, contact_only=args.train_contact_only, data_rep=args.data_rep)  # [bs, crop_size, n_features]
            x = x.to(device)
            y = kwargs['y']['intent_vec'].to(device)  # [bs, 29]
            model.batch_size = x.shape[0]
            optimizer.zero_grad()
            loss, _ = model(x, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * x.shape[0]
        avg_loss = running_loss / len(train_loader.dataset)
        if use_wandb:
            wandb.log({'train_loss': avg_loss}, step=epoch)
            wandb.log({'lr': optimizer.param_groups[0]['lr']}, step=epoch)

        if args.use_lr_annealing:
            scheduler.step()

        # Evaluation
        if (epoch+1) % args.save_interval == 0 or epoch == args.n_epochs-1:
            model.eval()
            acc = evaluate_accuracy_on_loader(test_loader, model, args, device, contact_only=args.eval_contact_only)
            if use_wandb:
                wandb.log({'test_accuracy': acc}, step=epoch)
            print(f"Epoch {epoch}: test_accuracy={acc:.4f}")
            if acc > best_acc:
                best_acc = acc
                torch.save(model.state_dict(), f'./save_classifier/{args.exp_name}/best.pt')
            torch.save(model.state_dict(), f'./save_classifier/{args.exp_name}/latest.pt')

            train_acc = evaluate_accuracy_on_loader(train_loader, model, args, device, contact_only=args.eval_contact_only)
            if use_wandb:
                wandb.log({'train_accuracy': train_acc}, step=epoch)
            print(f"Epoch {epoch}: train_accuracy={train_acc:.4f}")

            # Evaluate on gen_features
            gen_acc = evaluate_accuracy_on_features(gen_features, gen_intent_vecs, model, args, device, contact_only=args.eval_contact_only)
            if use_wandb:
                wandb.log({'gen_accuracy': gen_acc}, step=epoch)
            print(f"Epoch {epoch}: gen_accuracy={gen_acc:.4f}")
            
            gen_acc_infer = evaluate_accuracy_on_features(gen_infer_features, gen_infer_intent_vecs, model, args, device, contact_only=args.eval_contact_only)
            if use_wandb:
                wandb.log({'gen_accuracy_infer': gen_acc_infer}, step=epoch)
            print(f"Epoch {epoch}: gen_accuracy_infer={gen_acc_infer:.4f}")
            

    if use_wandb:
        wandb.finish()


def resample(smpl_data: SmplData, n_frames: int) -> SmplData:
    def _resample_tensor(x: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if x is None:
            return None
        T = x.shape[0]
        if T == n_frames:
            return x
        # flatten non-time dims into "features"
        rest = x.shape[1:]
        x_flat = x.reshape(T, -1).transpose(0, 1).unsqueeze(0)  # (1, features, T)
        y_flat = F.interpolate(x_flat, size=n_frames, mode='linear', align_corners=True)
        y = y_flat.squeeze(0).transpose(0, 1).reshape(n_frames, *rest)
        return y

    data_dict = smpl_data.to_dict()
    for key, val in data_dict.items():
        if isinstance(val, torch.Tensor) and key != 'contact':
            data_dict[key] = _resample_tensor(val)
    return SmplData(**data_dict)

def remove_short_false_segments(arr: torch.Tensor, min_length: int) -> torch.Tensor:
    arr = arr.bool()
    padded = torch.cat([torch.tensor([False], device=arr.device), arr, torch.tensor([False], device=arr.device)])
    diffs = padded[1:] != padded[:-1]
    idxs = torch.nonzero(diffs).flatten()
    starts, ends = idxs[::2], idxs[1::2]
    for s, e in zip(starts, ends):
        if not arr[s] and (e - s) < min_length:
            arr[s:e] = True
    return arr

def get_longest_contact_range(smpldata: SmplData):
    has_contact = (smpldata.contact > 0.5).any(dim=1)  # (seq, n_anchors)
    has_contact = remove_short_false_segments(has_contact, min_length=2)
    contact_blocks = find_contiguous_static_blocks(~has_contact)
    start, end = sorted(contact_blocks, key=lambda rng: rng[1] - rng[0])[-1]
    return start, end