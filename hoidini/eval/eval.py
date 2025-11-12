import gc
import os
import glob
import json
from tqdm import tqdm
import sys

# Early: map physical GPU to CUDA:0 before importing torch
def _early_set_cuda_visible_from_args_env():
    phys_env = os.environ.get("PHYSICAL_GPU_ID")
    if phys_env is not None:
        try:
            phys_val = int(phys_env)
            if phys_val >= 0:
                os.environ["CUDA_VISIBLE_DEVICES"] = str(phys_val)
                return
        except ValueError:
            pass
    argv = sys.argv[1:]
    for i, arg in enumerate(argv):
        if arg.startswith("--device"):
            # Support both --device=3 and --device 3
            val = None
            if "=" in arg:
                val = arg.split("=", 1)[1]
            elif i + 1 < len(argv):
                val = argv[i + 1]
            if val is not None:
                try:
                    phys_val = int(val)
                    if phys_val >= 0 and phys_val != 0:
                        os.environ["CUDA_VISIBLE_DEVICES"] = str(phys_val)
                        return
                except ValueError:
                    pass

_early_set_cuda_visible_from_args_env()

import torch

from hoidini.amasstools.geometry import matrix_to_axis_angle
from hoidini.blender_utils.visualize_mesh_figure_blender import get_smpl_template
from hoidini.closd.diffusion_planner.utils import dist_util
from hoidini.datasets.dataset_smplrifke import HoiGrabDataset
from hoidini.datasets.grab.grab_utils import grab_seq_path_to_object_name, load_mesh
from hoidini.datasets.smpldata import SmplData, SmplModelsFK, slice_smpldata, upsample_smpldata
from hoidini.inference_hoi_model import  HoiResult
from hoidini.objects_fk import ObjectModel

from hoidini.eval.action_rec import LSTM_Action_Classifier
from hoidini.eval.evaluate_statistical_metrics import *
from hoidini.eval.train_our_evaluator import load_classifier, process_smpldata_to_classification_batch
from hoidini.resource_paths import GRAB_DATA_PATH, PROJECT_ROOT
from hoidini.general_utils import TMP_DIR


class Evaluator:
    def __init__(self,name, files, args,  should_run_gt=False, imos=False, n_frames=30):
        self.debug = args.debug
        self.args=args
        self.should_run_gt = should_run_gt

        '''Load the IMoS action classifier model'''
        self.action_rec_model = LSTM_Action_Classifier().float()

        weights_path = args.action_rec_weights
        if not os.path.exists(weights_path):
            weights_path = os.path.join(PROJECT_ROOT, weights_path)
        m = torch.load(open(weights_path, 'rb'))
        self.action_rec_model.load_state_dict(m['model_pose'])
        self.action_rec_model = self.action_rec_model.to(dist_util.dev())
        
        self.our_action_rec_model, class_args = load_classifier(args.classifier)
        self.our_action_rec_model = self.our_action_rec_model.to(dist_util.dev())
        self.class_rep = class_args['data_rep']
        print('Model loaded')
        self.files = files
        self.name = name
        self.imos = imos
        self.n_frames = n_frames
        self.do_geometric_eval =  not args.mini

        dataset_path = args.dataset_path

        
        self.dataset = HoiGrabDataset(dataset_path, seq_len=60, lim=None, use_cache=True, features_string='hoi_no_contact' if imos else 'hoi', grab_seq_paths=None, grab_split='test')             
        
        self.joints_gen = []
        self.action_pred_gen = []
        self.activations_gen = []
        
        self.feats_for_classifier_gen = []
        self.feats_for_classifier_gt = []
        
        self.our_action_pred_gen = []
        self.our_activations_gen = []
                
        self.our_action_pred_gt = []
        self.our_activations_gt = []
        
        self.joints_gt = []
        self.action_pred_gt = []
        self.activations_gt = []
        
        self.intent_gt = []
        
        self.body_verts_gen = []
        self.body_verts_gt = []        
        
        self.obj_faces = []
        self.obj_verts_gen = []
        self.obj_verts_gt = []
        
        self.names = []
        
        self.table_hights_gen = []
        self.table_hights_gt = []
        
        self.body_faces = shut_your_mouth(get_smpl_template('smplx')[0])
        
    def proccess(self):
        pbar = tqdm(total=len(self.dataset), desc='process')
        for i in range(len(self.dataset)):
            gt = self.dataset[i]
            pbar.update(1)
        # for i, gt in enumerate(tqdm(self.dataset, leave=False, desc='proccess')):
            if self.debug and i > 4:
                break
            
            # motion -  batch, 556, 1, Frames
            
            if 'lift' in gt['name']: # ignored by imos
                continue
            
            smpl_data_gt =  gt['smpldata'] #  get_extended_smpldata(grab_seq_id_to_seq_path(gt['name']))['smpldata']
            # smpl_data_gt =  get_extended_smpldata(grab_seq_id_to_seq_path(gt['name']))['smpldata']

            if not self.imos:
                smpl_data_gen = self._sample_us(name=gt['name'], text=gt['text']).to(dist_util.dev()) 
                fist_obj_poses_gen = smpl_data_gen.poses_obj[:1],  smpl_data_gen.trans_obj[:1]

                smpl_data_gen = slice_smpldata(smpl_data_gen, 15, None) 
                smpl_data_gen = self.dataset.feature_processor.encode(smpl_data_gen)
                smpl_data_gen = self.dataset.feature_processor.decode(smpl_data_gen)

            else:
                smpl_data_gen = self._sampl_imos(seq_path=gt['seq_path'], trans_correction=smpl_data_gt.trans[0]).to(dist_util.dev()) 
                fist_obj_poses_gen = smpl_data_gen.poses_obj[:1],  smpl_data_gen.trans_obj[:1]
                smpl_data_gen = slice_smpldata(smpl_data_gen, 4, None) 
                smpl_data_gen= upsample_smpldata(smpl_data_gen, 2) # adjust framerate to 20 fps
                smpl_data_gen = self.dataset.feature_processor.encode(smpl_data_gen)
                smpl_data_gen = self.dataset.feature_processor.decode(smpl_data_gen)
                # np.save(os.path.join('/home/dcor/haimsawdayee/hoi/save/imos', f'{gt["name"].replace("/", "_")}_smpldata.npy'),
                #         smpl_data_gen.to('cpu').to_dict(), allow_pickle=True)
            
            fist_obj_poses_gt = smpl_data_gt.poses_obj[:1],  smpl_data_gt.trans_obj[:1]
            smpl_data_gt = smpl_data_gt.cut(15, None)
            smpl_data_gt=smpl_data_gt.to(dist_util.dev())
            
            if self.should_run_gt:
                smpl_data_gen = smpl_data_gt
                fist_obj_poses_gen = fist_obj_poses_gt

                
            # if self.n_frames is not None and len(smpl_data_gen) > self.n_frames:
            #     start = random.randint(0, len(smpl_data_gen) - self.n_frames)
            #     smpl_data_gen = slice_smpldata(smpl_data_gen, start, start+self.n_frames)  
                
            # if self.n_frames is not None and len(smpl_data_gt) > self.n_frames:
            #     start = random.randint(0, len(smpl_data_gt) - self.n_frames)
            #     smpl_data_gt = slice_smpldata(smpl_data_gt, start, start+self.n_frames)             
            
            feats, datas = process_smpldata_to_classification_batch([smpl_data_gt, smpl_data_gen], crop_size=self.n_frames, contact_only=True,
                                                                    data_rep=self.class_rep, method=self.args.crop_method ) 
            smpl_data_gt, smpl_data_gen = datas
            feats_gt, feats_gen = feats
            self.feats_for_classifier_gen.append(feats_gen)
            self.feats_for_classifier_gt.append(feats_gt)
        
            self.names.append(gt['name'])
            #### Roey --->
            smpl_data_gt = smpl_data_gt.to(dist_util.dev())
            #### Roey <---
            body_joints_gt = smpl_data_gt.joints.to(dist_util.dev())
            body_joints_gen = smpl_data_gen.joints # frames joints 3
            
            smpl_fk = SmplModelsFK.create('smplx', len(body_joints_gen), device=dist_util.dev())            
            body_verts_gen = smpl_fk.smpldata_to_smpl_output_batch([smpl_data_gen], cancel_offset=not self.imos and not self.should_run_gt)[0].vertices # TODO not cancel to imos
            
            smpl_fk = SmplModelsFK.create('smplx', len(body_joints_gt), device=dist_util.dev())
            body_verts_gt =  smpl_fk.smpldata_to_smpl_output_batch([smpl_data_gt], cancel_offset=False)[0].vertices
            
            obj_verts_T, obj_faces = load_mesh(grab_seq_path_to_object_name(gt['seq_path']), n_simplify_faces=1000)
             
            obj_poses_gt = smpl_data_gt.poses_obj,  smpl_data_gt.trans_obj
            obj_poses_gen = smpl_data_gen.poses_obj.to(dist_util.dev()),  smpl_data_gen.trans_obj.to(dist_util.dev())
            
            obj_model = ObjectModel(v_template=obj_verts_T, batch_size=len(body_joints_gt))
            obj_model = obj_model.to(dist_util.dev())
            obj_verts_gt = obj_model(*obj_poses_gt).vertices
            
            obj_model = ObjectModel(v_template=obj_verts_T, batch_size=1)
            table_hight_gt = obj_model(*fist_obj_poses_gt).vertices[:, 2].min() # the motion alwais starts with the object on the table
            
            obj_model = ObjectModel(v_template=obj_verts_T, batch_size=len(body_joints_gen))
            obj_model = obj_model.to(dist_util.dev())
            obj_verts_gen = obj_model(*obj_poses_gen).vertices
            
            obj_model = ObjectModel(v_template=obj_verts_T, batch_size=1)
            obj_model = obj_model.to(dist_util.dev())
            table_hight_gen = obj_model(*fist_obj_poses_gen).vertices[:, 2].min() # the motion alwais starts with the object on the table

            self.table_hights_gen.append(table_hight_gen)
            self.table_hights_gt.append(table_hight_gt)
            
            self.obj_faces.append(obj_faces)
            self.obj_verts_gen.append(obj_verts_gen)
            self.obj_verts_gt.append(obj_verts_gt) 
            
            self.body_verts_gen.append(body_verts_gen)
            self.body_verts_gt.append(body_verts_gt)

            
            self.joints_gen.append(body_joints_gen)
            self.joints_gt.append(body_joints_gt)
            
            intent_pred_gen, activations_gen = self.action_rec_model.predict(body_joints_gen[:, :22, :])
            self.action_pred_gen.append(intent_pred_gen)
            self.activations_gen.append(activations_gen)
            
            intent_pred_gt, activations_gt = self.action_rec_model.predict(body_joints_gt[:, :22, :])
            self.action_pred_gt.append(intent_pred_gt)
            self.activations_gt.append(activations_gt)

            intent_pred_gen, activations_gen = self.our_action_rec_model.predict2(feats_gen.unsqueeze(0))  if args.pred2 else self.our_action_rec_model.predict(feats_gen.unsqueeze(0)) 
            self.our_action_pred_gen.append(intent_pred_gen)
            self.our_activations_gen.append(activations_gen)
            
            intent_pred_gt, activations_gt = self.our_action_rec_model.predict2(feats_gt.unsqueeze(0))  if args.pred2 else  self.our_action_rec_model.predict(feats_gt.unsqueeze(0))
            self.our_action_pred_gt.append(intent_pred_gt)
            self.our_activations_gt.append(activations_gt)


            self.intent_gt.append(torch.tensor(gt['intent_vec']).unsqueeze(0))
    
        # self.joints_gen = torch.stack(self.joints_gen).to(self.device) # batch, 200, 22, 3
        # self.joints_gt = torch.stack(self.joints_gt).to(self.device) # batch, 200, 22, 3
        self.intent_gt = torch.stack(self.intent_gt).to(dist_util.dev()) # batch, 1, 29
        
        self.action_pred_gen = torch.stack(self.action_pred_gen).to(dist_util.dev())
        self.action_pred_gt = torch.stack(self.action_pred_gt).to(dist_util.dev())
        
        self.activations_gen = torch.stack(self.activations_gen).to(dist_util.dev())
        self.activations_gt = torch.stack(self.activations_gt).to(dist_util.dev())
        
                
        self.our_action_pred_gen = torch.stack(self.our_action_pred_gen).to(dist_util.dev())
        self.our_action_pred_gt = torch.stack(self.our_action_pred_gt).to(dist_util.dev())
        
        self.our_activations_gen = torch.stack(self.our_activations_gen).to(dist_util.dev())
        self.our_activations_gt = torch.stack(self.our_activations_gt).to(dist_util.dev())
        
        # for i in range(len(self.joints_gen)):
        #     obj_mesh = trimesh.Trimesh(vertices=self.body_verts_gen[i][0].cpu().numpy(), faces=self.body_faces)
        #     if obj_mesh.is_watertight:
        #         print(f'name: {self.names[i]}', flush=True)
        # exit()
        
    def calculate(self):
        # if not self.debug:
        #     torch.save(vars(self),f'/home/dcor/haimsawdayee/hoi/evals2/res_{self.name}_{self.debug}.pt')
            
        res = {}
        res.update(self._calc_realizem_metrics())
        res.update(self._calc_conditioning_metrics())
        if self.do_geometric_eval:
            res.update(self._calc_interaction_metrics())
        print(res)
        return res
    
    def _calc_realizem_metrics(self):
        ret = {}
        # body FID
        fid = evaluate_fid(self.joints_gt, self.joints_gen, 29, self.action_rec_model, self.intent_gt).item()
        ret['fid'] = fid
        
        fid = evaluate_fid_our(self.our_activations_gt, self.our_activations_gen).item()
        ret['our_fid'] = fid
        # hand/object FID
        
        # Diversity
        divers_gen = calculate_diversity_(self.activations_gen, self.intent_gt, 29).item()
        divers_gt = calculate_diversity_(self.activations_gt, self.intent_gt, 29).item()
        ret['diversity_gen'] = divers_gen
        ret['diversity_gt'] = divers_gt
        
        divers_gen = calculate_diversity_(self.our_activations_gen, self.intent_gt, 29).item()
        divers_gt = calculate_diversity_(self.our_activations_gt, self.intent_gt, 29).item()
        ret['our_diversity_gen'] = divers_gen
        ret['our_diversity_gt'] = divers_gt
        
        
        # AVE - Average Variance Error (AVE) [GCO*21], which measures the variance error between the joint positions. mean_sampl(mean_joint(l2(var_gt, var_pred)))
        mean_ave_loss = torch.zeros((len(self.joints_gen)))
        for idx in range(len(self.joints_gen)):
            m_1 = self.joints_gen[idx]
            m_2 = self.joints_gt[idx]
            mean_ave_loss[idx] = calc_AVE(m_1, m_2).item()
        ret['mean_ave'] = torch.mean(mean_ave_loss).item()
        
         
        # jt_gt = torch.cat(, dim=0) # 114,15,22,3
        # [self.joints_gen[i].shape[0] - self.joints_gt[i].shape[0] for i in range(10)]
        # mean_ape_loss = mean_l2di_(pad_sequence(self.joints_gen, batch_first =True), pad_sequence(self.joints_gt, batch_first =True)).item()
        # print("mean APE/MPJPE: ", mean_ape_loss) # MPJPE
        # ret['mean_ape'] = mean_ave_loss.item()
        # foot skating
        return ret
    
    def _calc_conditioning_metrics(self):
        ret = {}
        # Recognition Accuracy - action object matrix is very sparse, not sure if we need it?, each object has ~ 5/7 actions
        rec_accuracy = calculate_accuracy(self.joints_gen, 29, self.action_rec_model, self.intent_gt).item()
        # print("Recognition Accuracy", rec_accuracy)
        ret ['rec_accuracy'] = rec_accuracy
        
        rec_accuracy = calculate_accuracy(self.joints_gt, 29, self.action_rec_model, self.intent_gt).item()
        # print("Recognition Accuracy GT", rec_accuracy)
        ret ['rec_accuracy_gt'] = rec_accuracy
        
        rec_accuracy = topk_accuracy(self.our_action_pred_gt.squeeze(), self.intent_gt.argmax(dim=-1))
        # print("our Recognition Accuracy GT", rec_accuracy)
        ret ['our_rec_accuracy_gt'] = rec_accuracy
                
        rec_accuracy = topk_accuracy(self.our_action_pred_gen.squeeze(), self.intent_gt.argmax(dim=-1))
        # print("our Recognition Accuracy gen", rec_accuracy)
        ret ['our_rec_accuracy_gen'] = rec_accuracy
        
        # Multimodality # imos did the distacne between two embbedings of motions
        multimodality = calculate_multimodality_(self.activations_gen, self.intent_gt.squeeze(1), 29).item()
        # print("Multimodality", multimodality)
        ret ['multimodality'] = multimodality
        return ret
    
    def _calc_interaction_metrics(self):
        ret={}     
        res = get_dist_metrics(self.obj_verts_gen, self.obj_faces, self.body_verts_gen, self.body_faces, self.table_hights_gen, debug=self.debug)
        # print(f"dist gen: {res}") 
        ret['dist_gen'] = res
        
        # if self.should_run_gt:
        #     res = get_dist_metrics(self.obj_verts_gt, self.obj_faces, self.body_verts_gt, self.body_faces,self.table_hights_gt, debug=self.debug)
        #     # print(f"dist gt: {res}") 
        #     ret['dist_gt'] = res
        
        # penetration = get_peneration(self.obj_verts_gen, self.obj_faces, self.body_verts_gen, self.body_faces, self.names)
        # print(f"mean intersection volume gen: {penetration}")
        # ret['mean_intersection_volume_gen'] = penetration
        
        # if self.should_run_gt:
        #     penetration = get_peneration(self.obj_verts_gt, self.obj_faces, self.body_verts_gt, self.body_faces)
        #     print(f"mean intersection volume gt: {penetration}") 
        #     ret['mean_intersection_volume_gt'] = penetration

        # penetration -
            # collision depth
        # hand 6dof FID- nice to have
        # contact sliding - on verts
        return ret
        
    def _sample_us(self, name, text):
        paths = [file for file in self.files if name.replace('/', '_') in file]
        smpldata = HoiResult.load(paths[0]).smpldata
        add_joints_to_smpldata(smpldata, imos=False)
        return smpldata
                

    def _sampl_imos(self, seq_path, trans_correction):
        paths = glob.glob(os.path.join(self.files,   '_'.join(seq_path.split('/')[-2:])[:-4] + '_*.npz'))
        assert len(paths)==1
        data = torch.load(paths[0], weights_only=False, map_location='cpu')
        poses = torch.concat([
            data['sbj_p']['global_orient'],
            data['sbj_p']['body_pose'],
            data['sbj_p']['left_hand_pose'],
            data['sbj_p']['right_hand_pose'],
            ], dim=1)
        
        v = {
            'poses':  matrix_to_axis_angle(poses).view(len(poses), -1), # F, 52, 3
            'trans': data['sbj_p']['transl'] + trans_correction, # F, 3
            'joints': None,
            
            'poses_obj': data['obj_p']['global_orient'],  # F, 3
            'trans_obj': data['obj_p']['transl']+trans_correction, # F, 3
        }
        smpldata = SmplData(**v)
        add_joints_to_smpldata(smpldata, imos=False)
        return smpldata
    
def shut_your_mouth(faces):
    faces_to_add = [[2930,2933,2941],
                    [1852,1835,1830],
                    [8958,2783,2782],
                    [1666,8958,2782],
                    [1665,1666,2782],
                    [1665,2782,2854],
                    [1739,1665,2854],
                    [1739,2854,2857],
                    [1742,1739,2857],
                    [2930,2941,8941],
                    [8941,1852,1830],
                    [2930,8941,1830],
                    [2945,2930,1830],
                    [2945,1830,1862],
                    [2943,2945,1862],
                    [1746,1747,1742],
                    [1746,1742,2857],
                    [2857,2862,2861],
                    [1746,2857,2861],
                    [2708,2709,2943],
                    [2708,2943,1862],
                    [1860,1573,1572],
                    [1862,1860,1572],
                    [2708,1862,1572],
                    [1594,1595,1746],
                    [1594,1746,2861],
                    [2861,2731,2730],
                    [1594,2861,2730],
                    [1572,1594,2730],
                    [2708,1572,2730],
                    ]
    return np.concatenate([faces, faces_to_add], axis=0)

def save_verts_and_faces_to_obj(verts, faces=None, filepath="output.obj"):
    """
    Save a list of vertices and optionally faces to an OBJ file.

    Args:
        verts (list of list of float): A list of 3D vertices, where each vertex is [x, y, z].
        faces (list of list of int, optional): A list of triangular faces, where each face is [i1, i2, i3].
                                               Indices are 1-based for OBJ format. Defaults to None.
        filepath (str): The path to save the OBJ file. Defaults to "output.obj".
    """
    with open(filepath, 'w') as obj_file:
        obj_file.write("# OBJ file generated from vertices and faces\n")
        
        # Write vertices
        for vert in verts:
            obj_file.write(f"v {vert[0]} {vert[1]} {vert[2]}\n")
        
        # Write faces if provided
        if faces is not None:
            for face in faces:
                obj_file.write(f"f {face[0] + 1} {face[1] + 1} {face[2] + 1}\n")  # OBJ format uses 1-based indexing

    print(f"OBJ file saved to {filepath}")
    

def add_joints_to_smpldata(smpldata: SmplData, imos=False):
    smpl_fk = SmplModelsFK.create('smplx', len(smpldata), device=dist_util.dev())
    with torch.no_grad():
        smpl_output = smpl_fk.smpldata_to_smpl_output(smpldata.to(dist_util.dev()), cancel_offset=not imos)
    smpldata.joints = smpl_output.joints.to(smpldata.poses.device)

def resolve_relative_path(path):
    """
    Resolve a relative path to an absolute path using PROJECT_ROOT as base.
    
    Args:
        path: Path that might be relative or absolute
        
    Returns:
        str: Absolute path
    """
    if os.path.isabs(path):
        return path
    return os.path.join(PROJECT_ROOT, path)

def load_experiments_json(json_path):
    """
    Load experiments JSON file and resolve all relative paths to absolute paths.
    
    Args:
        json_path: Path to the JSON file (can be relative or absolute)
        
    Returns:
        dict: Dictionary with experiment names as keys and lists of absolute file paths as values
    """
    # Resolve the JSON file path
    json_path = resolve_relative_path(json_path)
    
    # Load the JSON file
    with open(json_path, 'r') as f:
        exps = json.load(f)
    
    # Resolve all file paths in the experiments
    resolved_exps = {}
    for exp_name, file_list in exps.items():
        resolved_exps[exp_name] = [resolve_relative_path(file_path) for file_path in file_list]
    
    return resolved_exps

def main_ablation(args, mod=None):
    with torch.no_grad():
        if args.debug:
            print('DEBUG!!!!! not actual results')
            
        with open(os.path.join(args.out_path, f'args_ablations_one_step{mod}{"_mini" if args.mini else ""}.json'), 'w') as f:
            json.dump(vars(args), f, indent=4)
            
        exps = load_experiments_json(args.experiments_json)
        out = {}
        for i, (name, files) in enumerate(tqdm(exps.items(), leave=False, desc=f'exps')):
            
            eval = Evaluator(name, files, args)
            eval.proccess()
            res = eval.calculate()
            
            res['DEBUG'] = args.debug
            out[name] = res
            if not args.debug:
                with open(os.path.join(args.out_path, f"eval_ablation_one_step{mod}{'_mini' if args.mini else ''}2.json"), 'w') as f:
                    json.dump(out, f, indent=4)
            del eval
            gc.collect()
            torch.cuda.empty_cache()
            
        if args.debug:
            print('DEBUG!!!!! not actual results')
            

def main_gt(args):
    with torch.no_grad():
        if args.debug:
            print('DEBUG!!!!! not actual results')
            
        with open(os.path.join(args.out_path, f'args_gt{"_mini" if args.mini else ""}.json'), 'w') as f:
            json.dump(vars(args), f, indent=4)
            
        exps = load_experiments_json(args.experiments_json)
        out = {}
        name = '1_ours_0_gt'
        files = exps['1_ours_0']
        eval = Evaluator(name, files, args, should_run_gt=True)
        eval.proccess()
        res = eval.calculate()
        
        res['DEBUG'] = args.debug
        out[name] = res
        if not args.debug:
            with open(os.path.join(args.out_path, f"eval_gt{'_mini' if args.mini else ''}.json"), 'w') as f:
                json.dump(out, f, indent=4)
        del eval
        gc.collect()
        torch.cuda.empty_cache()


def main_imos(args):
    with torch.no_grad():
        if args.debug:
            print('DEBUG!!!!! not actual results')
        with open(os.path.join(args.out_path, f'args_imos{"_mini" if args.mini else ""}.json'), 'w') as f:
            json.dump(vars(args), f, indent=4)
            
        out = {}
        name = 'imos'
        files = resolve_relative_path(args.imos_files_path)
        eval = Evaluator(name, files, args, imos=True)
        eval.proccess()
        res = eval.calculate()
        
        res['DEBUG'] = args.debug
        out[name] = res
        if not args.debug:
            with open(os.path.join(args.out_path, f"eval_imos{'_mini' if args.mini else ''}.json"), 'w') as f:
                json.dump(out, f, indent=4)
        del eval
        gc.collect()
        torch.cuda.empty_cache()
        
        if args.debug:
            print('DEBUG!!!!! not actual results')
    

def get_args():
    import argparse
    parser = argparse.ArgumentParser(description='Evaluate the HOI model with various metrics and comparisons')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode (processes only first 5 samples)')
    parser.add_argument('--mini', action='store_true', help='Enable mini mode (skips geometric evaluation for faster processing)')
    parser.add_argument('--pred2', action='store_true', help='Use predict2 method instead of predict for action recognition')

    parser.add_argument('--classifier', type=str, default='hoidini_data/evaluation/classifier_C30Datarep_DataRep_loc_6d', help='Path to the action classifier model directory (relative to project root or absolute)')
    parser.add_argument('--out_path', type=str, default=os.path.join(TMP_DIR, "eval_results"), help='Output directory path for saving evaluation results')
    parser.add_argument('--crop_method', type=str, default='opt_1', help='Method for cropping sequences (opt_1, opt_2, etc.)')
    parser.add_argument('--crop_start', action='store_true', help='Start cropping from the beginning of sequences')
    
    # Path arguments for hardcoded paths
    parser.add_argument('--action_rec_weights', type=str, default='hoidini_data/evaluation/exp_121_model_CVAE_object_nojoint_lr_0-0005_batchsize_1_000500.p', help='Path to action recognition model weights file (relative to project root or absolute)')
    parser.add_argument('--dataset_path', type=str, default=GRAB_DATA_PATH, help='Path to the GRAB dataset directory')
    parser.add_argument('--experiments_json', type=str, default='hoidini_data/evaluation/paper_results_grab/result_paths_per_experiment.json', help='Path to JSON file containing experiment results and file paths')
    parser.add_argument('--imos_files_path', type=str, default='hoidini_data/evaluation/paper_results_grab/imos_exp_31_model_Interaction_Prior_Posterior_CVAE_clip_lr_0-0005_batchsize_64_latentD_100_languagemodel_clip_usediscriminator_False', help='Path to IMoS model results directory')

    # Physical GPU selection similar to training/inference
    parser.add_argument('--device', type=int, default=None, help='Physical GPU id to use (mapped to logical CUDA:0)')

    args = parser.parse_args()
    return args



if __name__ == '__main__':
    args = get_args()
    # Map physical GPU id to CUDA:0 to keep dist_util.dev() consistent
    phys_id = args.device if args.device is not None else (
        int(os.environ.get('PHYSICAL_GPU_ID')) if os.environ.get('PHYSICAL_GPU_ID') is not None else None
    )
    if phys_id is not None and phys_id >= 0:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(phys_id)
        print(f"Using physical GPU {phys_id} (visible as CUDA:0)")
    # Ensure dist_util uses device 0 (logical)
    dist_util.setup_dist(0)
    os.makedirs(args.out_path, exist_ok=True)
    
    # main_imos(args)
    # main_gt(args)
    main_ablation(args)
    # main_ablation(args, mod=1)
    # main_ablation(args, mod=2)