import numpy as np
import scipy
import torch
from scipy import linalg
from tqdm import tqdm
import trimesh

def evaluate_fid(gt_joint, out_jt, num_labels, classifier, gt_labels):
    # todo haim - this is written like shit, can optimize using torch
    with torch.no_grad():
        fid = np.zeros(len(gt_joint))
        for idx in range(len(gt_joint)):
            y_pred_gt, ground_truth_activations = classifier.predict(gt_joint[idx][:,:22,:])
            statistics_1 = calculate_activation_statistics(ground_truth_activations)
            y_pred, pred_activations = classifier.predict(out_jt[idx][:,:22,:])
            statistics_2 = calculate_activation_statistics(pred_activations)
            fid[idx] = calculate_frechet_distance(statistics_1[0], statistics_1[1],
                                      statistics_2[0], statistics_2[1])
        final_fid = np.mean(fid)
        return final_fid


def evaluate_fid_our(activations_gt, activations_gen, eps=1e-6):
    activations_gt = activations_gt.cpu().numpy().squeeze()
    activations_gen = activations_gen.cpu().numpy().squeeze()

    mu1, sigma1 = np.mean(activations_gt, axis=0), np.cov(activations_gt, rowvar=False)
    mu2, sigma2 = np.mean(activations_gen, axis=0), np.cov(activations_gen, rowvar=False)
    
    
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert (mu1.shape == mu2.shape
            ), "Training and test mean vectors have different lengths"
    assert (sigma1.shape == sigma2.shape
            ), "Training and test covariances have different dimensions"

    diff = mu1 - mu2
    # Product might be almost singular
    covmean, _ = scipy.linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ("fid calculation produces singular product; "
            "adding %s to diagonal of cov estimates") % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = scipy.linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError("Imaginary component {}".format(m))
            # print("Imaginary component {}".format(m))
        covmean = covmean.real
    tr_covmean = np.trace(covmean)

    return diff.dot(diff) + np.trace(sigma1) + np.trace(
        sigma2) - 2 * tr_covmean
                
def calculate_activation_statistics(activations):
    activations = activations.cpu().numpy()
    mu = np.mean(activations, axis=0)
    sigma = np.cov(activations, rowvar=False)
    return mu, sigma

def calculate_accuracy(joints_3d_vec, num_labels, classifier, gt_labels):
    # todo haim - this is written like shit, can optimize using torch
    confusion = torch.zeros(num_labels, num_labels, dtype=torch.long)
    with torch.no_grad():
        for idx in range(len(joints_3d_vec)):
            y_pred, _ = classifier.predict(joints_3d_vec[idx][:,:22,:])
            batch_pred = y_pred.max(dim=1).indices
            batch_label = gt_labels[idx].max(dim=1).indices
            for label, pred in zip(batch_label, batch_pred):
                # print(label.data, pred.data)
                confusion[label][pred] += 1
    return np.trace(confusion.numpy())/len(joints_3d_vec)

def calculate_diversity_(activations, labels, num_labels):
    diversity_times = 200
    
    num_motions = len(labels)
    diversity = 0
    first_indices = np.random.randint(0, num_motions, diversity_times)
    second_indices = np.random.randint(0, num_motions, diversity_times)
    for first_idx, second_idx in zip(first_indices, second_indices):
        diversity += torch.dist(activations[first_idx, :],
                                activations[second_idx, :])
    diversity /= diversity_times
    return diversity

def calculate_multimodality_(activations, labels, num_labels):
    num_motions = len(labels)
    multimodality = 0
    multimodality_times = 20
    labal_quotas = np.repeat(multimodality_times, num_labels)
    count = 0
    while np.any(labal_quotas > 0) and count <= 10000:
        count+=1
        first_idx = np.random.randint(0, num_motions)
        first_label = labels[first_idx].max(dim=0).indices
        if not labal_quotas[first_label.cpu().detach().numpy()]:
            continue
        second_idx = np.random.randint(0, num_motions)
        second_label = labels[second_idx].max(dim=0).indices
        while first_label != second_label:
            second_idx = np.random.randint(0, num_motions)
            second_label = labels[second_idx].max(dim=0).indices
        labal_quotas[first_label] -= 1
        first_activation = activations[first_idx, :]
        second_activation = activations[second_idx, :]
        multimodality += torch.dist(first_activation,
                                    second_activation)
    multimodality /= (multimodality_times * num_labels)
    return multimodality

def calc_AVE(joints_sbj, joints_gt):
    T = joints_gt.shape[0]    
    J = joints_gt.shape[1]
    var_gt = torch.zeros((J))
    var_pred = torch.zeros((J))
    for j in range(J):
        var_gt[j] = torch.var(joints_gt[:, j], dim=0).mean().item()
        var_pred[j] = torch.var(joints_sbj[:, j], dim=0).mean().item()
    mean_ave_loss = mean_l2di_(var_gt, var_pred)
    return mean_ave_loss

def mean_l2di_(a,b):
    x = torch.mean(torch.sqrt(torch.sum((a - b)**2, -1)))
    return x

def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
    Stable version by Dougal J. Sutherland.
    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.
    Returns:
    --   : The Frechet Distance.
    """
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)
    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'
    diff = mu1 - mu2
    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real
    tr_covmean = np.trace(covmean)
    return (diff.dot(diff) + np.trace(sigma1) +
            np.trace(sigma2) - 2 * tr_covmean)
    
    
def get_peneration(obj_verts, obj_faces, body_verts, body_faces, names):
    volume_ratios = []
    for motion_obj_verts, motion_obj_faces, motion_body_verts, name in tqdm(list(zip(obj_verts, obj_faces, body_verts, names)), leave=False, desc='penetration'):
        motion_volume_ratios=[]
        for frame in range(motion_obj_verts.shape[0]):
            try:
                motion_volume_ratios.append(_calculate_intersection_volume_ratio(motion_obj_verts[frame], motion_obj_faces, motion_body_verts[frame], body_faces))
            except Exception as e:
                print(f"Error calculating intersection volume ratio for {name} at frame {frame}: {e}")
                raise e
        volume_ratios.append(sum(motion_volume_ratios)/len(motion_volume_ratios))
    return sum(volume_ratios)/len(volume_ratios)


# from pytorch3d.structures import Meshes, Pointclouds

def get_dist_metrics(obj_verts, obj_faces, body_verts, body_faces, table_hights, debug=False):
    mean = lambda x : 0 if len(x) == 0 else  sum(x) / len(x)
    penetration_dists=[]
    floating_dists=[]
    num_penetration_frames = 0
    num_floating_frames = 0
    n_frames = 0
    
    for motion_obj_verts, motion_obj_faces, motion_body_verts, table_hight in tqdm(list(zip(obj_verts, obj_faces, body_verts, table_hights)), leave=False, desc='dist'):
        penetration_dists_motion=[]
        floating_dists_motion=[]

        eps = 0.005 # 5mm
        n = 10 if debug else motion_obj_verts.shape[0]
        
        for frame in tqdm(list(range(n)), leave=False, desc='frame'):     
            obj_verts_np = motion_obj_verts[frame].cpu().numpy()
            body_verts_np = motion_body_verts[frame].cpu().numpy()
            
            if obj_verts_np[:,2].min() <= table_hight + 0.05: # 5cm
                # object on table
                continue 

            body_mesh = trimesh.Trimesh(vertices=body_verts_np, faces=body_faces)
            
            # the distance is negative if the point is inside the mesh, otherwise positive
            frame_dists = -1 * trimesh.proximity.signed_distance(body_mesh, obj_verts_np)
            min_dist = frame_dists.min()
            
            if min_dist < 0:
                penetration_dists_motion.append(min_dist)
                if abs(min_dist) > eps:
                    num_penetration_frames += 1
            else:
                floating_dists_motion.append(min_dist)
                if abs(min_dist) > eps: 
                    num_floating_frames += 1
            n_frames += 1
        
        penetration_dists.append(mean(penetration_dists_motion))
        floating_dists.append(mean(floating_dists_motion))
  
    ret = {'mean_penetration_dist': mean(penetration_dists),
           'mean_float_dist': mean(floating_dists),
           'std_penetration_dist': np.std(penetration_dists),
           'std_float_dist': np.std(floating_dists),
           'penetration_frame_ratio':num_penetration_frames/n_frames,
           'float_frame_ratio': num_floating_frames/n_frames
    }
    return ret



def _calculate_intersection_volume_ratio(obj_verts, obj_faces, body_verts, body_faces):
    
    # Convert tensors to numpy arrays
    obj_verts_np = obj_verts.cpu().numpy()
    body_verts_np = body_verts.cpu().numpy()

    # Create trimesh objects
    obj_mesh = trimesh.Trimesh(vertices=obj_verts_np, faces=obj_faces)
    body_mesh = trimesh.Trimesh(vertices=body_verts_np, faces=body_faces)

    # Compute the intersection
    intersection = trimesh.boolean.intersection([obj_mesh, body_mesh])

    # Return the volume of the intersection
    return (intersection.volume / obj_mesh.volume) if intersection is not None else 0.0



def topk_accuracy(outputs, labels, topk=(1,3,5)):
    """
    Compute the top-k accuracy for the given outputs and labels.

    :param outputs: Tensor of model outputs, shape [batch_size, num_classes]
    :param labels: Tensor of labels, shape [batch_size]
    :param topk: Tuple of k values for which to compute top-k accuracy
    :return: List of top-k accuracies for each k in topk
    """
    maxk = max(topk)
    
    batch_size = labels.size(0)
    outputs = outputs.squeeze()
    # Get the top maxk indices along the last dimension (num_classes)
    _, pred = outputs.topk(maxk, 1, True, True)

    pred = pred.t()

    # Check if the labels are in the top maxk predictions
    correct = pred.eq(labels.view(1, -1).expand_as(pred))

    # Compute accuracy for each k
    accuracies = {}
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        accuracies[f'rec_acc_top_{k}'] = correct_k.mul_(100.0 / batch_size).item()
    return accuracies