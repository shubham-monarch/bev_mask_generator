
from scripts.occ_mask_generator import OccMap


if __name__ == "__main__":  
    
    pcd = o3d.t.io.read_point_cloud("debug/9/left.ply")
    K = np.load("debug/9/K.npy")
    P = np.load("debug/9/P.npy")
    OccMap.project_pcd_to_img(pcd, K, P, img_shape=(1080, 1920), visualize=True)
