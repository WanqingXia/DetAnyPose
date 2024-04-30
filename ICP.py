import open3d as o3d
import numpy as np
import cv2

def create_point_cloud_from_rgbd(rgb_image, depth_image):
    """Create a point cloud from RGB and depth images."""
    # Create Open3D RGBD image from numpy arrays
    rgb_o3d = o3d.geometry.Image(rgb_image)
    depth_o3d = o3d.geometry.Image(depth_image)
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        rgb_o3d, depth_o3d, depth_scale=10000.0, depth_trunc=3.0, convert_rgb_to_intensity=False)
    
    # Create point cloud
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd_image,
        o3d.camera.PinholeCameraIntrinsic(
            o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault))
    pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])  # Transform to align with coordinate system
    return pcd

def load_and_transform_model(model_path, initial_pose):
    """Load a textured 3D model and apply an initial pose."""
    mesh = o3d.io.read_triangle_mesh(model_path)
    mesh.compute_vertex_normals()
    pcd = mesh.sample_points_poisson_disk(5000)
    pcd.transform(initial_pose)
    return pcd

def colored_icp(source, target):
    """Refine alignment using Colored ICP."""
    voxel_size = 0.05  # Define the voxel size for downsampling
    source = source.voxel_down_sample(voxel_size)
    target = target.voxel_down_sample(voxel_size)
    source.estimate_normals()
    target.estimate_normals()

    # Colored ICP
    criteria = o3d.pipelines.registration.ICPConvergenceCriteria(relative_fitness=1e-6, relative_rmse=1e-6)
    result = o3d.pipelines.registration.registration_colored_icp(
        source, target, voxel_size, criteria=criteria)
    return result

# Load RGB and depth images
rgb_image = cv2.imread('./000100-color.png')
depth_image = cv2.imread('./000100-depth.png', -1)
label_image = cv2.imread('./000100-label.png', 0)
ground_mask = (np.array(label_image) == 15)
masked_depth_image = depth_image * (ground_mask.astype(np.uint16))

ground_mask_cv2 = ground_mask.astype(np.uint8) * 255
masked_rgb_image = cv2.bitwise_and(rgb_image, rgb_image, mask=ground_mask_cv2)


# Display or save results
cv2.imshow("Masked RGB Image", masked_rgb_image)
cv2.imshow("Masked Depth Image", masked_depth_image.astype(np.uint8))  # Convert for visualization
cv2.waitKey(0)
cv2.destroyAllWindows()

# Create point cloud from RGBD
scene_pcd = create_point_cloud_from_rgbd(masked_rgb_image, masked_depth_image)

# Load model and apply initial guess for the pose
initial_pose = [[-4.576717019081115723e-01, -8.891212344169616699e-01, -7.151477632305613952e-08, 0.000000000000000000e+00],
[3.764135241508483887e-01, -1.937573701143264771e-01, -9.059640169143676758e-01, 0.000000000000000000e+00],
[8.055118918418884277e-01, -4.146341085433959961e-01, 4.233545660972595215e-01, 1.614000000000000101e+00],
[0.000000000000000000e+00, 0.000000000000000000e+00, 0.000000000000000000e+00, 1.000000000000000000e+00]]  # Example: Identity matrix, replace with actual initial pose
model_pcd = load_and_transform_model('/media/iai-lab/wanqing/YCB_Video_Dataset/models/035_power_drill/textured.obj', initial_pose)

# Refine pose using Colored ICP
result = colored_icp(model_pcd, scene_pcd)
print("Transformation Matrix:")
print(result.transformation)

# Apply the transformation to the model and visualize
model_pcd.transform(result.transformation)
o3d.visualization.draw_geometries([model_pcd, scene_pcd])

