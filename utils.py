import numpy as np
from scipy.ndimage import gaussian_filter, distance_transform_edt
from matplotlib.path import Path
from scipy.ndimage import zoom
import cv2

# -----------------------------
# Membrane generation
# -----------------------------
def generate_membrane(N=512, R=120, roughness=20, thickness=6):
    cx, cy = N//2, N//2
    theta = np.linspace(0,2*np.pi,800)
    noise_vals = np.random.normal(0,1,len(theta))
    r_outer = R + roughness*noise_vals + thickness/2
    r_inner = R + roughness*noise_vals - thickness/2

    outer_x = cx + r_outer*np.cos(theta)
    outer_y = cy + r_outer*np.sin(theta)
    inner_x = cx + r_inner*np.cos(theta)
    inner_y = cy + r_inner*np.sin(theta)

    X,Y = np.meshgrid(np.arange(N), np.arange(N))
    coords = np.vstack((X.flatten(),Y.flatten())).T
    outer_path = Path(np.vstack((outer_x,outer_y)).T)
    inner_path = Path(np.vstack((inner_x,inner_y)).T)

    outer_mask = outer_path.contains_points(coords).reshape(N,N)
    inner_mask = inner_path.contains_points(coords).reshape(N,N)
    membrane_mask = outer_mask & (~inner_mask)

    texture = np.random.normal(0,5,(N,N))
    membrane_image = membrane_mask*(1000+texture)
    membrane_image[membrane_image<0] = 0
    membrane_image = gaussian_filter(membrane_image, 1.3)

    membrane_distance = distance_transform_edt(~membrane_mask)

    dy, dx = np.gradient(membrane_distance.astype(float))
    nx = dx / (np.sqrt(dx**2 + dy**2) + 1e-6)
    ny = dy / (np.sqrt(dx**2 + dy**2) + 1e-6)

    return membrane_image, membrane_mask, membrane_distance, nx, ny

# -----------------------------
# Astigmatic PSF
# -----------------------------
def astigmatic_psf_patch(psf, z, psf_radius=20):
    patch_size = 2*psf_radius + 1
    center = psf.shape[0]//2

    patch = psf[center-psf_radius:center+psf_radius+1,
                center-psf_radius:center+psf_radius+1].copy()

    scale_x = 1 + 0.5*z
    scale_y = 1 - 0.5*z

    patch = zoom(patch, (scale_y, scale_x), order=1)
    patch = cv2.resize(patch, (patch_size, patch_size))
    patch /= patch.sum()
    return patch
