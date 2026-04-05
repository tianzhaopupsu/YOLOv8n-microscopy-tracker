import os
import numpy as np
import cv2
from utils import generate_membrane, astigmatic_psf_patch

# -----------------------------
# Config
# -----------------------------
PROJECT_ROOT = "./yolo_particle"
DATA_ROOT = f"{PROJECT_ROOT}/dataset"
os.makedirs(DATA_ROOT + "/images/train", exist_ok=True)
os.makedirs(DATA_ROOT + "/images/val", exist_ok=True)
os.makedirs(DATA_ROOT + "/labels/train", exist_ok=True)
os.makedirs(DATA_ROOT + "/labels/val", exist_ok=True)

N = 512
num_particles = 10
psf_radius = 20
photons = 5000
background = 10
box_size = 14
threshold = 30
num_images = 10  # small example for GitHub
z_range = 1.0
sigma_z = 0.1

# -----------------------------
# Precompute PSF
# -----------------------------
pixel_size = 100e-9
wavelength = 650e-9
NA = 1.4
L = N*pixel_size
df = 1/L
fx = np.arange(-N/2, N/2)*df
FX, FY = np.meshgrid(fx, fx)
rho = np.sqrt(FX**2 + FY**2)
fc = NA / wavelength
pupil = rho <= fc
psf = np.abs(np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(pupil))))**2
psf /= psf.sum()

# -----------------------------
# Membrane
# -----------------------------
membrane_image, membrane_mask, membrane_distance, nx, ny = generate_membrane(N)

# -----------------------------
# Particles
# -----------------------------
x = np.random.uniform(N*0.1, N*0.7, num_particles)
y = np.random.uniform(N*0.1, N*0.7, num_particles)
z = np.random.uniform(-0.5,0.5,num_particles)

for img_id in range(num_images):
    frame = membrane_image.copy()
    labels = []
    for p in range(num_particles):
        xi = int(x[p])
        yi = int(y[p])
        frame_patch = astigmatic_psf_patch(psf, z[p], psf_radius)
        x0, x1 = xi-psf_radius, xi+psf_radius+1
        y0, y1 = yi-psf_radius, yi+psf_radius+1
        if x0>=0 and y0>=0 and x1<=N and y1<=N:
            frame[y0:y1, x0:x1] += frame_patch*photons

        # YOLO label
        dist = membrane_distance[yi, xi]
        class_id = 1 if dist < threshold else 0
        xc = xi/N
        yc = yi/N
        w = box_size/N
        h = box_size/N
        labels.append(f"{class_id} {xc} {yc} {w} {h}")

    frame = np.random.poisson(frame + background)
    frame = (frame/frame.max()*255).astype(np.uint8)

    if img_id%10==0:
        img_path = f"{DATA_ROOT}/images/val/{img_id:05d}.png"
        lab_path = f"{DATA_ROOT}/labels/val/{img_id:05d}.txt"
    else:
        img_path = f"{DATA_ROOT}/images/train/{img_id:05d}.png"
        lab_path = f"{DATA_ROOT}/labels/train/{img_id:05d}.txt"
    cv2.imwrite(img_path, frame)
    with open(lab_path,"w") as f:
        for l in labels:
            f.write(l+"\n")
    print(f"Generated {img_id+1}/{num_images} images")
