"""
Generates synthetic microscopy videos of diffusing particles near membranes.
"""

import numpy as np
import imageio
from noise import pnoise1
from matplotlib.path import Path
from scipy.ndimage import gaussian_filter, distance_transform_edt
import argparse

def astigmatic_widths(z, sigma0=1.3, zR=0.6, delta=0.4):
    sigma_x = sigma0 * np.sqrt(1 + ((z - delta)/zR)**2)
    sigma_y = sigma0 * np.sqrt(1 + ((z + delta)/zR)**2)
    return sigma_x, sigma_y

def astigmatic_psf_patch(z, psf_radius=20):
    sigma_x, sigma_y = astigmatic_widths(z)
    size = psf_radius*2 + 1
    x = np.arange(size) - psf_radius
    y = np.arange(size) - psf_radius
    X,Y = np.meshgrid(x,y)
    psf = np.exp(-(X**2/(2*sigma_x**2) + Y**2/(2*sigma_y**2)))
    psf /= psf.sum()
    return psf

def generate_video(
    N=512, frames=2000, num_particles=20, pixel_size=100e-9,
    photons=8000, background=10, dt=0.02, D=0.9e-12,
    membrane_thickness=6, roughness=20, psf_radius=20,
    output_file="synthetic_particles.mp4"
):
    # Initialize particle positions
    x = np.random.uniform(N*0.1, N*0.7, num_particles)
    y = np.random.uniform(N*0.1, N*0.7, num_particles)
    z = np.random.uniform(-0.5,0.5,num_particles)

    # Generate irregular membrane
    cx, cy, R = N//3, N//3, N//4
    theta = np.linspace(0,2*np.pi,800)
    noise_vals = np.array([pnoise1(i*0.05) for i in range(len(theta))])
    r_outer = R + roughness*noise_vals + membrane_thickness/2
    r_inner = R + roughness*noise_vals - membrane_thickness/2
    outer_x = cx + r_outer*np.cos(theta)
    outer_y = cy + r_outer*np.sin(theta)
    inner_x = cx + r_inner*np.cos(theta)
    inner_y = cy + r_inner*np.sin(theta)
    X,Y = np.meshgrid(np.arange(N),np.arange(N))
    coords = np.vstack((X.flatten(),Y.flatten())).T
    outer_path = Path(np.vstack((outer_x,outer_y)).T)
    inner_path = Path(np.vstack((inner_x,inner_y)).T)
    outer_mask = outer_path.contains_points(coords).reshape(N,N)
    inner_mask = inner_path.contains_points(coords).reshape(N,N)
    membrane_mask = outer_mask & (~inner_mask)
    dist_to_membrane = distance_transform_edt(~membrane_mask)

    # Membrane fluorescence
    texture = np.random.normal(0,5,(N,N))
    membrane_image = gaussian_filter(membrane_mask*(200+texture),1.3)
    nx, ny = np.gradient(membrane_mask.astype(float))

    # Particle diffusion parameters
    sigma_xy = np.sqrt(2*D*dt)/pixel_size
    sigma_z = np.sqrt(2*D*dt)*1e6

    frames_list = []
    for i in range(frames):
        frame = membrane_image.copy()
        # update particles
        for p in range(num_particles):
            dx, dy = np.random.normal(0,sigma_xy,2)
            x_new, y_new = x[p]+dx, y[p]+dy
            xi, yi = int(round(x_new)), int(round(y_new))
            if 0 <= xi < N and 0 <= yi < N:
                d = dist_to_membrane[yi, xi]
                if d < 5:  # repel buffer
                    dx, dy = -dx + 1.5*nx[yi,xi], -dy + 1.5*ny[yi,xi]
                    x_new, y_new = x[p]+dx, y[p]+dy
            x[p], y[p] = x_new, y_new
            # Z diffusion
            z[p] += np.random.normal(0,sigma_z)
            # PSF
            xi, yi = int(round(x[p])), int(round(y[p]))
            if 0 <= xi-psf_radius < N and xi+psf_radius < N and 0 <= yi-psf_radius < N and yi+psf_radius < N:
                psf_patch = astigmatic_psf_patch(z[p], psf_radius)
                frame[yi-psf_radius:yi+psf_radius+1, xi-psf_radius:xi+psf_radius+1] += psf_patch*photons
        # camera noise
        frame = np.random.poisson(frame+background)
        frames_list.append((frame/frame.max()*255).astype(np.uint8))

    # save video
    writer = imageio.get_writer(output_file,fps=20)
    for f in frames_list:
        writer.append_data(f)
    writer.close()
    print(f"Saved synthetic video to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="synthetic_particles.mp4", help="Output video file")
    args = parser.parse_args()
    generate_video(output_file=args.output)
