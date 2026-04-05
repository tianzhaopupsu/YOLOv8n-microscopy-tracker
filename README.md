### YOLO Particle Tracking Simulation

This repository generates synthetic microscopy images of particles near irregular membranes and trains a YOLOv8 model to detect them.

## Setup

```bash
pip install ultralytics noise numpy opencv-python matplotlib scipy
```
### YOLOv8 Particle Tracking Results

Our YOLOv8 model (73 layers, 3M parameters, 8.1 GFLOPs) was trained to detect and classify particles in synthetic microscopy images as either **near the membrane (Ne)** or **far from the membrane (Fa)**.  

**Evaluation on 98 validation images (1,470 particle instances):**  

| Class | Precision | Recall | mAP50 | mAP50-95 |
|-------|----------|--------|-------|-----------|
| Fa    | 0.993    | 0.884  | 0.925 | 0.902     |
| Ne    | 0.989    | 0.990  | 0.995 | 0.987     |
| **All** | 0.991  | 0.937  | 0.960 | 0.944     |

**Interpretation:**  
- Near-membrane particles (Ne), which are critical for tracking interactions, are detected **almost perfectly**.  
- Far particles (Fa) are detected with slightly lower recall due to edge effects and diffusion near frame borders.  
- Overall, the model achieves **high precision and recall across both classes**, demonstrating effective particle detection for real-time microscope control.

💡 **Note:** Metrics are calculated on a small validation set; performance can be further improved with larger datasets and enhanced PSF handling at image edges.
