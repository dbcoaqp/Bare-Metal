import numpy as np
from plyfile import PlyData
import sys

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def convert_ply_to_baremetal_npy(ply_path, npy_path):
    print(f"[Forging] Loading raw PLY data from {ply_path}...")
    try:
        plydata = PlyData.read(ply_path)
        v = plydata.elements[0]
    except Exception as e:
        print(f"Fatal: Failed to load PLY. {e}")
        sys.exit(1)

    num_points = v.count
    print(f"[Forging] Extracted {num_points} Gaussian points. Processing math and pruning...")

    # 1. Position & Opacity
    x, y, z = v['x'], v['y'], v['z']

    # 2. Extract and calculate scale
    # The PLY stores a separate log-scale for each axis.
    # Older isotropic code averaged the three scales into one value. The current
    # engine keeps all three axes, so that older conversion path is left disabled.
    # scale_0, scale_1, scale_2 = v['scale_0'], v['scale_1'], v['scale_2']
    # real_scale_x = np.exp(scale_0)
    # real_scale_y = np.exp(scale_1)
    # real_scale_z = np.exp(scale_2)
    # final_scale = (real_scale_x + real_scale_y + real_scale_z) / 3.0

    # 2. Restore the three axis scales to preserve each Gaussian's ellipsoid shape.
    sx, sy, sz = np.exp(v['scale_0']), np.exp(v['scale_1']), np.exp(v['scale_2'])

    # 3. Base Color (DC component of SH)
    # SH degree-0 to RGB conversion constant: RGB = SH0 * 0.28209 + 0.5.
    SH_C0 = 0.28209479177387814
    f_dc_0, f_dc_1, f_dc_2 = v['f_dc_0'], v['f_dc_1'], v['f_dc_2']
    r = np.clip(f_dc_0 * SH_C0 + 0.5, 0.0, 1.0)
    g = np.clip(f_dc_1 * SH_C0 + 0.5, 0.0, 1.0)
    b = np.clip(f_dc_2 * SH_C0 + 0.5, 0.0, 1.0)

    # 4. Convert opacity from the PLY's logit value to the 0.0–1.0 range.
    opacity = sigmoid(v['opacity'])

    # 5. Read the rotation quaternion and normalize it to unit length.
    rot_0, rot_1, rot_2, rot_3 = v['rot_0'], v['rot_1'], v['rot_2'], v['rot_3']
    norms = np.sqrt(rot_0**2 + rot_1**2 + rot_2**2 + rot_3**2)
    # Prevent division by zero for a quaternion whose length is zero.
    norms[norms == 0] = 1e-10
    rw, rx, ry, rz = rot_0/norms, rot_1/norms, rot_2/norms, rot_3/norms

    # 6. Pack each Gaussian into the 64-byte C++ record layout with 16-byte alignment.
    # The field order must match GaussianPoint in core/GaussianData.h.
    baremetal_dtype = np.dtype([
        ('px', 'f4'), ('py', 'f4'), ('pz', 'f4'), ('opacity', 'f4'),
        ('sx', 'f4'), ('sy', 'f4'), ('sz', 'f4'), ('pad1', 'f4'),
        ('r', 'f4'), ('g', 'f4'), ('b', 'f4'), ('pad2', 'f4'),
        ('rw', 'f4'), ('rx', 'f4'), ('ry', 'f4'), ('rz', 'f4')
    ])

    final_array = np.zeros(num_points, dtype=baremetal_dtype)
    final_array['px'], final_array['py'], final_array['pz'] = x, y, z
    #final_array['scale'] = final_scale
    final_array['opacity'] = opacity
    final_array['sx'], final_array['sy'], final_array['sz'] = sx, sy, sz
    final_array['r'], final_array['g'], final_array['b'] = r, g, b
    final_array['rw'], final_array['rx'], final_array['ry'], final_array['rz'] = rw, rx, ry, rz

    print(f"[Forging] Saving highly optimized binary payload to {npy_path}...")
    np.save(npy_path, final_array)
    print("[Forging] SUCCESS. The payload is ready for Bare Metal DMA ingestion.")

if __name__ == "__main__":
    # Example: python convert_ply_to_npy.py input_model.ply test.npy
    if len(sys.argv) < 3:
        print("Usage: python convert_ply_to_npy.py <input.ply> <output.npy>")
        sys.exit(1)
    
    convert_ply_to_baremetal_npy(sys.argv[1], sys.argv[2])
