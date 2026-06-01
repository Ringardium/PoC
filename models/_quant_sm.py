"""Re-quantize a given SavedModel with the EXACT same pipeline (326 jpgs) — used
to isolate the clip effect from calibration confounds. Args: <saved_model> <out.tflite>"""
import sys, glob, numpy as np, cv2, tensorflow as tf
SM, OUT = sys.argv[1], sys.argv[2]
imgs=sorted(glob.glob("/Users/gentiqxpc1/Documents/Garin/code/calibration/train/images/*.jpg"))
def rep_gen():
    for p in imgs:
        im=cv2.imread(p)
        if im is None: continue
        rgb=cv2.cvtColor(cv2.resize(im,(640,640)),cv2.COLOR_BGR2RGB)
        yield [ (rgb.astype(np.float32)/255.0)[np.newaxis] ]
conv=tf.lite.TFLiteConverter.from_saved_model(SM)
conv.optimizations=[tf.lite.Optimize.DEFAULT]
conv.representative_dataset=rep_gen
conv.target_spec.supported_ops=[tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
conv.inference_input_type=tf.uint8; conv.inference_output_type=tf.float32
conv.allow_custom_ops=False; conv.experimental_new_converter=True
open(OUT,'wb').write(conv.convert())
print(f"saved {OUT} (calib={len(imgs)})")
