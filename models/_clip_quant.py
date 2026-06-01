"""INT8-quantize the clipped SavedModel (same settings as convert_npu_int8).
Output: npu_int8_clip.tflite (2-output split-style, with cls-logit clip)."""
import glob, numpy as np, cv2, tensorflow as tf
SM="/Users/gentiqxpc1/Documents/Garin/code/weights/npu_int8_clip_saved_model"
OUT="/Users/gentiqxpc1/Documents/Garin/code/weights/npu_int8_clip.tflite"
imgs=sorted(glob.glob("/Users/gentiqxpc1/Documents/Garin/code/calibration/train/images/*.jpg"))
print(f"calibration images: {len(imgs)}")

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
conv.inference_input_type=tf.uint8
conv.inference_output_type=tf.float32
conv.allow_custom_ops=False
conv.experimental_new_converter=True
tfl=conv.convert()
open(OUT,'wb').write(tfl)
print(f"saved {OUT} ({len(tfl)/1024/1024:.1f} MB)")

# inspect cls-logit quant scale (the tensor that was coarse before)
it=tf.lite.Interpreter(model_path=OUT); it.allocate_tensors()
for d in it.get_tensor_details():
    q=d['quantization']; sh=tuple(d['shape'])
    if q[0] not in (0.0,) and sh==(1,3,8400):
        print(f"  cls-branch '{d['name'][:40]:40s}' scale={q[0]:.6f} zp={q[1]} max_repr~{q[0]*(127-q[1]):.4f}")
