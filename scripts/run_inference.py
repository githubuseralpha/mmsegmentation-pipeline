from mmseg.apis import init_model, inference_model
from PIL import Image
import numpy as np

config_path = '../configs/segformer/segformer_mit-b0_8xb2-160k_ade20k-512x512.py'
checkpoint_path = '/workspace/mmsegmentation-pipeline/work_dirs/segformer_mit-b0_8xb2-160k_ade20k-512x512/iter_15000.pth'

# init model and load checkpoint
model = init_model(config_path, checkpoint_path)

# test a single image
img = '/workspace/mmsegmentation-pipeline/scripts/output_images/frame_00010.jpg'

result = inference_model(model, img)
result_np = result.pred_sem_seg.data.cpu().numpy()

# show the results
print(result_np.shape, result_np.min(), result_np.max())
mask = result_np > 0
# apply mask on original image
image = np.array(Image.open(img))
mask = mask[0]

image[mask] = image[mask] * 0.5 + np.array([255, 255, 255]) * 0.5

Image.fromarray(image).save("result.jpg")
