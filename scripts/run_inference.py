from mmseg.apis import init_model, inference_model
from PIL import Image
import matplotlib.pyplot as plt

config_path = '../configs/segformer/segformer_mit-b0_8xb2-160k_ade20k-512x512.py'
checkpoint_path = '/workspace/mmsegmentation-pipeline/work_dirs/segformer_mit-b0_8xb2-160k_ade20k-512x512/iter_6000.pth'

# init model and load checkpoint
model = init_model(config_path, checkpoint_path)

# test a single image
img = '/workspace/mmsegmentation-pipeline/scripts/output_images/frame_00010.jpg'

result = inference_model(model, img)
result_tensor = result.pred_sem_seg.data.cpu().numpy()

# show the results
print(result_tensor.shape, result_tensor.min(), result_tensor.max())
plt.imshow(result_tensor.squeeze())
plt.savefig('plot.png')
