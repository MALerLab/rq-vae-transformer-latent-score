import torch
import torchvision
from pathlib import Path
from tqdm import tqdm
import PIL.Image
import yaml
from omegaconf import OmegaConf

import torchvision.transforms as transforms
import numpy as np

from rqvae.models.rqvae.rqvae import RQVAE
from rqvae.utils.config import load_config, augment_arch_defaults
from rqvae.models import create_model



if __name__ == "__main__":
  brightness_threshold = True
  model_name = "unirqvae2_f16_c1024_k4"
  config_path = list((Path("logs")/ model_name).rglob("config.yaml"))[0]
  config = OmegaConf.load(config_path)
  config = load_config(config_path)
  config.arch = augment_arch_defaults(config.arch)
  
  model, _ = create_model(config.arch)
  
  ckpt_path = list((Path("logs")/ model_name).rglob("*.pt"))[0]
  model.load_state_dict(torch.load(ckpt_path)["state_dict"])
  model.cuda().eval()
  
  torch.set_grad_enabled(False)
  
  image_path = "path/to/your/image/directory"
  save_dir = Path("path/to/save/shifted/tokens")

  image_path_list = list(Path(image_path).rglob(".png")) + list(Path(image_path).rglob(".jpg"))
  
  totensor = transforms.ToTensor()
  normalize = transforms.Normalize([0.5], [0.5])

  # Create log file path outside the loop
  from datetime import datetime
  timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
  log_path = Path(f"error_{timestamp}.log")
  
  for image_path in tqdm(image_path_list):
    save_path = save_dir / image_path.stem.with_suffix(".pt") # Edit this line for saving path.
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    if save_path.exists():
      continue
    
    print("Encoding : ", image_path)
    image = PIL.Image.open(image_path).convert("L")

    # Filter (If not needed, remove this part)
    width, height = image.size
    if height < 70 or height > 390 or height > width:
      print(f"Skipping {image_path} due to invalid dimensions: {width}x{height}")
      continue
    
    # Brightness Thresholding (RQVAE model is also trained with this brightness thresholding setting; so if you want to use the same setting, keep this part. Keep in mind that YTSV dataset includes many dark scanned score images.)
    if brightness_threshold:
      # Convert PIL image to numpy array
      img_array = np.array(image)
      
      # Get median value
      median = np.median(img_array)
      
      # Create binary mask where pixels > (median-20) are 255, others unchanged
      img_array[img_array > (median-20)] = 255
      
      # Convert back to PIL Image
      image = PIL.Image.fromarray(img_array)
  
    image = totensor(image)

    # Pad image to be divisible by 16
    # Calculate padding needed to make height and width divisible by 16
    h_padding = (16 - image.shape[-2] % 16) % 16
    w_padding = (16 - image.shape[-1] % 16) % 16
    
    # Pad right and bottom with white (1.0 since image will be normalized later) 
    # +7(4,3) for 8 pixel x_shifted tokens, +3(2,1) for 8 pixel y_shifted tokens
    # +8 for additional padding for 4x8 shifted tokens
    image = torch.nn.functional.pad(image, (4+8, 3+w_padding+8, 2+8, 1+h_padding+8), mode='constant', value=1.0)

    # Normalize image
    image = normalize(image)
    
    # pixel shifting to get 4*8 shifted tokens for a single image
    x_y_shifted_tokens = []
    for j in range(4):
      y_shifted_img = image[:, j:image.shape[-2]-3+j]
      x_shifted_imgs = []
      for i in range(8):
        x_shifted_imgs.append(y_shifted_img[...,i:y_shifted_img.shape[-1]-7+i])
      x_shifted_imgs = torch.stack(x_shifted_imgs)
      
      try:
        out = model.get_codes(x_shifted_imgs.cuda())
      except RuntimeError as e: # OOM case
        if "out of memory" in str(e):
          # Split batch in half and process separately
          batch_size = len(x_shifted_imgs)
          half = batch_size // 2
          out1 = model.get_codes(x_shifted_imgs[:half].cuda())
          out2 = model.get_codes(x_shifted_imgs[half:].cuda())
          out = torch.cat([out1, out2], dim=0)
        else:
          # Log error with timestamp
          with open(log_path, "a") as f:
            f.write(f"Error processing {image_path}:\n{str(e)}\n")
          
      x_y_shifted_tokens.append(out.squeeze(0))
    x_y_shifted_tokens = torch.stack(x_y_shifted_tokens)
    x_y_shifted_tokens = x_y_shifted_tokens.transpose(0, 1) # x_y_shifted_tokens: [x_shift, y_shift, ...]
    torch.save(x_y_shifted_tokens.to(torch.int16).cpu(), str(save_path))