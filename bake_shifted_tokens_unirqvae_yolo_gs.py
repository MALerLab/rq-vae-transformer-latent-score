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
  brightness_threshold = False

  model_name = "unirqvae_f16_c1024_k4"
  config_path = list((Path("logs")/ model_name).rglob("config.yaml"))[0]
  config = OmegaConf.load(config_path)
  config = load_config(config_path)
  config.arch = augment_arch_defaults(config.arch)
  
  model, _ = create_model(config.arch)
  
  ckpt_path = list((Path("logs")/ model_name).rglob("*.pt"))[0]
  model.load_state_dict(torch.load(ckpt_path)["state_dict"])
  model.cuda().eval()
  
  torch.set_grad_enabled(False)
  
  image_path_list = list(Path("/home/sake/userdata/olimpic_dataset_yolo/").rglob("*.jpg"))
  # image_path_list = list(Path("/home/sake/userdata/latent_score_dataset_yolo_resize/").rglob("*/*/*/images/crop_resized/*.png"))
  # filtered_pathlist = []
  # for p in image_path_list:
  #   # if p.parents[4].stem in ["0-3", "0-4", "0-6", "1-0", "1-2", "1-3", "2-2", "4-1", "4-2", "8-0", "8-2"]:
  #   if p.parents[4].stem in ["2-0", "3-0", "3-2", "4-0", "5-0"]:
  #     filtered_pathlist.append(p)
  # image_path_list = filtered_pathlist
  
  totensor = transforms.ToTensor()
  normalize = transforms.Normalize([0.5], [0.5])

  # Create log file path outside the loop
  from datetime import datetime
  timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
  log_path = Path(f"error_{timestamp}.log")
  
  for image_path in tqdm(image_path_list):
    save_path = (image_path.parent / "image_tokens" / (model_name) / "yolo_shifted" / image_path.stem).with_suffix(".pt")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    if save_path.exists():
      print(f"Skipping {image_path} because {save_path} already exists")
      continue
    
    print("Encoding : ", image_path)
    image = PIL.Image.open(image_path).convert("L")

    # Filter
    width, height = image.size
    # if height < 70 or height > 390 or height > width:
    #   print(f"Skipping {image_path} due to invalid dimensions: {width}x{height}")
    #   continue
    if width < 30:
      continue
    
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
    image = normalize(image)
    
    # pixel shifting to get 4*8 shifted tokens for a single image
    x_y_shifted_tokens = []
    for j in range(4):
      y_shifted_img = torch.nn.functional.pad(image[:, j:image.shape[-2]-4+j], (0, 0, 4-j, j), mode='replicate')
      x_shifted_imgs = []
      for i in range(8):
        x_shifted_imgs.append(y_shifted_img[...,i:y_shifted_img.shape[-1]-7+i])
      x_shifted_imgs = torch.stack(x_shifted_imgs)
      
      try:
        out = model.get_codes(x_shifted_imgs.cuda())
      except RuntimeError as e:
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
            continue
          
      x_y_shifted_tokens.append(out.squeeze(0))
    x_y_shifted_tokens = torch.stack(x_y_shifted_tokens)
    x_y_shifted_tokens = x_y_shifted_tokens.transpose(0, 1) # x_y_shifted_tokens: [x_shift, y_shift, ...]
    torch.save(x_y_shifted_tokens.to(torch.int16).cpu(), str(save_path))