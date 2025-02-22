import torch
import torchvision
from pathlib import Path
from tqdm import tqdm
import PIL.Image
import yaml
from omegaconf import OmegaConf

import torchvision.transforms as transforms
import numpy as np
import image_utils

from rqvae.models.rqvae.rqvae import RQVAE
from rqvae.utils.config import load_config, augment_arch_defaults
from rqvae.models import create_model


if __name__ == "__main__":
  torch.manual_seed(42)

  brightness_threshold = True

  model_name = "unirqvae3_f16_c1024_k4"
  config_path = list((Path("logs")/ model_name).rglob("config.yaml"))[0]
  config = OmegaConf.load(config_path)
  config = load_config(config_path)
  config.arch = augment_arch_defaults(config.arch)
  
  model, _ = create_model(config.arch)
  
  ckpt_path = list((Path("logs")/ model_name).rglob("*.pt"))[0]
  model.load_state_dict(torch.load(ckpt_path)["state_dict"])
  model.cuda().eval()
  
  torch.set_grad_enabled(False)
  
  image_path_list = list(Path("/home/sake/userdata/ls_beethoven_piano_sonata_dataset/Beethoven_Piano_Sonata_Dataset_v2/4_Splitted/images").rglob("*/scan/crop_yolo_resized/*.png"))
  # image_path_list = list(Path("/home/sake/userdata/olimpic_dataset/olimpic-1.0-synthetic_3/").rglob("*.png"))
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
    save_path = (image_path.parents[4] / "image_tokens" / image_path.parents[2].stem / (model_name) / "yolo" / image_path.stem).with_suffix(".pt")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    print("Encoding : ", image_path)
    image = PIL.Image.open(image_path).convert("L")

    
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
    image = torch.nn.functional.pad(image, (0, w_padding, 0, h_padding), mode='constant', value=1.0)

    # Normalize image
    image = normalize(image)

    out = model.get_codes(image.cuda().unsqueeze(0))

    
    torch.save(out.unsqueeze(0).cpu().to(torch.int16), str(save_path))
