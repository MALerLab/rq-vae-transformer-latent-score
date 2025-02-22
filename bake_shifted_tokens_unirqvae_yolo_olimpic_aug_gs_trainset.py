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
import json

from rqvae.models.rqvae.rqvae import RQVAE
from rqvae.utils.config import load_config, augment_arch_defaults
from rqvae.models import create_model

def get_random_augmentation():
  augmentations = []
  is_applied = image_utils.is_applied
  
  # # horizontal shift, (-8, 8) pixels
  # if is_applied(0.5):
  #   augmentations.append(
  #     transforms.Lambda( lambda x: image_utils.random_horizontal_shift(x, 8) )
  #   )
  
  # rotation, [-1, 1] degrees
  if is_applied(0.5):
    augmentations.append(
      transforms.Lambda( lambda x: image_utils.random_rotate(x, 1.) )
    )
  
  # # vertical shift, [-4, 4] pixels
  # if is_applied(0.5):
  #   augmentations.append(
  #     transforms.Lambda( lambda x: image_utils.random_vertical_shift(x, 4) )
  #   )
  
  # erosion / dilatation
  if is_applied(0.5):
    augmentations.append(
      transforms.Lambda( lambda x: image_utils.random_erosion_dilatation(x) )
    )
  
  # neihborhood pixel negation (3*3 window)
  if is_applied(0.5):
    augmentations.append(
      transforms.Lambda( lambda x: image_utils.random_local_pixel_negation(x, 0.2) )
    )
  
  # global pixel negation
  if is_applied(0.5):
    augmentations.append(
      transforms.Lambda( lambda x: image_utils.random_global_pixel_negation(x, 0.01) )
    )

  # adjust contrast, by factor [-1, 1]
  if is_applied(0.5):
    augmentations.append(
      transforms.Lambda( lambda x: image_utils.random_adjust_contrast(x, -1., 1.) )
    )
  
  # adjust brightness, by delta [-0.5, 0.2]
  if is_applied(0.5):
    augmentations.append( 
      transforms.Lambda(lambda x: image_utils.random_adjust_brightness(x, -0.5, 0.2)) 
    )
  
  return transforms.Compose(augmentations)

def augment_image(img):
  img = get_random_augmentation()(img)
  return img

if __name__ == "__main__":
  torch.manual_seed(42)

  brightness_threshold = False

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
  
  image_path_list = list(Path("/home/sake/userdata/olimpic_dataset_yolo/grandstaff-lmx").rglob("*.jpg"))

  print("Loading dataset...")
  print("Length of image path list: ", len(image_path_list))

  with open("/home/sake/userdata/sake/latent-score-amt/dataset_pair_paths/grandstaff-lmx.json", "r") as f:
    dataset = json.load(f)

  # Get list of paths from train set, excluding distorted ones
  train_paths = []
  for item in dataset["train"]:
    if "distorted" not in item["pt"]:
      # Get path up to sonata folder (e.g. "beethoven/piano-sonatas/sonata01-2")
      base_path = "/".join(item["pt"].split("/")[:3])
      # Get filename without extension
      filename = item["pt"].split("/")[-1].replace(".pt", "")
      train_paths.append((base_path, filename))

  # Filter image paths that match with train set paths
  filtered_image_paths = []
  for image_path in image_path_list:
    image_path = Path(image_path)
    # Get corresponding parts from image path
    img_base = "/".join(image_path.parts[-4:-1])  # beethoven/piano-sonatas/sonata01-2
    img_filename = image_path.stem  # maj2_down_m-0-5_yolo_resized
    
    if (img_base, img_filename) in train_paths:
      filtered_image_paths.append(image_path)

  image_path_list = filtered_image_paths

  print("Filtered image paths: ", len(image_path_list))
  
  totensor = transforms.ToTensor()
  normalize = transforms.Normalize([0.5], [0.5])

  # Create log file path outside the loop
  from datetime import datetime
  timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
  log_path = Path(f"error_{timestamp}.log")
  
  for image_path in tqdm(image_path_list):
    save_path = (image_path.parent / "image_tokens" / (model_name) / "yolo_shifted_augmented" / image_path.stem).with_suffix(".pt")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    print("Encoding : ", image_path)
    image = PIL.Image.open(image_path).convert("L")

    # Filter
    width, height = image.size
    # if height < 70 or height > 390 or height > width:
    #   print(f"Skipping {image_path} due to invalid dimensions: {width}x{height}")
    #   continue
    
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

    # 7 times of different augmentations. 0 is no augmentation.
    augmented_imgs = []
    for a in range(6):
      # pixel shifting to get 4*8 shifted tokens for a single image
      x_y_shifted_tokens = []
      for j in range(4):
        y_shifted_img = image[:, j:image.shape[-2]-3+j]
        x_shifted_imgs = []
        for i in range(8):
          img = y_shifted_img[...,i:y_shifted_img.shape[-1]-7+i]
          if a > 0:
            img = augment_image(img.squeeze(0)).unsqueeze(0)
          img = normalize(img)
          x_shifted_imgs.append(img)
        x_shifted_imgs = torch.stack(x_shifted_imgs)
        
        try:
          out = model.get_codes(x_shifted_imgs.cuda())
        except Exception as e:
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
            raise e
            
        x_y_shifted_tokens.append(out.squeeze(0))
      x_y_shifted_tokens = torch.stack(x_y_shifted_tokens)
      x_y_shifted_tokens = x_y_shifted_tokens.transpose(0, 1) # x_y_shifted_tokens: [x_shift, y_shift, ...]
      augmented_imgs.append(x_y_shifted_tokens)
    augmented_imgs = torch.stack(augmented_imgs)
    torch.save(augmented_imgs.to(torch.int16).cpu(), str(save_path))