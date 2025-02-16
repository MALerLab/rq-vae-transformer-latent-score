## Requirements
All dependencies are managed by pipenv. please check Pipfile and Pipfile.lock

# Image Tokenization Process(UNIRQVAE2)

This script tokenizes grayscale images using a trained RQVAE model, applying various preprocessing steps and generating shifted tokens.
Refer to `tokenize_unirqvae.py` for more details.

## Prerequisites
- RQVAE model checkpoint
    - download from [here](https://drive.google.com/file/d/19Y7m4kx4ZmoCsQGNcMacnRowKXMj91mp/view?usp=drive_link)
    - and extract the checkpoint file to `logs/unirqvae2_f16_c1024_k4/`

## Process Overview

### 1. Model Loading
```python
model_name = "unirqvae2_f16_c1024_k4"
config_path = Path("logs")/model_name/"config.yaml"
ckpt_path = Path("logs")/model_name/"checkpoint.pt"  # .pt file
```

### 2. Image Loading and Filtering (If not needed, remove this part)
- Loads grayscale images and applies dimension filtering
- Skips images that are:
  - Less than 70 pixels in height
  - More than 390 pixels in height
  - Height greater than width
```python
image = PIL.Image.open(image_path).convert("L")
```

### 3. Brightness Thresholding
RQVAE model is also trained with this brightness thresholding setting; so if you want to use the same setting, keep this part. Keep in mind that YTSV dataset includes many dark scanned score images.
Applies an adaptive brightness threshold based on the median value:
```python
img_array = np.array(image)
median = np.median(img_array)
img_array[img_array > (median-20)] = 255
```

### 4. Image Normalization
Converts image to tensor and normalizes to [-1, 1] range:
```python
image = transforms.ToTensor()(image)
image = transforms.Normalize([0.5], [0.5])(image)
```

### 5. Padding
Adds padding to ensure dimensions are divisible by 16 and accommodates shifting:
- Adds padding for x-shifted tokens: +7 (4 left, 3 right)
- Adds padding for y-shifted tokens: +3 (2 top, 1 bottom)
- Additional +8 padding at the left, right, top, bottom for not dropping any image information
```python
h_padding = (16 - image.shape[-2] % 16) % 16
w_padding = (16 - image.shape[-1] % 16) % 16
image = torch.nn.functional.pad(
    image, 
    (4+8, 3+w_padding+8, 2+8, 1+h_padding+8), 
    mode='constant', 
    value=1.0
)
```

If you are not going to use the token shifting, the code below is enough.
```python
h_padding = (16 - image.shape[-2] % 16) % 16
w_padding = (16 - image.shape[-1] % 16) % 16
image = torch.nn.functional.pad(
    image, 
    (0, w_padding, 0, h_padding), 
    mode='constant', 
    value=1.0
)
```

### 6. Token Shifting
Generates multiple shifted versions of the image(augmentation since the token codes change with the shift):
- Creates 4 vertical shifts
- For each vertical shift, creates 8 horizontal shifts
- Results in 32 (4x8) shifted versions of the image
```python
# For each vertical shift (0-3)
y_shifted_img = image[:, j:image.shape[-2]-3+j]

# For each horizontal shift (0-7)
x_shifted_img = y_shifted_img[..., i:y_shifted_img.shape[-1]-7+i]
```

### 7. Tokenization
- Processes shifted images through RQVAE model
- Handles out-of-memory errors by splitting batches
- Saves tokens as int16 tensor
```python
tokens = model.get_codes(x_shifted_imgs.cuda())
torch.save(tokens.to(torch.int16).cpu(), save_path)
```

## Output
- Saves tokenized representations as .pt files
- Output shape: [x_shift, y_shift, ...]
- Tokens are saved in int16 format to save memory

## Error Handling
- Creates timestamped error log for failed tokenizations
- Implements batch splitting for large images to handle memory constraints

## Notes
- Images are processed in grayscale
- The script skips already processed images
- Errors are logged to `error_YYYYMMDD_HHMMSS.log`
