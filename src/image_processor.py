from itertools import count
import os
import glob
import cv2
import numpy as np
from PIL import Image, ImageFilter
from tqdm import tqdm
import unicodedata
import re

class ImageProcessor:
    def __init__(self, input_folder, output_base, dims):
        self.input_folder = input_folder
        self.output_base = output_base
        self.dims = dims
    
    @staticmethod
    def sanitize_filename(filename):
        # 1. Normalize unicode characters (e.g., 'è' becomes 'e' + '`')
        # 2. Encode to ASCII and ignore errors (drops the '`')
        # 3. Decode back to string
        clean_name = unicodedata.normalize('NFKD', filename).encode('ascii', 'ignore').decode('ascii')
        
        # 4. Remove any other non-alphanumeric characters (except dots and underscores)
        clean_name = re.sub(r'[^a-zA-Z0-9_.-]', '', clean_name)
        return clean_name

    @staticmethod
    def histeq(a):
        NPTS = 256
        n = 64
        a = np.array(a)
        hgram = np.ones(n) * (a.size / n)
        n = NPTS
        hgram = hgram * (a.size / np.sum(hgram))
        m = len(hgram)
        nn, cum = ImageProcessor.computeCumulativeHistogram(a, n)
        T = ImageProcessor.createTransformationToIntensityImage(a, hgram, m, n, nn, cum)
        a_norm = a.astype(np.float64)
        a_norm = np.round(a_norm * (len(T) - 1) / float(np.max(a)))
        a_norm[a_norm < 0] = 0
        a_norm[a_norm >= len(T)] = len(T) - 1
        a_norm = a_norm.astype(np.int32)
        mapped = T[a_norm]
        b = np.clip(np.round(mapped * float(np.max(a))), 0, 255).astype(a.dtype)
        out = Image.fromarray(b)
        return out

    @staticmethod
    def computeCumulativeHistogram(img, nbins):
        bins = np.arange(257) - 0.5
        nn, _ = np.histogram(img.ravel(), bins=bins)
        nn = nn.astype(np.float64)
        cum = np.cumsum(nn)
        return nn, cum

    @staticmethod
    def createTransformationToIntensityImage(a, hgram, m, n, nn, cum):
        cumd = np.cumsum(hgram)
        tol = np.zeros((m, n))
        nn1 = np.concatenate(([0], nn[:-1]))
        nn2 = np.concatenate((nn[1:], [0]))
        tol_min = np.minimum(nn1, nn2) / 2
        tol = np.tile(tol_min, (m, 1))
        err = (cumd[:, np.newaxis] - cum[np.newaxis, :]) + tol
        d = np.where(err < -a.size * np.sqrt(np.finfo(np.float64).eps))
        if d[0].size > 0:
            err[d] = a.size
        T = np.argmin(err, axis=0)
        T = (T) / (m - 1)
        return T

    @staticmethod
    def rotate_image_pillow_opencv(pil_img, angle_deg):
        cv_img = np.array(pil_img)
        h, w = cv_img.shape[:2]
        center = (w / 2, h / 2)
        M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
        rotated_cv = cv2.warpAffine(cv_img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
        rotated_pil = Image.fromarray(rotated_cv)
        return rotated_pil
    
    def MEDIAN_FILTER(self, _image, size=3):

        image_eq_np = np.array(_image)
        image_eqmed_np = np.zeros_like(image_eq_np)
        for i in range(3):
            channel = Image.fromarray(image_eq_np[:, :, i])
            filtered_channel = channel.filter(ImageFilter.MedianFilter(size=3))
            image_eqmed_np[:, :, i] = np.array(filtered_channel)
        return image_eqmed_np

    def process(self, n=16):
        print(n)
        image_paths = glob.glob(os.path.join(self.input_folder, '*.jpg'))
        # You might want to support other formats like png too:
        image_paths += glob.glob(os.path.join(self.input_folder, '*.png'))
        
        print(f"Processing images in: {self.input_folder}")
        for new_dim in self.dims:
            output_path = os.path.join(self.output_base, f'Dim{new_dim}')
            os.makedirs(output_path, exist_ok=True)
            for img in tqdm(image_paths, desc=f"Generating ({new_dim}x{new_dim}) images"):
                #try:
                image = Image.open(img)
                raw_file_name = os.path.basename(img)
                file_name = self.sanitize_filename(raw_file_name)

                resized_image = image.resize((new_dim,new_dim), Image.BICUBIC)
                variations = {}
                variations['original'] = resized_image    # 1
                variations['LR'] = image_LR = resized_image.transpose(Image.FLIP_LEFT_RIGHT)
                variations['UD'] = image_UD = resized_image.transpose(Image.FLIP_TOP_BOTTOM)
                variations['90'] = image_90 = resized_image.rotate(90, expand=True)
                variations['180'] = image_180 = resized_image.rotate(180, expand=True)
                variations['270'] = image_270 = resized_image.rotate(270, expand=True) 
                variations['original_rot_5'] = self.rotate_image_pillow_opencv(resized_image, 5)
                variations['original_rot_-5'] = self.rotate_image_pillow_opencv(resized_image, -5)

                resized_np = np.array(resized_image)
                if new_dim==32:
                    cropped = resized_np[6:32, 0:26, :]
                elif new_dim==48:
                    cropped = resized_np[2:48, 0:36, :]
                elif new_dim==64:
                    cropped = resized_np[18:64, 0:46, :]
                elif new_dim==96:
                    cropped = resized_np[34:96, 0:66, :]
                else:
                    cropped = resized_np# Fallback
                cropped_img = Image.fromarray(cropped)
                variations['original_cropzoom'] = resized_image_crop = cropped_img.resize((new_dim, new_dim),
                                                                                            resample=Image.BICUBIC)

                variations["original_med"] = resized_image_median = Image.fromarray(self.MEDIAN_FILTER(resized_image))
                variations["LR_med"] = image_LR_median = Image.fromarray(self.MEDIAN_FILTER(image_LR))
                variations["UD_med"] = image_UD_median = Image.fromarray(self.MEDIAN_FILTER(image_UD))
                variations["90_med"] = image_90_median = Image.fromarray(self.MEDIAN_FILTER(image_90))
                variations["180_med"] = image_180_median = Image.fromarray(self.MEDIAN_FILTER(image_180))
                variations["270_med"] = image_270_median = Image.fromarray(self.MEDIAN_FILTER(image_270))
                variations["original_rot5_med"] = rotated_5_median = Image.fromarray(self.MEDIAN_FILTER(variations[f'original_rot_5']))
                variations["original_rot-5_med"] = rotated_neg5_median = Image.fromarray(self.MEDIAN_FILTER(variations[f'original_rot_-5']))
                variations["original_cropzoom_med"] = cropped_median = Image.fromarray(self.MEDIAN_FILTER(resized_image_crop))


                resized_image_eq = self.histeq(resized_image)
                variations['original_eq'] = resized_image_eq
                variations['LR_eq'] = image_LR_eq = self.histeq(image_LR)
                variations['UD_eq'] = image_UD_eq = self.histeq(image_UD)
                variations['90_eq'] = image_90_eq = self.histeq(image_90)
                variations['180_eq'] = image_180_eq = self.histeq(image_180)
                variations['270_eq'] = image_270_eq = self.histeq(image_270)
                variations['original_eqcrop'] = resized_image_eqcrop = self.histeq(resized_image_crop)
                variations['original_eqrot_5'] = resized_image_eqrot_5 = self.histeq(variations['original_rot_5'])
                variations['original_eqrot_-5'] = resized_image_eqrot_neg5 = self.histeq(variations['original_rot_-5'])

                variations['original_eqmed'] = resized_image_eqmed = Image.fromarray(self.MEDIAN_FILTER(resized_image_eq))
                variations['LR_eqmed'] = image_LR_eqmed = Image.fromarray(self.MEDIAN_FILTER(image_LR_eq))
                variations['UD_eqmed'] = image_UD_eqmed = Image.fromarray(self.MEDIAN_FILTER(image_UD_eq))
                variations['90_eqmed'] = image_90_eqmed = Image.fromarray(self.MEDIAN_FILTER(image_90_eq))
                variations['180_eqmed'] = image_180_eqmed = Image.fromarray(self.MEDIAN_FILTER(image_180_eq))
                variations['270_eqmed'] = image_270_eqmed = Image.fromarray(self.MEDIAN_FILTER(image_270_eq))
                variations['original_eqrot5_med'] = rotated_5_eqmed = Image.fromarray(self.MEDIAN_FILTER(resized_image_eqrot_5))
                variations['original_eqrot-5_med'] = rotated_neg5_eqmed = Image.fromarray(self.MEDIAN_FILTER(resized_image_eqrot_neg5))
                variations['original_eqcropmed'] = cropped_eqmed = Image.fromarray(self.MEDIAN_FILTER(resized_image_eqcrop))

                # around 36 variations created
                
                save_path = os.path.join(output_path, file_name)
               
                #resized_image_eqrot.save(save_path+f'.{new_dim}_heqrot{deg}.bmp', format="BMP")
                # --- ADDED LINES: Sharpen and Blur non-equalized variations ---
                if n<=72:
                    
                    count = 0
                    
                    # save 72 variations ((36 images without blur/sharpen) x 2 formats)
                    for key, var_img in variations.items():
                        var_img.save(save_path+f'.{new_dim}_{key}.jpeg', format="JPEG", quality=60)
                        var_img.save(save_path+f'.{new_dim}_{key}.bmp', format="BMP")
                        count += 2  #2,4
                        if count >= n:
                            break
                    
                else:
                    
                    count = 0
                # We define the keys we want to process (the geometric base ones)
                    base_keys = ['original', 'LR', 'UD', '90', '180', '270', 
                                'original_rot_5', 'original_rot_-5', 'original_cropzoom',
                                'original_med', 'LR_med', 'UD_med', '90_med', '180_med', '270_med',
                                'original_rot5_med', 'original_rot-5_med', 'original_cropzoom_med']
                    
                    # Generate 18x2 variations with Blur and Sharpen for the base keys
                    for key in base_keys:
                        if key in variations:
                            # Add Blur
                            variations[f'{key}_blur'] = variations[key].filter(ImageFilter.BLUR)
                            # Add Sharpen
                            variations[f'{key}_sharp'] = variations[key].filter(ImageFilter.SHARPEN)
                    
                    # save 72 variations ((18 blur + 18 sharp) x 2 formats)
                    for key, var_img in variations.items():
                        var_img.save(save_path+f'.{new_dim}_{key}.jpeg', format="JPEG", quality=60)
                        var_img.save(save_path+f'.{new_dim}_{key}.bmp', format="BMP")
                        count += 2
                        if count >= n:
                            break
                
                #except Exception as e:
                   # print(f"Error processing {img}: {e}")
        print("Generating process finished")
