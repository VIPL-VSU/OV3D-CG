import os
import time
import logging
from pathlib import Path
from typing import List, Tuple, Optional, Union

import torch
import clip
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from evaluation.scannet_constants import CLASS_LABELS_200

import json
import argparse

# set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(), logging.FileHandler('labeling_process.log')]
)


class OpenVocabLabeler:
    def __init__(self, api_key: str, label_list: List[str], device: str = None):

        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.labels = label_list

        # 1. initialize CLIP
        logging.info(f"Loading CLIP model on {self.device}...")
        self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=self.device)
        self.clip_model.eval()

        logging.info("Encoding label list with CLIP...")
        self.label_features = self._precompute_label_features(label_list)

        # initialize Gemini
        logging.info("Initializing Gemini Client...")
        genai.configure(api_key=api_key)
        self.gemini_model = genai.GenerativeModel('gemini-2.0-flash')

        self.safety_settings = {
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
        }

    def _precompute_label_features(self, labels: List[str]) -> torch.Tensor:
        text_tokens = clip.tokenize(labels).to(self.device)
        with torch.no_grad():
            text_features = self.clip_model.encode_text(text_tokens)
            text_features /= text_features.norm(dim=-1, keepdim=True)
        return text_features

    def _get_visual_prompts(self, mask_dir: Path, mask_id: int, method: str) -> Optional[List[str]]:
        """image path construction based on method"""
        mask_dir = Path(mask_dir)
        paths = []

        if method == 'landmark':
            paths = [mask_dir / f'{mask_id}_rgb.png', mask_dir / f'{mask_id}_mask.png']
        elif method == 'bbox':
            paths = [mask_dir / f'{mask_id}_bbox.png']
        elif method == 'crop':
            paths = [mask_dir / f'{mask_id}_crop.png']
        elif method == 'sam_mask':
            paths = [mask_dir / f'{mask_id}_rgb.png', mask_dir / f'{mask_id}_sam_mask.png']
        else:
            raise ValueError(f"Unknown method: {method}")

        if not all(p.exists() for p in paths):
            return None
        return [str(p) for p in paths]

    def _construct_text_prompt(self, method: str, stage: str, caption: str = "") -> str:
        """text prompt construction based on method and stage"""
        if stage == 'stage1_caption':
            if method == 'circle':
                return "There are two images given. One is the original image and another is original image with some green spots. Describe the object which these green masks mainly belongs to."
            elif method == 'bbox':
                return "Describe only the object inside the red bounding box. Consider the surrounding background context as well. Limit the description to 30 words."
            elif method == 'crop':
                return "Describe the object in the image. Limit the description to 30 words."
            elif method == 'sam_mask':
                return "There are two images given. The first one is the original image. The second one is the original image with a green mask. Describe the object that the green mask mainly covers. Limit the description to 20 words."

        elif stage == 'stage2_classify':
            label_str = ", ".join(self.labels)
            return (
                f"According to the description '{caption}'. "
                f"Identify what is the object from the 200 available labels. "
                f"Here is the list of possible labels: {label_str}. "
                "Answer must be only one label from the list and the answer must be one word or phrase."
            )
        return ""

    def query_gemini_chain(self, image_paths: List[str], method: str) -> Tuple[str, str]:
        """gemini prediction chain: stage 1 captioning + stage 2 classification"""
        images = [Image.open(p) for p in image_paths]

        try:
            # Stage 1: Generate Description
            prompt1 = self._construct_text_prompt(method, 'stage1_caption')
            response1 = self.gemini_model.generate_content(
                [prompt1, *images],
                safety_settings=self.safety_settings
            )
            caption = response1.text.strip()

            # Stage 2: Classification based on Description
            prompt2 = self._construct_text_prompt(method, 'stage2_classify', caption=caption)
            response2 = self.gemini_model.generate_content(
                prompt2,
                safety_settings=self.safety_settings
            )
            raw_label = response2.text.strip()

            return caption, raw_label

        except Exception as e:
            logging.error(f"Gemini API Error: {e}")
            raise e

    def match_label_with_clip(self, raw_text: str) -> str:
        if not raw_text:
            return "wall"

        text_token = clip.tokenize([raw_text]).to(self.device)
        with torch.no_grad():
            text_feature = self.clip_model.encode_text(text_token)
            text_feature /= text_feature.norm(dim=-1, keepdim=True)
            similarities = torch.nn.functional.cosine_similarity(text_feature, self.label_features)
            best_idx = similarities.argmax().item()

        return self.labels[best_idx]

    def process_scene(self, scene_id: str, mask_dir_root: str, mask_file_path: str, method: str):
        """process a single scene and save results"""
        mask_dir = Path(mask_dir_root) / scene_id

        if not os.path.exists(mask_file_path):
            logging.warning(f"Mask file not found: {mask_file_path}")
            return

        try:
            mask_3d = torch.load(mask_file_path, map_location='cpu')
            mask_num = mask_3d.shape[1]
            del mask_3d
        except Exception as e:
            logging.error(f"Failed to load mask file {mask_file_path}: {e}")
            return

        scene_labels = []
        scene_captions = []

        logging.info(f"Processing Scene {scene_id} ({mask_num} masks) with method '{method}'")

        for mask_id in tqdm(range(mask_num), desc=f"Scene {scene_id}", leave=False):
            image_paths = self._get_visual_prompts(mask_dir, mask_id, method)

            if not image_paths:
                scene_labels.append("wall")
                scene_captions.append("image_missing")
                continue

            max_retries = 3
            success = False
            final_label = "wall"
            final_caption = "wall"

            for attempt in range(max_retries):
                try:
                    caption, raw_res = self.query_gemini_chain(image_paths, method)

                    if raw_res:
                        # CLIP feature
                        final_label = self.match_label_with_clip(raw_res)
                        final_caption = caption
                        success = True
                        break
                except Exception:
                    time.sleep(2 * (attempt + 1))

            if not success:
                logging.warning(f"Failed to process mask {mask_id} after retries.")

            scene_labels.append(final_label)
            scene_captions.append(final_caption)

        # save results
        output_label_path = mask_dir / f"{method}_cot.txt"
        output_caption_path = mask_dir / f"{method}_cot_caption.txt"

        with open(output_label_path, "w") as f:
            f.write("\n".join(scene_labels))

        with open(output_caption_path, "w") as f:
            f.write("\n".join(scene_captions))

        logging.info(f"Saved results to {output_label_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default="config.json", help='Path to shared JSON config')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = json.load(f)

    paths = config['paths']
    params = config['parameters']
    api_settings = config['api_settings']

    API_KEY = api_settings['gemini_api_key']
    METHOD = params['prompt_method']

    MASK_DIR_ROOT = Path(paths['mask_2d_save_dir'])
    MASK_DIR_ROOT = os.path.abspath(MASK_DIR_ROOT)
    SCANNET_VAL_TXT = Path(paths['scene_list_file'])
    MASK_3D_OUTPUT_DIR = Path(paths['mask_3d_dir'])

    labeler = OpenVocabLabeler(api_key=API_KEY, label_list=CLASS_LABELS_200)

    # read scene ids
    with open(SCANNET_VAL_TXT, 'r') as f:
        scene_ids = [line.strip() for line in f.readlines()]


    for scene_id in tqdm(scene_ids, desc="Total Progress"):

        mask_file_path = MASK_3D_OUTPUT_DIR / f"{scene_id}_masks.pt"

        labeler.process_scene(
            scene_id=scene_id,
            mask_dir_root=str(MASK_DIR_ROOT),
            mask_file_path=str(mask_file_path),
            method=METHOD
        )