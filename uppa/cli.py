"""Command-line configuration for UPPA optimization and evaluation."""
import argparse
from contextlib import redirect_stdout
from datetime import datetime
import gc
import json
import os
from pathlib import Path
import sys


def _read_models(path):
    models = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(models, list) or not models:
        raise ValueError("Model config must be a nonempty JSON list.")
    for model in models:
        for key in ("name", "config", "checkpoint"):
            if not isinstance(model.get(key), str) or not model[key]:
                raise ValueError(f"Each model requires a nonempty {key}.")
        if any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in model["name"]):
            raise ValueError("Model name may contain only letters, digits, '_' and '-'.")
    if len({m["name"] for m in models}) != len(models):
        raise ValueError("Model names must be unique.")
    return models


def _run(args, models_config):
    print("="*80)
    print("Universal Adversarial Attack with PSO and Bezier Curves")
    print("="*80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # Load evaluation images.
    val_images_dir = str(args.images)
    if not os.path.exists(val_images_dir):
        print(f"Error: evaluation image directory does not exist: {val_images_dir}")
        return

    all_image_files = [os.path.join(val_images_dir, f) for f in os.listdir(val_images_dir) 
                       if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if len(all_image_files) == 0:
        print(f"Error: no image files found in {val_images_dir}")
        return
    
    print(f"Found {len(all_image_files)} images\n")

    # Accumulate attack success rates.
    total_attack_success_rate = 0.0
    processed_models = 0
    skipped_models = []

    # Process each model.
    for model_cfg in models_config:
        model_name = model_cfg['name']
        config_file = model_cfg['config']
        checkpoint_file = model_cfg['checkpoint']

        print(f"\n{'='*60}")
        print(f"Generating a PSO Bezier attack patch for model: {model_name}")
        print(f"{'='*60}")

        # Check for existing patch outputs.
        output_dir = str(args.output)
        patch_path = os.path.join(output_dir, f'{model_name}_universal_patch.png')
        params_path = os.path.join(output_dir, f'{model_name}_bezier_params.npy')
        
        if os.path.exists(patch_path) or os.path.exists(params_path):
            raise FileExistsError(f"Output files already exist; choose a new output directory: {output_dir}")

        # Initialize the detector.
        print("Loading model...")
        model = init_detector(config_file, checkpoint_file, device=args.device)
        model.eval()

        # Configure parameters.
        grid_dim = args.grid_dimension
        
        # Initialize the attack components.
        bezier_patch = BezierPatch(grid_dimension=grid_dim, fill_color=(0, 0, 0), render_scale=8)
        regularizer = GeometryRegularizer(grid_dimension=grid_dim)
        detection_fitness = DetectionFitness()
        
        # Use the particle count and iteration count from the command line.
        attack = AttackOrchestrator(
            model=model,
            image_paths=all_image_files,
            bezier_patch=bezier_patch,
            regularizer=regularizer,
            detection_fitness=detection_fitness,
            n_particles=args.particles,
            max_iterations=args.iterations,
            pso_params={'w': 0.9, 'c1': 1.6, 'c2': 1.4, 'r1': 0.5, 'r2': 0.5, 'v_max': 0.1},
            batch_size=8
        )

        if len(attack.image_cache) == 0:
            print(f"Warning: no single-person images selected for model {model_name}; skipping.")
            continue

        best_parameters = attack.optimize()

        # Save the patch.
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        patch_size = 256
        best_master_patch = bezier_patch.rasterize_master(best_parameters)
        best_patch = cv2.resize(best_master_patch, (patch_size, patch_size), interpolation=cv2.INTER_AREA)

        cv2.imwrite(patch_path, best_patch)
        print(f"\n✓ Universal patch for model {model_name} saved to: {patch_path}")
        np.save(params_path, best_parameters)
        print(f"✓ Bezier parameters saved to: {params_path}")

        # Evaluate patch performance.
        success_count, total_count = attack.evaluate(best_parameters)
        
        if total_count == 0:
            print(f"Warning: no single-person images for model {model_name}; skipping evaluation.")
            continue
        
        # Save patched images.
        attacked_dir = os.path.join(output_dir, 'attacked', model_name)
        success_dir = os.path.join(attacked_dir, 'success')
        failed_dir = os.path.join(attacked_dir, 'failed')
        os.makedirs(success_dir, exist_ok=True)
        os.makedirs(failed_dir, exist_ok=True)
        
        print(f"\nSaving patched images to {attacked_dir}/")
        for image_path, (image, target_bbox, clean_result) in tqdm(attack.image_cache.items(), desc=f"Save {model_name} patched images"):
            image_patched = attack._apply_patch(image, best_master_patch, target_bbox)
            
            # Detect objects in the patched image to determine success or failure.
            with torch.no_grad():
                result = inference_detector(attack.model, image_patched)
            
            person_bboxes = result[0]
            if isinstance(person_bboxes, list):
                person_bboxes = person_bboxes[0] if len(person_bboxes) > 0 else np.array([])
            
            # Determine attack success and extract the confidence score.
            is_success = False
            max_score = 0.0
            if len(person_bboxes) == 0:
                is_success = True
                max_score = 0.0
            else:
                if person_bboxes.ndim == 1:
                    max_score = person_bboxes[-1] if len(person_bboxes) > 4 else 0
                else:
                    max_score = np.max(person_bboxes[:, -1]) if len(person_bboxes) > 0 else 0
                
                if max_score < 0.5:
                    is_success = True
            
            # Save the image.
            try:
                base_name = os.path.splitext(os.path.basename(image_path))[0]
                ext = os.path.splitext(os.path.basename(image_path))[1]
                
                if is_success:
                    # Save successful attacks under success/ with confidence in the filename.
                    save_name = f"{base_name}_conf{max_score:.4f}{ext}"
                    attacked_save_path = os.path.join(success_dir, save_name)
                else:
                    # Save unsuccessful attacks under failed/.
                    save_name = f"{base_name}{ext}"
                    attacked_save_path = os.path.join(failed_dir, save_name)
                
                cv2.imwrite(attacked_save_path, image_patched)
            except Exception as e:
                print(f"  Warning: failed to save image - {e}")

        # Calculate the attack success rate.
        attack_success_rate = success_count / total_count
        total_attack_success_rate += attack_success_rate
        processed_models += 1
        
        print(f"\n{'='*60}")
        print(f"Evaluation results for model {model_name}:")
        print(f"  - Single-person images: {len(attack.image_cache)}")
        print(f"  - Successful attacks: {success_count}/{total_count}")
        print(f"  - Attack success rate: {attack_success_rate:.2%}")
        print(f"{'='*60}")

        # Release model resources.
        del model
        del attack
        torch.cuda.empty_cache()
        gc.collect()

    # Print the final summary.
    print(f"\n{'='*80}")
    print(f"--- Final Attack Summary ---")
    print(f"{'='*80}")
    print(f"Dataset: {val_images_dir} ({len(all_image_files)} images)")
    print(f"")
    
    if processed_models > 0:
        avg_success_rate = total_attack_success_rate / processed_models
        print(f"✓ Models processed: {processed_models}/{len(models_config)}")
        print(f"✓ Mean attack success rate on the evaluation set: {avg_success_rate:.2%}")
    else:
        print(f"✗ No new models processed")
    
    if len(skipped_models) > 0:
        print(f"⊙ Skipped models ({len(skipped_models)}): {', '.join(skipped_models)}")

    print(f"\nEnd time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")



def main(argv=None):
    parser = argparse.ArgumentParser(description="Optimize and evaluate universal infrared patches with PSO.")
    parser.add_argument("--config", type=Path, default=Path("configs/models.json"))
    parser.add_argument("--images", type=Path, default=Path("val_people/val_people"))
    parser.add_argument("--output", type=Path, default=Path("attack_results"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--model", nargs="+", help="Choose model names from the JSON config")
    parser.add_argument("--grid-dimension", type=int, default=6)
    parser.add_argument("--particles", type=int, default=50)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--check", action="store_true", help="Check local input paths without loading models")
    args = parser.parse_args(argv)
    try:
        if min(args.grid_dimension, args.particles, args.iterations) < 1:
            raise ValueError("Grid dimension, particles and iterations must be positive.")
        models = _read_models(args.config)
        if args.model:
            if set(args.model) - {m["name"] for m in models}:
                raise ValueError("Unknown model name in --model.")
            models = [m for m in models if m["name"] in args.model]
        if not args.images.is_dir():
            raise ValueError(f"Image directory does not exist: {args.images}")
        if not any(p.is_file() and p.suffix.lower() in (".png", ".jpg", ".jpeg") for p in args.images.iterdir()):
            raise ValueError(f"No PNG/JPG/JPEG images found: {args.images}")
        for model in models:
            for key in ("config", "checkpoint"):
                if not Path(model[key]).is_file():
                    raise ValueError(f"Required model file not found: {model[key]}")
        if args.check:
            print("Input paths exist. Model compatibility and inference have not been tested.")
            return
        # Load heavy dependencies only after argument parsing and file checks.
        global cv2, mmcv, torch, np, tqdm, init_detector, inference_detector
        global BezierPatch, GeometryRegularizer, DetectionFitness, AttackOrchestrator
        import cv2
        import mmcv
        import torch
        import numpy as np
        from tqdm import tqdm
        from mmdet.apis import init_detector, inference_detector
        from .geometry import BezierPatch, GeometryRegularizer
        from .fitness import DetectionFitness
        from .engine import AttackOrchestrator
        torch.backends.cudnn.benchmark = True  # Enable cuDNN algorithm selection for fixed input shapes.
        args.output.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        with (args.output / f"uppa_{timestamp}.log").open("w", encoding="utf-8") as log:
            from .utils import Tee
            with redirect_stdout(Tee(sys.stdout, log)):
                _run(args, models)
    except (OSError, ValueError, ImportError) as exc:
        parser.exit(1, f"UPPA: {exc}\n")
