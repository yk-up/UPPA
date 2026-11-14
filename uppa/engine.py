"""Universal patch optimization and evaluation workflow."""
import cv2
import mmcv
import torch
import numpy as np
from tqdm import tqdm
from mmdet.apis import inference_detector
from .pso import PSO

class AttackOrchestrator:
    """
    Coordinate the complete attack workflow.
    Coordinate the components and run PSO optimization.
    """
    def __init__(self, model, image_paths, bezier_patch, regularizer, detection_fitness,
                 n_particles=50, max_iterations=10, pso_params=None, batch_size=8):
        """
        :param model: Target object detector
        :param image_paths: List of input image paths
        :param bezier_patch: BezierPatch instance
        :param regularizer: GeometryRegularizer instance
        :param detection_fitness: DetectionFitness instance
        :param n_particles: Number of PSO particles
        :param max_iterations: Maximum number of PSO iterations
        :param pso_params: PSO parameter dictionary (w, c1, c2, r1, r2, v_max)
        :param batch_size: Batch size
        """
        self.model = model
        self.image_paths = image_paths
        self.bezier_patch = bezier_patch
        self.regularizer = regularizer
        self.detection_fitness = detection_fitness
        self.n_particles = n_particles
        self.max_iterations = max_iterations
        self.batch_size = batch_size
        
        # PSO
        if pso_params is None:
            pso_params = {'w': 0.9, 'c1': 1.6, 'c2': 1.4, 'r1': 0.5, 'r2': 0.5, 'v_max': 0.1}
        self.pso = PSO(dim=bezier_patch.total_params, n_particles=n_particles, **pso_params)
        
        # Preprocess images and select single-person images.
        print("Precomputing clean detections and selecting single-person images...")
        self.image_cache = {}
        for image_path in tqdm(image_paths, desc="Load and select single-person images"):
            image = mmcv.imread(image_path)
            with torch.no_grad():
                clean_result = inference_detector(self.model, image)
            person_bboxes = clean_result[0]
            
            if isinstance(person_bboxes, list):
                person_bboxes = person_bboxes[0] if len(person_bboxes) > 0 else np.array([])
            
            if len(person_bboxes) > 0:
                if person_bboxes.ndim == 1:
                    scores = np.array([person_bboxes[-1]])
                else:
                    scores = person_bboxes[:, -1]
                num_persons = np.sum(scores > 0.5)
                
                if num_persons == 1:
                    target_bbox = np.atleast_2d(person_bboxes)[scores > 0.5][0, :4].astype(int)
                    self.image_cache[image_path] = (image, target_bbox, clean_result)
        
        print(f"From {len(image_paths)} input images, selected {len(self.image_cache)} single-person images")
        
        # Initialize the first particle from a regular grid.
        self.pso.positions[0] = self.bezier_patch.initialize_parameters()
        
        # Apply constraints to all particles.
        for i in range(self.n_particles):
            self.pso.positions[i] = self.regularizer.apply_constraints(self.pso.positions[i])
        
    
    def _apply_patch(self, image, patch_master, target_bbox):
        """
        Apply the patch to the target region of an image.
        
        :param image: Original image, (H, W, 3) uint8
        :param patch_master: Master patch image
        :param target_bbox: Target bounding box [x1, y1, x2, y2]
        :return: Image with the patch applied
        """
        x1, y1, x2, y2 = target_bbox
        x1, y1 = max(0, x1), max(0, y1)
        x2 = min(image.shape[1], x2)
        y2 = min(image.shape[0], y2)
        
        bbox_width = x2 - x1
        bbox_height = y2 - y1
        
        if bbox_width <= 0 or bbox_height <= 0:
            return image.copy()
        
        # Set the nominal patch size to one quarter of the bbox height.
        grid_width = max(1, int(bbox_height / 4))
        patch_size = min(grid_width, bbox_width)
        grid_height = grid_width = patch_size
        
        # Resize the patch.
        if patch_master.shape[0] == grid_height and patch_master.shape[1] == grid_width:
            patch_resized = patch_master
        else:
            interp = cv2.INTER_AREA if (grid_height < patch_master.shape[0] or grid_width < patch_master.shape[1]) else cv2.INTER_LINEAR
            patch_resized = cv2.resize(patch_master, (grid_width, grid_height), interpolation=interp)
        
        # Center the patch in the bounding box.
        image_patched = image.copy()
        x_offset = x1 + (bbox_width - grid_width) // 2
        y_offset = y1 + (bbox_height - grid_height) // 2
        
        y_end = min(y_offset + grid_height, y2)
        x_end = min(x_offset + grid_width, x2)
        actual_height = y_end - y_offset
        actual_width = x_end - x_offset
        
        region = image_patched[y_offset:y_offset + actual_height, x_offset:x_offset + actual_width]
        patch_region = patch_resized[:actual_height, :actual_width]
        
        # Create a mask that covers only non-white pixels.
        mask = np.any(patch_region != 255, axis=2)
        region[mask] = patch_region[mask]
        
        return image_patched
    
    def _calculate_fitness(self, parameters):
        """
        Calculate fitness for the given parameter vector.
        Evaluate each image using the current parameters. Do not share a set of
        successful images across particles or iterations, so one particle cannot affect another score.
        
        :param parameters: Parameter vector
        :return: Fitness value (higher is better)
        """
        if len(self.image_cache) == 0:
            return 0.0
        
        # Render the master patch.
        patch_master = self.bezier_patch.rasterize_master(parameters)
        
        total_fitness = 0.0
        
        for image_path, (image, target_bbox, clean_result) in self.image_cache.items():
            # Apply the patch.
            image_patched = self._apply_patch(image, patch_master, target_bbox)
            
            # Run detection.
            with torch.no_grad():
                result = inference_detector(self.model, image_patched)
            
            # Calculate fitness.
            fitness = self.detection_fitness(result)
            total_fitness += fitness
        
        return total_fitness / len(self.image_cache)
    
    def optimize(self):
        """
        Run PSO optimization.
        
        :return: Best parameter vector
        """
        print(f"\nStarting PSO optimization (particles={self.n_particles}, max iterations={self.max_iterations})")
        print(f"Using {len(self.image_cache)} images")
        
        fitness_history = []
        
        for iteration in tqdm(range(self.max_iterations), desc="PSO iterations"):
            # Calculate fitness for all particles.
            scores = np.array([self._calculate_fitness(self.pso.positions[i]) 
                              for i in range(self.n_particles)])
            
            # Update the particle swarm.
            self.pso.update(scores, constraint_fn=self.regularizer.apply_constraints)
            
            # Record fitness history.
            fitness_history.append(self.pso.global_best_score)
            
            # Print progress.
            if (iteration + 1) % 10 == 0 or iteration == 0:
                print(f"\nIteration {iteration+1}/{self.max_iterations}: "
                      f"Best fitness = {self.pso.global_best_score:.4f}")
        
        # Print the final result.
        print(f"\nPSO optimization complete: finished all {self.max_iterations} iterations")
        print(f"Final fitness = {self.pso.global_best_score:.4f}")
        
        return self.pso.global_best_position
    
    def evaluate(self, best_parameters):
        """
        Evaluate attack performance on the image set.
        
        :param best_parameters: Best parameter vector
        :return: (success_count, total_count)
        """
        print(f"\nEvaluating patch performance...")
        print(f"Number of images: {len(self.image_cache)}")
        
        if len(self.image_cache) == 0:
            print("Warning: no images are available for evaluation.")
            return 0, 0
        
        # Render the best patch.
        best_master_patch = self.bezier_patch.rasterize_master(best_parameters)
        
        # Evaluate the attack success rate.
        success_count = 0
        total_count = len(self.image_cache)
        
        for image_path, (image, target_bbox, clean_result) in tqdm(self.image_cache.items(), desc="Evaluate attack performance"):
            image_patched = self._apply_patch(image, best_master_patch, target_bbox)
            
            # Run detection; each call counts as one query.
            with torch.no_grad():
                result = inference_detector(self.model, image_patched)
            
            person_bboxes = result[0]
            if isinstance(person_bboxes, list):
                person_bboxes = person_bboxes[0] if len(person_bboxes) > 0 else np.array([])
            
            # Determine whether the attack succeeded.
            if len(person_bboxes) == 0:
                success_count += 1
            else:
                score = float(np.max(np.atleast_2d(person_bboxes)[:, -1]))

                if score < 0.5:
                    success_count += 1
        
        return success_count, total_count
