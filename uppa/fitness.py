"""Detection confidence fitness for universal patch optimization."""
import numpy as np

class DetectionFitness:
    """
    Detection fitness for evaluating adversarial attack performance.
    Extract fitness from detection results; higher values indicate a stronger attack.
    """
    def __call__(self, detection_result):
        """
        Calculate detection fitness for a single image.
        
        Untargeted attack: reduce detection confidence.
        - No person detected: fitness = 1.0 (best).
        - Otherwise: fitness = 1 - max_score (lower confidence gives higher fitness).
        
        :param detection_result: MMDetection inference result
        :return: Fitness value (higher is better)
        """
        person_bboxes = detection_result[0]
        
        # Handle result formats returned by different models.
        if isinstance(person_bboxes, list):
            person_bboxes = person_bboxes[0] if len(person_bboxes) > 0 else np.array([])
        
        if len(person_bboxes) == 0:
            return 1.0
        
        # Handle both one-dimensional and two-dimensional arrays.
        if person_bboxes.ndim == 1:
            max_score = person_bboxes[-1] if len(person_bboxes) > 0 else 0
        else:
            max_score = np.max(person_bboxes[:, -1]) if len(person_bboxes) > 0 else 0
        
        return 1.0 - max_score
