"""Bézier curved-grid rendering and topology constraints."""
import hashlib
import cv2
import numpy as np

class BezierPatch:
    """
    Bezier curve patch generator.
    Render a Bezier curved-grid image from control-point parameters.
    """
    def __init__(self, grid_dimension=2, fill_color=(0, 0, 0), render_scale=8):
        """
        :param grid_dimension: Grid dimension (2 for 2x2, 4 for 4x4)
        :param fill_color: Fill color (B, G, R)
        :param render_scale: Supersampling factor (render larger, then downsample)
        """
        self.grid_dimension = grid_dimension
        self.fill_color = fill_color
        self.render_scale = render_scale
        
        # Calculate the parameter dimensions.
        self.num_horizontal_segments = (grid_dimension + 1) * grid_dimension
        self.num_vertical_segments = grid_dimension * (grid_dimension + 1)
        self.total_lines = self.num_horizontal_segments + self.num_vertical_segments
        self.params_per_line = 6  # Each quadratic Bezier edge has 3 control points with 2 coordinates each.
        self.geometry_params = self.total_lines * self.params_per_line
        self.num_cells = grid_dimension * grid_dimension
        self.visibility_params = self.num_cells
        self.total_params = self.geometry_params + self.visibility_params
        
        # Cache master patches to improve rendering performance.
        self.master_size = 512
        self._master_cache = {}
        self._master_cache_max_items = 32
    
    def _master_cache_key(self, parameters, master_size: int) -> str:
        """Generate a cache key for the parameter vector."""
        arr = np.ascontiguousarray(parameters)
        h = hashlib.blake2b(arr.view(np.uint8), digest_size=16)
        h.update(str(int(master_size)).encode("utf-8"))
        return h.hexdigest()
    
    def bezier_curve(self, p0, p1, p2, num_points=100):
        """
        Evaluate a quadratic Bezier curve using recursive interpolation.
        :param p0, p1, p2: Three control points (x, y)
        :param num_points: Number of points sampled along the curve
        :return: Curve coordinates with shape (num_points, 2)
        """
        points = np.array([p0, p1, p2])
        
        def calNextPoints(pts, rate):
            if len(pts) == 1:
                return pts
            left = pts[0]
            ans = []
            for i in range(1, len(pts)):
                right = pts[i]
                disX = right[0] - left[0]
                disY = right[1] - left[1]
                nowX = left[0] + disX * rate
                nowY = left[1] + disY * rate
                ans.append([nowX, nowY])
                left = right
            return calNextPoints(np.array(ans), rate)
        
        curve_points = []
        for i in range(num_points):
            t = i / (num_points - 1) if num_points > 1 else 0
            point = calNextPoints(points, rate=t)
            curve_points.append(point[0])
        
        return np.array(curve_points, dtype=np.float32)
    
    def rasterize(self, parameters, width, height):
        """
        Convert a parameter vector to a filled Bezier curved-grid image.
        
        :param parameters: One-dimensional array of geometry and visibility parameters
        :param width: Target image width
        :param height: Target image height
        :return: Filled Bezier curved-grid image, (H, W, 3) uint8
        """
        target_H, target_W = int(height), int(width)
        render_H = target_H * self.render_scale
        render_W = target_W * self.render_scale
        
        # Initialize a white background.
        image = np.ones((render_H, render_W, 3), dtype=np.uint8) * 255
        
        # Split geometry and visibility parameters.
        geometry_params = parameters[:self.geometry_params]
        visibility_params = parameters[self.geometry_params:]
        
        # Reshape to (total_lines, 3, 2).
        params_reshaped = geometry_params.reshape(self.total_lines, 3, 2)
        
        # Map normalized coordinates to the rendering dimensions.
        params_scaled = params_reshaped.copy()
        params_scaled[:, :, 0] = params_reshaped[:, :, 0] * (render_W - 1)
        params_scaled[:, :, 1] = params_reshaped[:, :, 1] * (render_H - 1)
        
        # Build the visibility grid.
        visible_cells = visibility_params > 0.5
        visible_grid = visible_cells.reshape(self.grid_dimension, self.grid_dimension)
        
        # Precompute all Bezier curves.
        horizontal_curves = []
        vertical_curves = []
        curve_idx = 0
        
        # Horizontal segments.
        for row in range(self.grid_dimension + 1):
            row_curves = []
            for col in range(self.grid_dimension):
                p0, p1, p2 = params_scaled[curve_idx]
                curve_length = np.linalg.norm(p1 - p0) + np.linalg.norm(p2 - p1)
                num_points = max(100, int(curve_length * 2))
                curve_points = self.bezier_curve(p0, p1, p2, num_points=num_points)
                row_curves.append(curve_points)
                curve_idx += 1
            horizontal_curves.append(row_curves)
        
        # Vertical segments.
        for col in range(self.grid_dimension + 1):
            col_curves = []
            for row in range(self.grid_dimension):
                p0, p1, p2 = params_scaled[curve_idx]
                curve_length = np.linalg.norm(p1 - p0) + np.linalg.norm(p2 - p1)
                num_points = max(100, int(curve_length * 2))
                curve_points = self.bezier_curve(p0, p1, p2, num_points=num_points)
                col_curves.append(curve_points)
                curve_idx += 1
            vertical_curves.append(col_curves)
        
        # Fill visible cells.
        for row in range(self.grid_dimension):
            for col in range(self.grid_dimension):
                if visible_grid[row, col]:
                    top_curve = horizontal_curves[row][col]
                    bottom_curve = horizontal_curves[row + 1][col]
                    left_curve = vertical_curves[col][row]
                    right_curve = vertical_curves[col + 1][row]
                    
                    polygon_points = []
                    polygon_points.extend(top_curve)
                    polygon_points.extend(right_curve)
                    polygon_points.extend(bottom_curve[::-1])
                    polygon_points.extend(left_curve[::-1])
                    
                    polygon_int = np.array(polygon_points, dtype=np.int32).reshape(-1, 1, 2)
                    cv2.fillPoly(image, [polygon_int], self.fill_color)
        
        # Downsample to the target dimensions.
        if render_W != target_W or render_H != target_H:
            image = cv2.resize(image, (target_W, target_H), interpolation=cv2.INTER_AREA)
        
        return image
    
    def rasterize_master(self, parameters, use_cache=True):
        """
        Render a master patch at a fixed resolution for efficient reuse.
        
        :param parameters: Parameter vector
        :param use_cache: Whether to use the master-patch cache
        :return: Master patch, (master_size, master_size, 3) uint8
        """
        if not use_cache:
            return self.rasterize(parameters, self.master_size, self.master_size)
        
        key = self._master_cache_key(parameters, self.master_size)
        cached = self._master_cache.get(key)
        if cached is not None:
            return cached
        
        patch = self.rasterize(parameters, self.master_size, self.master_size)
        
        # Evict the oldest cache entry (FIFO).
        if len(self._master_cache) >= self._master_cache_max_items:
            try:
                oldest_key = next(iter(self._master_cache.keys()))
                self._master_cache.pop(oldest_key, None)
            except Exception:
                self._master_cache.clear()
        
        self._master_cache[key] = patch
        return patch
    
    def initialize_parameters(self):
        """
        Initialize a regular grid with random curvature and visibility.
        
        :return: Initial parameter vector with shape (total_params,)
        """
        params = []
        grid_spacing = 1.0 / self.grid_dimension
        amp = 0.45 * grid_spacing
        
        # Horizontal segments.
        for row in range(self.grid_dimension + 1):
            y = row / self.grid_dimension
            for col in range(self.grid_dimension):
                x_start = col / self.grid_dimension
                x_end = (col + 1) / self.grid_dimension
                x_mid = (x_start + x_end) / 2
                
                p0 = [x_start, y]
                p2 = [x_end, y]
                
                if row == 0:
                    offset_y = np.random.uniform(0.1 * amp, amp)
                elif row == self.grid_dimension:
                    offset_y = np.random.uniform(-amp, -0.1 * amp)
                else:
                    offset_y = np.random.uniform(-amp, amp)
                
                p1 = [x_mid, y + offset_y]
                params.extend([p0, p1, p2])
        
        # Vertical segments.
        for col in range(self.grid_dimension + 1):
            x = col / self.grid_dimension
            for row in range(self.grid_dimension):
                y_start = row / self.grid_dimension
                y_end = (row + 1) / self.grid_dimension
                y_mid = (y_start + y_end) / 2
                
                p0 = [x, y_start]
                p2 = [x, y_end]
                
                if col == 0:
                    offset_x = np.random.uniform(0.1 * amp, amp)
                elif col == self.grid_dimension:
                    offset_x = np.random.uniform(-amp, -0.1 * amp)
                else:
                    offset_x = np.random.uniform(-amp, amp)
                
                p1 = [x + offset_x, y_mid]
                params.extend([p0, p1, p2])
        
        params_array = np.array(params).flatten()
        params_array = np.clip(params_array, 0.0, 1.0)
        
        # Randomly initialize visibility; approximately half the cells are visible.
        visibility_array = np.random.rand(self.num_cells)
        
        return np.concatenate([params_array, visibility_array])

class GeometryRegularizer:
    """
    Geometry constraints to prevent tangled grids, collapsed points, and excessive curvature.
    Constrain the parameters to maintain a valid grid structure.
    """
    def __init__(self, grid_dimension):
        self.grid_dimension = grid_dimension
        self.num_horizontal_segments = (grid_dimension + 1) * grid_dimension
        self.num_vertical_segments = grid_dimension * (grid_dimension + 1)
        self.total_lines = self.num_horizontal_segments + self.num_vertical_segments
        self.geometry_params = self.total_lines * 6
        self.visibility_params = grid_dimension * grid_dimension
    
    def apply_constraints(self, parameters):
        """
        Apply orthogonal constraints and geometric safety bounds to prevent tangling.
        
        :param parameters: Parameter vector
        :return: Constrained parameter vector
        """
        params = parameters.copy()
        
        # Split geometry and visibility parameters.
        geometry_params = params[:self.geometry_params]
        visibility_params = params[self.geometry_params:]
        
        # Reshape the geometry parameters.
        params_reshaped = geometry_params.reshape(self.total_lines, 3, 2)
        grid_spacing = 1.0 / self.grid_dimension
        
        # Set the maximum displacement along the edge normal.
        max_amplitude = grid_spacing * 0.45
        
        curve_idx = 0
        
        # Constrain horizontal segments.
        for row in range(self.grid_dimension + 1):
            y_base = row / self.grid_dimension
            for col in range(self.grid_dimension):
                x_start = col / self.grid_dimension
                x_end = (col + 1) / self.grid_dimension
                x_mid = (x_start + x_end) / 2
                
                # Fix the endpoints.
                params_reshaped[curve_idx, 0] = [x_start, y_base]
                params_reshaped[curve_idx, 2] = [x_end, y_base]
                
                # Lock the axial coordinate to the midpoint; disallow horizontal motion.
                params_reshaped[curve_idx, 1, 0] = x_mid
                
                # Bound the normal displacement.
                if row == 0:
                    y_min, y_max = y_base, y_base + max_amplitude
                elif row == self.grid_dimension:
                    y_min, y_max = y_base - max_amplitude, y_base
                else:
                    y_min = y_base - max_amplitude
                    y_max = y_base + max_amplitude
                
                params_reshaped[curve_idx, 1, 1] = np.clip(
                    params_reshaped[curve_idx, 1, 1],
                    y_min,
                    y_max
                )
                
                curve_idx += 1
        
        # Constrain vertical segments.
        for col in range(self.grid_dimension + 1):
            x_base = col / self.grid_dimension
            for row in range(self.grid_dimension):
                y_start = row / self.grid_dimension
                y_end = (row + 1) / self.grid_dimension
                y_mid = (y_start + y_end) / 2
                
                # Fix the endpoints.
                params_reshaped[curve_idx, 0] = [x_base, y_start]
                params_reshaped[curve_idx, 2] = [x_base, y_end]
                
                # Lock the axial coordinate to the midpoint; disallow vertical motion.
                params_reshaped[curve_idx, 1, 1] = y_mid
                
                # Bound the normal displacement.
                if col == 0:
                    x_min, x_max = x_base, x_base + max_amplitude
                elif col == self.grid_dimension:
                    x_min, x_max = x_base - max_amplitude, x_base
                else:
                    x_min = x_base - max_amplitude
                    x_max = x_base + max_amplitude
                
                params_reshaped[curve_idx, 1, 0] = np.clip(
                    params_reshaped[curve_idx, 1, 0],
                    x_min,
                    x_max
                )
                
                curve_idx += 1
        
        # Flatten and return the constrained parameters.
        constrained_geometry = params_reshaped.flatten()
        visibility_params = np.clip(visibility_params, 0.0, 1.0)
        
        return np.concatenate([constrained_geometry, visibility_params])
