"""Particle swarm optimization for patch parameters."""
import numpy as np

class PSO:
    """
    Particle swarm optimizer.
    Update particle positions and velocities independently of the attack workflow.
    """
    def __init__(self, dim, n_particles, w=0.9, c1=1.6, c2=1.4, r1=0.5, r2=0.5, v_max=0.1):
        """
        :param dim: Parameter dimensionality
        :param n_particles: Number of particles
        :param w: Inertia weight
        :param c1: Cognitive coefficient
        :param c2: Social coefficient
        :param r1: First acceleration scalar (fixed in the manuscript)
        :param r2: Second acceleration scalar (fixed in the manuscript)
        :param v_max: Maximum velocity magnitude
        """
        self.dim = dim
        self.n_particles = n_particles
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.r1 = r1
        self.r2 = r2
        self.v_max = v_max
        
        # Initialize particle positions and velocities.
        self.positions = np.random.rand(n_particles, dim)
        self.velocities = np.random.uniform(-v_max, v_max, (n_particles, dim))
        
        # Initialize personal and global best states.
        self.personal_best_positions = self.positions.copy()
        self.personal_best_scores = np.full(n_particles, -np.inf)
        self.global_best_position = None
        self.global_best_score = -np.inf
    
    def update(self, scores, constraint_fn=None):
        """
        Update the particle swarm state.
        
        :param scores: Current fitness scores for all particles, shape (n_particles,)
        :param constraint_fn: Optional function that constrains particle positions
        """
        # Update personal best states.
        for i in range(self.n_particles):
            if scores[i] > self.personal_best_scores[i]:
                self.personal_best_scores[i] = scores[i]
                self.personal_best_positions[i] = self.positions[i].copy()
            
            # Update the global best state.
            if scores[i] > self.global_best_score:
                self.global_best_score = scores[i]
                self.global_best_position = self.positions[i].copy()
        
        # Update velocities and positions.
        self.velocities = (self.w * self.velocities + 
                          self.c1 * self.r1 * (self.personal_best_positions - self.positions) +
                          self.c2 * self.r2 * (self.global_best_position - self.positions))
        
        # Limit velocity magnitudes.
        self.velocities = np.clip(self.velocities, -self.v_max, self.v_max)
        
        # Update particle positions.
        self.positions = self.positions + self.velocities
        
        # Clip positions to the unit interval.
        self.positions = np.clip(self.positions, 0.0, 1.0)
        
        # Apply position constraints.
        if constraint_fn is not None:
            for i in range(self.n_particles):
                self.positions[i] = constraint_fn(self.positions[i])
