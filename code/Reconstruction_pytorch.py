import torch
import torch.nn as nn
import torch.nn.functional as F

class TomoReconstructionModel(nn.Module):
    def __init__(
            self, 
            volume_size,
            dim=2,
            initial_volume='gaussian'):
        super(TomoReconstructionModel, self).__init__()

        self.N = volume_size
        self.dim = dim
        self.initial_volume = initial_volume
        
        x = torch.arange(self.N) - self.N / 2
        self.xx, self.yy, self.zz = torch.meshgrid(x, x, x, indexing='ij')
        self.grid = torch.stack([self.xx, self.yy, self.zz], dim=-1)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to_device()
        self.initialize_volume()

    def to_device(self,):
        """
        Move the grid to the device.
        """

        self.grid = self.grid.to(self.device)

    def initialize_volume(self):
        """
        Initialize the volume.

        Returns:
        - volume (torch.Tensor): Initialized volume.
        """

        self.volume = nn.Parameter(torch.zeros(self.N, self.N, self.N))

        if self.initial_volume =='gaussian':
            x = torch.arange(self.N) - self.N / 2
            xx, yy, zz = torch.meshgrid(x, x, x)
            cloud = torch.exp(-0.1 * (xx**2 + yy**2 + zz**2))
            self.volume = nn.Parameter(cloud)
        elif self.initial_volume == 'constant':
            self.volume = nn.Parameter(torch.ones(self.N, self.N, self.N))
    

    def forward(self, projections, quaternions):
        """
        Forward pass of the model.

        Parameters:
        - projections (torch.Tensor): 2D projections of the volume.
        - quaternions (torch.Tensor): Quaternions representing rotations.

        Returns:
        - estimated_projections (torch.Tensor): Estimated 2D projections of the volume.
        """

        batch_size, _, _, _ = projections.shape
        estimated_projections = torch.zeros_like(projections)

        for i in range(batch_size):
            rotated_volume = self.apply_rotation(self.volume, quaternions[i])
            estimated_projections[i] = self.project(rotated_volume)
        
        return estimated_projections

    def apply_rotation(self, volume, q):
        """
        Rotate the object using quaternions.

        Parameters:
        - volume (torch.Tensor): The volume to rotate.
        - q (torch.Tensor): Quaternions representing rotations.

        Returns:
        - rotated_volume (torch.Tensor): Rotated volume.
        """

        # Convert quaternions to rotation matrix
        q = q / q.norm()  # Ensure unit quaternion
        R = self.quaternion_to_rotation_matrix(q)
        
        # Create a rotation grid
        grid = self.grid.view(-1, 3)
        rotated_grid = torch.matmul(grid, R.t()).view(self.N, self.N, self.N, 3)
        
        # Normalize the grid values to be in the range [-1, 1] for grid_sample
        rotated_grid = (rotated_grid / (self.N / 2)).clamp(-1, 1)
        
        # Apply grid_sample to rotate the volume
        rotated_volume = F.grid_sample(volume.unsqueeze(0).unsqueeze(0), rotated_grid.unsqueeze(0), align_corners=True)
        return rotated_volume.squeeze()

    def quaternion_to_rotation_matrix(self, q):
        """
        Convert a quaternion to a rotation matrix.

        Parameters:
        - q (torch.Tensor): Quaternions representing rotations.

        Returns:
        - R (torch.Tensor): Rotation matrix.
        """

        qw, qx, qy, qz = q
        R = torch.tensor([
            [1 - 2*qy*qy - 2*qz*qz, 2*qx*qy - 2*qz*qw, 2*qx*qz + 2*qy*qw],
            [2*qx*qy + 2*qz*qw, 1 - 2*qx*qx - 2*qz*qz, 2*qy*qz - 2*qx*qw],
            [2*qx*qz - 2*qy*qw, 2*qy*qz + 2*qx*qw, 1 - 2*qx*qx - 2*qy*qy]
        ], dtype=torch.float32, device=self.device)
        return R

    def project(self, volume):
        """
        Project the volume to a 2D plane.

        Parameters:
        - volume (torch.Tensor): The volume to project.

        Returns:
        - projection (torch.Tensor): 2D projection of the volume.
        """

        projection = torch.sum(volume, dim=self.dim)
        return projection

    def full_projection(self, volume, quaternions):
        """
        Compute the full projection for a set of quaternions.

        Parameters:
        - volume (torch.Tensor): The volume to project.
        - quaternions (torch.Tensor): Set of quaternions for projection.

        Returns:
        - projections (torch.Tensor): 2D projections of the volume.
        """

        projections = torch.zeros(len(quaternions), volume.shape[1], volume.shape[2], device=self.device)

        for i in range(len(quaternions)):
            rotated_volume = self.apply_rotation(volume, quaternions[i])
            projections[i] = self.project(rotated_volume)
        
        return projections
    

