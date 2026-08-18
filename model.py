import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from skimage.measure import label
from skimage.morphology import binary_closing, disk

try:
    from sklearn.cluster import DBSCAN
    from sklearn.neighbors import NearestNeighbors
    SKLEARN_AVAILABLE = True
except ImportError:
    DBSCAN = None
    NearestNeighbors = None
    SKLEARN_AVAILABLE = False

class DualResidualBlock(nn.Module):

    def __init__(self, channels, reduction=8):
        super(DualResidualBlock, self).__init__()
        self.channels = channels
        self.main_branch = nn.Sequential(nn.Conv2d(channels, channels, kernel_size=3, padding=1), nn.PReLU(), nn.Conv2d(channels, channels, kernel_size=3, padding=1), nn.PReLU(), nn.Conv2d(channels, channels, kernel_size=1))
        self.channel_attention = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(channels, max(channels // reduction, 1), 1), nn.PReLU(), nn.Conv2d(max(channels // reduction, 1), channels, 1), nn.Sigmoid())
        color_channels = channels // 4
        self.channel_protection = nn.Parameter(torch.cat([torch.ones(color_channels) * 1.3, torch.ones(channels - color_channels)]))
        self.short_residual_weight = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        identity = x
        out = self.main_branch(x)
        ca_weight = self.channel_attention(out)
        protected_weight = ca_weight * self.channel_protection.view(1, -1, 1, 1)
        protected_weight = protected_weight / (protected_weight.mean() + 1e-08)
        out = out * protected_weight
        out = out + self.short_residual_weight * identity
        return out

class LongResidualConnection(nn.Module):

    def __init__(self, input_channels, output_channels):
        super(LongResidualConnection, self).__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        if input_channels != output_channels:
            self.channel_adapter = nn.Sequential(nn.Conv2d(input_channels, output_channels, kernel_size=1), nn.PReLU())
        else:
            self.channel_adapter = nn.Identity()
        self.long_residual_weight = nn.Parameter(torch.tensor(0.3))

    def forward(self, input_feat, output_feat):
        adapted_input = self.channel_adapter(input_feat)
        if adapted_input.shape[2:] != output_feat.shape[2:]:
            adapted_input = F.interpolate(adapted_input, size=output_feat.shape[2:], mode='bilinear', align_corners=False)
        result = output_feat + self.long_residual_weight * adapted_input
        return result

class AdaptiveFrequencyLearning(nn.Module):

    def __init__(self, channels, reduction=4):
        super(AdaptiveFrequencyLearning, self).__init__()
        self.channels = channels
        self.reduction = reduction
        reduced_channels = max(channels // reduction, 8)
        self.freq_compress = nn.Conv2d(channels, reduced_channels, 1, bias=False)
        self.freq_weight_real = nn.Parameter(torch.randn(reduced_channels, 1, 1) * 0.01)
        self.freq_weight_imag = nn.Parameter(torch.randn(reduced_channels, 1, 1) * 0.01)

        def find_valid_groups(channels, max_groups=8):
            for groups in range(min(channels, max_groups), 0, -1):
                if channels % groups == 0:
                    return groups
            return 1
        valid_groups = find_valid_groups(reduced_channels)
        self.freq_enhance = nn.Sequential(nn.Conv2d(reduced_channels, reduced_channels, 3, padding=1, groups=valid_groups), nn.PReLU(), DualResidualBlock(reduced_channels), nn.Conv2d(reduced_channels, reduced_channels, 1), nn.PReLU())
        self.freq_expand = nn.Conv2d(reduced_channels, channels, 1, bias=False)
        self.adaptive_weight = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(channels, max(channels // 8, 1), 1), nn.PReLU(), nn.Conv2d(max(channels // 8, 1), 1, 1), nn.Sigmoid())

    def forward(self, x):
        try:
            b, c, h, w = x.shape
            x_compressed = self.freq_compress(x)
            x_freq = torch.fft.fft2(x_compressed, dim=(-2, -1))
            x_real = x_freq.real
            x_imag = x_freq.imag
            x_real_filtered = x_real * self.freq_weight_real
            x_imag_filtered = x_imag * self.freq_weight_imag
            x_freq_filtered = torch.complex(x_real_filtered, x_imag_filtered)
            x_spatial = torch.fft.ifft2(x_freq_filtered, dim=(-2, -1)).real
            x_enhanced = self.freq_enhance(x_spatial)
            x_restored = self.freq_expand(x_enhanced)
            weight = self.adaptive_weight(x)
            output = x + weight * x_restored
            return output
        except Exception:
            return x

class EnhancedLBPExtractor(nn.Module):

    def __init__(self, radius=1, n_points=8, channels=None):
        super(EnhancedLBPExtractor, self).__init__()
        self.radius = radius
        self.n_points = n_points
        angles = torch.linspace(0, 2 * np.pi, n_points + 1)[:-1]
        x_offsets = radius * torch.cos(angles)
        y_offsets = radius * torch.sin(angles)
        self.register_buffer('offsets', torch.stack([y_offsets, x_offsets], dim=1))
        if channels:
            self.texture_enhance = nn.Sequential(nn.Conv2d(channels, channels, kernel_size=3, padding=1), nn.PReLU(), DualResidualBlock(channels), nn.Conv2d(channels, channels, kernel_size=1), nn.PReLU())
        else:
            self.texture_enhance = nn.Identity()

    def forward(self, x):
        batch_size, channels, height, width = x.shape
        device = x.device
        y_coords = torch.arange(height, device=device, dtype=torch.float32)
        x_coords = torch.arange(width, device=device, dtype=torch.float32)
        y_grid, x_grid = torch.meshgrid(y_coords, x_coords, indexing='ij')
        x_grid_norm = 2.0 * x_grid / (width - 1) - 1.0
        y_grid_norm = 2.0 * y_grid / (height - 1) - 1.0
        lbp_features = []
        for c in range(min(channels, 8)):
            channel_data = x[:, c:c + 1]
            center_values = channel_data
            binary_codes = []
            for i in range(self.n_points):
                offset_y = self.offsets[i, 0] / height * 2.0
                offset_x = self.offsets[i, 1] / width * 2.0
                neighbor_x = (x_grid_norm + offset_x).clamp(-1, 1)
                neighbor_y = (y_grid_norm + offset_y).clamp(-1, 1)
                grid = torch.stack([neighbor_x, neighbor_y], dim=-1).unsqueeze(0).repeat(batch_size, 1, 1, 1)
                neighbor_values = F.grid_sample(channel_data, grid, mode='bilinear', padding_mode='border', align_corners=True)
                binary_code = (neighbor_values >= center_values).float()
                binary_codes.append(binary_code)
            lbp_code = torch.zeros_like(center_values)
            for i, binary_code in enumerate(binary_codes):
                lbp_code += binary_code * 2 ** i
            lbp_code = lbp_code / (2 ** self.n_points - 1)
            lbp_features.append(lbp_code)
        if lbp_features:
            lbp_output = torch.cat(lbp_features, dim=1)
            if channels > len(lbp_features):
                padding = torch.zeros(batch_size, channels - len(lbp_features), height, width, device=device)
                lbp_output = torch.cat([lbp_output, padding], dim=1)
        else:
            lbp_output = torch.zeros_like(x)
        enhanced_output = self.texture_enhance(lbp_output)
        return enhanced_output

class EnhancedDensityBasedClustering:

    def __init__(self, eps_ratio=0.3, min_samples_ratio=0.01, min_area_ratio=0.005):
        self.eps_ratio = eps_ratio
        self.min_samples_ratio = min_samples_ratio
        self.min_area_ratio = min_area_ratio

    def ensure_connectivity(self, labels, height, width, min_area=None):
        if min_area is None:
            min_area = max(int(height * width * self.min_area_ratio), 4)
        labels_2d = labels.reshape(height, width)
        new_labels = np.zeros_like(labels_2d, dtype=np.int32)
        current_label = 0
        unique_labels = np.unique(labels_2d)
        valid_labels = [label_val for label_val in unique_labels if label_val != -1]
        for label_val in valid_labels:
            mask = (labels_2d == label_val).astype(np.uint8)
            try:
                kernel = disk(1)
                mask_closed = binary_closing(mask, kernel).astype(np.uint8)
            except Exception:
                mask_closed = mask
            try:
                labeled_components, num_components = label(mask_closed, return_num=True)
                for component_id in range(1, num_components + 1):
                    component_mask = labeled_components == component_id
                    component_area = np.sum(component_mask)
                    if component_area >= min_area:
                        new_labels[component_mask] = current_label
                        current_label += 1
                    else:
                        new_labels[component_mask] = current_label
                        current_label += 1
            except Exception:
                mask_coords = np.where(mask)
                if len(mask_coords[0]) > 0:
                    new_labels[mask_coords] = current_label
                    current_label += 1
        unassigned_mask = (new_labels == 0) & (labels_2d == -1)
        if np.any(unassigned_mask) and current_label > 0:
            new_labels = self._assign_unassigned_pixels(new_labels, unassigned_mask)
        if current_label == 0:
            new_labels[:] = 0
            current_label = 1
        return new_labels.flatten()

    def _assign_unassigned_pixels(self, labels, unassigned_mask):
        height, width = labels.shape
        assigned_mask = labels > 0
        if not np.any(assigned_mask):
            return labels
        unassigned_coords = np.where(unassigned_mask)
        for y, x in zip(unassigned_coords[0], unassigned_coords[1]):
            search_radius = 1
            found = False
            while search_radius <= min(height, width) // 4 and (not found):
                y_min = max(0, y - search_radius)
                y_max = min(height, y + search_radius + 1)
                x_min = max(0, x - search_radius)
                x_max = min(width, x + search_radius + 1)
                neighborhood = labels[y_min:y_max, x_min:x_max]
                assigned_in_neighborhood = neighborhood[neighborhood > 0]
                if len(assigned_in_neighborhood) > 0:
                    unique, counts = np.unique(assigned_in_neighborhood, return_counts=True)
                    labels[y, x] = unique[np.argmax(counts)]
                    found = True
                search_radius += 1
            if not found:
                labels[y, x] = 0
        return labels

    def adaptive_dbscan(self, features, spatial_coords):
        try:
            height = int(np.max(spatial_coords[:, 0])) + 1
            width = int(np.max(spatial_coords[:, 1])) + 1
        except Exception:
            height, width = (64, 64)
        if not SKLEARN_AVAILABLE:
            return self._create_basic_clusters(spatial_coords, height, width)
        try:
            feature_std = np.std(features, axis=0)
            feature_mean = np.mean(features, axis=0)
            spatial_std = np.std(spatial_coords, axis=0)
            spatial_mean = np.mean(spatial_coords, axis=0)
            feature_std = np.where(feature_std < 1e-08, 1.0, feature_std)
            spatial_std = np.where(spatial_std < 1e-08, 1.0, spatial_std)
            features_norm = (features - feature_mean) / feature_std
            spatial_norm = (spatial_coords - spatial_mean) / spatial_std
            combined_features = np.concatenate([features_norm, spatial_norm * 3.0], axis=1)
            n_samples = len(combined_features)
            min_samples = max(int(n_samples * self.min_samples_ratio), 3)
            min_samples = min(min_samples, 10)
            try:
                nbrs = NearestNeighbors(n_neighbors=min(min_samples + 1, n_samples))
                nbrs.fit(combined_features)
                distances, _ = nbrs.kneighbors(combined_features)
                k_distance = distances[:, -1]
                eps_base = np.percentile(k_distance, 50)
                eps = max(eps_base * self.eps_ratio, 0.1)
            except Exception:
                eps = 0.5
            for attempt in range(3):
                try:
                    current_eps = eps * (1.0 + attempt * 0.5)
                    current_min_samples = max(min_samples - attempt, 2)
                    dbscan = DBSCAN(eps=current_eps, min_samples=current_min_samples, metric='euclidean')
                    cluster_labels = dbscan.fit_predict(combined_features)
                    n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
                    if n_clusters >= 2 and n_clusters <= 20:
                        connected_labels = self.ensure_connectivity(cluster_labels, height, width)
                        final_clusters = len(np.unique(connected_labels))
                        if final_clusters >= 2:
                            return connected_labels
                    elif n_clusters == 1:
                        eps *= 0.7
                    elif n_clusters == 0:
                        eps *= 1.5
                except Exception:
                    continue
            return self._create_basic_clusters(spatial_coords, height, width)
        except Exception:
            return self._create_basic_clusters(spatial_coords, height, width)

    def _create_basic_clusters(self, spatial_coords, height, width):
        labels = np.zeros(len(spatial_coords), dtype=np.int32)
        h_mid = height // 2
        w_mid = width // 2
        for i, (y, x) in enumerate(spatial_coords):
            y, x = (int(y), int(x))
            if y < h_mid and x < w_mid:
                labels[i] = 0
            elif y < h_mid and x >= w_mid:
                labels[i] = 1
            elif y >= h_mid and x < w_mid:
                labels[i] = 2
            else:
                labels[i] = 3
        return labels

class PositionalEncoding(nn.Module):

    def __init__(self, channels, max_position=512):
        super(PositionalEncoding, self).__init__()
        self.channels = channels
        self.max_position = max_position
        pe = torch.zeros(max_position, channels)
        position = torch.arange(0, max_position, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, channels, 2).float() * (-np.log(10000.0) / channels))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
        self.position_weight = nn.Parameter(torch.ones(1) * 0.1)
        self.position_fusion = nn.Sequential(nn.Linear(channels, channels // 2), nn.ReLU(inplace=True), nn.Linear(channels // 2, channels), nn.Sigmoid())

    def forward(self, x, positions):
        batch_size, channels, height, width = x.shape
        device = x.device
        if positions is None or positions.shape[1] == 0:
            return (x, None)
        try:
            normalized_positions = positions.clone()
            normalized_positions[:, :, 0] = positions[:, :, 0] / height * (self.max_position - 1)
            normalized_positions[:, :, 1] = positions[:, :, 1] / width * (self.max_position - 1)
            normalized_positions = normalized_positions.long().clamp(0, self.max_position - 1)
            block_encodings = []
            relative_encodings = []
            for b in range(batch_size):
                batch_positions = normalized_positions[b]
                num_blocks = batch_positions.shape[0]
                if num_blocks == 0:
                    block_encodings.append(torch.zeros(0, channels, device=device))
                    relative_encodings.append(torch.zeros(0, 0, 2, device=device))
                    continue
                y_encodings = self.pe[batch_positions[:, 0]]
                x_encodings = self.pe[batch_positions[:, 1]]
                combined_encoding = (y_encodings + x_encodings) / 2.0
                block_encodings.append(combined_encoding)
                relative_pos = self._compute_relative_positions(batch_positions, num_blocks)
                relative_encodings.append(relative_pos)
            enhanced_features = self._apply_position_encoding_to_features(x, block_encodings, relative_encodings, positions)
            return (enhanced_features, {'block_encodings': block_encodings, 'relative_encodings': relative_encodings, 'positions': positions})
        except Exception as e:
            return (x, None)

    def _compute_relative_positions(self, positions, num_blocks):
        if num_blocks <= 1:
            return torch.zeros(num_blocks, num_blocks, 2, device=positions.device)
        positions_expanded_i = positions.unsqueeze(1).expand(-1, num_blocks, -1)
        positions_expanded_j = positions.unsqueeze(0).expand(num_blocks, -1, -1)
        relative_positions = positions_expanded_i - positions_expanded_j
        return relative_positions

    def _apply_position_encoding_to_features(self, features, block_encodings, relative_encodings, positions):
        batch_size, channels, height, width = features.shape
        device = features.device
        enhanced_features = features.clone()
        for b in range(batch_size):
            if len(block_encodings[b]) == 0:
                continue
            current_positions = positions[b]
            current_encodings = block_encodings[b]
            current_relative = relative_encodings[b]
            try:
                global_context = self._create_global_context(current_encodings, current_relative, current_positions, height, width)
            except Exception:
                global_context = torch.zeros(channels, device=device)
            position_map = torch.zeros(channels, height, width, device=device)
            for block_idx, pos in enumerate(current_positions):
                try:
                    y, x = (int(pos[0].item()), int(pos[1].item()))
                    y = max(0, min(y, height - 1))
                    x = max(0, min(x, width - 1))
                    block_encoding = current_encodings[block_idx]
                    radius = min(4, height // 4, width // 4)
                    y_range = (max(0, y - radius), min(height, y + radius + 1))
                    x_range = (max(0, x - radius), min(width, x + radius + 1))
                    for dy in range(y_range[0], y_range[1]):
                        for dx in range(x_range[0], x_range[1]):
                            dist = np.sqrt((dy - y) ** 2 + (dx - x) ** 2)
                            weight = np.exp(-dist / 2.0)
                            position_map[:, dy, dx] += block_encoding * weight * 0.1
                except Exception:
                    continue
            try:
                if global_context.shape[0] == channels:
                    position_weight = self.position_weight * torch.sigmoid(global_context)
                    position_weight = position_weight.view(channels, 1, 1)
                    enhanced_features[b] = enhanced_features[b] + position_weight * position_map
                else:
                    enhanced_features[b] = enhanced_features[b] + self.position_weight * position_map
            except Exception:
                continue
        return enhanced_features

    def _create_global_context(self, block_encodings, relative_encodings, positions, height, width):
        num_blocks = len(block_encodings)
        if num_blocks == 0:
            return torch.zeros(self.channels, device=block_encodings.device)
        try:
            position_variance = torch.var(positions.float(), dim=0)
            position_mean = torch.mean(positions.float(), dim=0)
            if num_blocks > 1:
                relative_distances = torch.norm(relative_encodings.float(), dim=2)
                distance_stats = torch.tensor([torch.mean(relative_distances), torch.std(relative_distances), torch.max(relative_distances), torch.min(relative_distances[relative_distances > 0]) if torch.sum(relative_distances > 0) > 0 else torch.tensor(0.0)], device=block_encodings.device)
            else:
                distance_stats = torch.zeros(4, device=block_encodings.device)
            global_encoding_mean = torch.mean(block_encodings, dim=0)
            global_info_parts = [global_encoding_mean]
            remaining_dims = self.channels - global_encoding_mean.shape[0]
            if remaining_dims > 0:
                pos_info = torch.cat([position_variance, position_mean, distance_stats])
                repeat_times = (remaining_dims + len(pos_info) - 1) // len(pos_info)
                repeated_pos_info = pos_info.repeat(repeat_times)[:remaining_dims]
                global_info_parts.append(repeated_pos_info)
            global_info = torch.cat(global_info_parts)[:self.channels]
            global_context = self.position_fusion(global_info)
            return global_context
        except Exception:
            return torch.zeros(self.channels, device=block_encodings.device)

class RelativePositionBias(nn.Module):

    def __init__(self, channels, max_relative_position=64):
        super(RelativePositionBias, self).__init__()
        self.channels = channels
        self.max_relative_position = max_relative_position
        self.relative_position_bias_table = nn.Parameter(torch.zeros((2 * max_relative_position - 1) ** 2, channels))
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)
        coords_h = torch.arange(max_relative_position)
        coords_w = torch.arange(max_relative_position)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += max_relative_position - 1
        relative_coords[:, :, 1] += max_relative_position - 1
        relative_coords[:, :, 0] *= 2 * max_relative_position - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer('relative_position_index', relative_position_index)

    def forward(self, num_blocks):
        if num_blocks <= 1:
            return torch.zeros(1, 1, self.channels, device=self.relative_position_bias_table.device)
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index[:num_blocks, :num_blocks].view(-1)].view(num_blocks, num_blocks, -1)
        return relative_position_bias.permute(2, 0, 1).contiguous()

class AdaptiveWoodGrainAnalyzer(nn.Module):

    def __init__(self, channels=200):
        super(AdaptiveWoodGrainAnalyzer, self).__init__()
        self.channels = channels
        self.density_clusterer = EnhancedDensityBasedClustering(eps_ratio=0.3, min_samples_ratio=0.01, min_area_ratio=0.005)
        self.position_encoder = PositionalEncoding(channels, max_position=512)
        self.relative_position_bias = RelativePositionBias(channels, max_relative_position=64)
        self.gabor_angles = [0, 30, 60, 90, 120, 150]
        self.gabor_frequencies = [0.1, 0.3, 0.5]
        self.texture_extractor = nn.Sequential(nn.Conv2d(channels, channels // 2, kernel_size=3, padding=1), nn.PReLU(), DualResidualBlock(channels // 2), nn.Conv2d(channels // 2, channels // 4, kernel_size=3, padding=1), nn.PReLU(), nn.Conv2d(channels // 4, len(self.gabor_angles), kernel_size=1), nn.Sigmoid())
        self.direction_continuity = nn.Sequential(nn.Conv2d(channels, channels // 4, kernel_size=5, padding=2), nn.PReLU(), DualResidualBlock(channels // 4), nn.Conv2d(channels // 4, 3, kernel_size=3, padding=1), nn.Sigmoid())
        self.local_texture_descriptor = nn.Sequential(nn.Conv2d(channels, channels // 4, kernel_size=7, padding=3), nn.PReLU(), DualResidualBlock(channels // 4), nn.Conv2d(channels // 4, 4, kernel_size=1), nn.Sigmoid())
        self.spatial_analyzer = nn.Sequential(nn.Conv2d(channels, channels // 8, kernel_size=3, padding=1), nn.PReLU(), nn.Conv2d(channels // 8, 2, kernel_size=1), nn.Tanh())
        self.global_context_fusion = nn.Sequential(nn.Conv2d(channels * 2, channels, kernel_size=1), nn.PReLU(), DualResidualBlock(channels), nn.Conv2d(channels, channels, kernel_size=3, padding=1), nn.PReLU(), nn.Conv2d(channels, channels, kernel_size=1))

    def create_gabor_kernel(self, angle, frequency, sigma_x=2, sigma_y=2, kernel_size=15):
        angle_rad = np.radians(angle)
        cos_angle = np.cos(angle_rad)
        sin_angle = np.sin(angle_rad)
        x = np.arange(-kernel_size // 2, kernel_size // 2 + 1)
        y = np.arange(-kernel_size // 2, kernel_size // 2 + 1)
        X, Y = np.meshgrid(x, y)
        x_prime = X * cos_angle + Y * sin_angle
        y_prime = -X * sin_angle + Y * cos_angle
        gaussian = np.exp(-(x_prime ** 2 / (2 * sigma_x ** 2) + y_prime ** 2 / (2 * sigma_y ** 2)))
        sinusoid = np.cos(2 * np.pi * frequency * x_prime)
        gabor = gaussian * sinusoid
        gabor = gabor / np.sqrt(np.sum(gabor ** 2))
        return torch.FloatTensor(gabor)

    def extract_comprehensive_features(self, x):
        batch_size, channels, height, width = x.shape
        device = x.device
        try:
            direction_features = self.direction_continuity(x)
            texture_features = self.local_texture_descriptor(x)
            spatial_features = self.spatial_analyzer(x)
            gabor_features = self.apply_gabor_filters(x)
            all_features = torch.cat([direction_features, texture_features, spatial_features, gabor_features], dim=1)
            return (all_features, direction_features, texture_features, spatial_features)
        except Exception as e:
            direction_features = torch.randn(batch_size, 3, height, width, device=device)
            texture_features = torch.randn(batch_size, 4, height, width, device=device)
            spatial_features = torch.randn(batch_size, 2, height, width, device=device)
            return (None, direction_features, texture_features, spatial_features)

    def apply_gabor_filters(self, x):
        batch_size, channels, height, width = x.shape
        device = x.device
        try:
            if channels > 1:
                x_gray = torch.mean(x, dim=1, keepdim=True)
            else:
                x_gray = x
            gabor_responses = []
            for angle in self.gabor_angles[:3]:
                for freq in self.gabor_frequencies[:2]:
                    try:
                        kernel = self.create_gabor_kernel(angle, freq).to(device)
                        kernel = kernel.unsqueeze(0).unsqueeze(0)
                        padding = kernel.shape[-1] // 2
                        response = F.conv2d(x_gray, kernel, padding=padding)
                        if response.shape[2:] != (height, width):
                            response = F.interpolate(response, size=(height, width), mode='bilinear', align_corners=False)
                        gabor_responses.append(response)
                    except Exception:
                        response = torch.zeros(batch_size, 1, height, width, device=device)
                        gabor_responses.append(response)
            if gabor_responses:
                gabor_features = torch.cat(gabor_responses, dim=1)
            else:
                gabor_features = torch.zeros(batch_size, 6, height, width, device=device)
            return gabor_features
        except Exception:
            return torch.zeros(batch_size, 6, height, width, device=device)

    def create_adaptive_texture_blocks(self, x, num_blocks=6):
        batch_size, channels, height, width = x.shape
        device = x.device
        try:
            all_features, direction_features, texture_features, spatial_features = self.extract_comprehensive_features(x)
        except Exception as e:
            direction_features = torch.randn(batch_size, 3, height, width, device=device)
            texture_features = torch.randn(batch_size, 4, height, width, device=device)
            return self._simple_grid_blocks(x, num_blocks, direction_features, texture_features)
        adaptive_blocks = []
        adaptive_masks = []
        block_positions_list = []
        for b in range(batch_size):
            try:
                current_direction = direction_features[b].detach().cpu().numpy()
                current_texture = texture_features[b].detach().cpu().numpy()
                current_spatial = spatial_features[b].detach().cpu().numpy()
                y_coords, x_coords = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')
                spatial_coords = np.stack([y_coords.flatten(), x_coords.flatten()], axis=1)
                feature_vectors = []
                for c in range(current_direction.shape[0]):
                    feature_vectors.append(current_direction[c].flatten())
                for c in range(current_texture.shape[0]):
                    feature_vectors.append(current_texture[c].flatten())
                for c in range(current_spatial.shape[0]):
                    feature_vectors.append(current_spatial[c].flatten())
                features = np.stack(feature_vectors, axis=1)
                block_labels = self.density_clusterer.adaptive_dbscan(features, spatial_coords)
                block_labels = block_labels.reshape(height, width)
                unique_labels = np.unique(block_labels)
                if len(unique_labels) < 2:
                    pass
            except Exception as e:
                block_labels = self._create_simple_labels(height, width, num_blocks)
            try:
                current_blocks, current_masks, block_positions = self._create_blocks_from_labels_with_positions(x[b:b + 1], block_labels, num_blocks, device)
                adaptive_blocks.append(current_blocks)
                adaptive_masks.append(current_masks)
                block_positions_list.append(block_positions)
            except Exception as e:
                simple_blocks, simple_masks, simple_positions = self._create_simple_blocks(x[b:b + 1], num_blocks, height, width, device)
                adaptive_blocks.append(simple_blocks)
                adaptive_masks.append(simple_masks)
                block_positions_list.append(simple_positions)
        try:
            enhanced_features = self._apply_global_position_encoding(x, adaptive_blocks, adaptive_masks, block_positions_list)
        except Exception as e:
            enhanced_features = x
        return (adaptive_blocks, adaptive_masks, direction_features, texture_features, enhanced_features)

    def _create_blocks_from_labels_with_positions(self, x, block_labels, num_blocks, device):
        height, width = block_labels.shape
        current_blocks = []
        current_masks = []
        block_positions = []
        unique_labels = np.unique(block_labels)
        for label_val in unique_labels:
            if len(current_blocks) >= num_blocks:
                break
            mask = (block_labels == label_val).astype(np.float32)
            mask_area = np.sum(mask)
            if mask_area > 0:
                mask_tensor = torch.FloatTensor(mask).to(device).unsqueeze(0).unsqueeze(0)
                block_feature = x * mask_tensor
                current_blocks.append(block_feature)
                current_masks.append(mask_tensor)
                y_indices, x_indices = np.where(mask > 0)
                if len(y_indices) > 0:
                    center_y = np.mean(y_indices)
                    center_x = np.mean(x_indices)
                    block_positions.append([center_y, center_x])
                else:
                    block_positions.append([height // 2, width // 2])
        while len(current_blocks) < num_blocks:
            if current_blocks:
                current_blocks.append(current_blocks[-1].clone())
                current_masks.append(current_masks[-1].clone())
                block_positions.append(block_positions[-1].copy())
            else:
                mask_tensor = torch.ones(1, 1, height, width, device=device)
                block_feature = x * mask_tensor
                current_blocks.append(block_feature)
                current_masks.append(mask_tensor)
                block_positions.append([height // 2, width // 2])
        return (current_blocks[:num_blocks], current_masks[:num_blocks], block_positions[:num_blocks])

    def _create_simple_blocks(self, x, num_blocks, height, width, device):
        blocks = []
        masks = []
        positions = []
        rows = int(np.sqrt(num_blocks))
        cols = (num_blocks + rows - 1) // rows
        h_step = height // rows
        w_step = width // cols
        for i in range(rows):
            for j in range(cols):
                if len(blocks) >= num_blocks:
                    break
                start_h = i * h_step
                end_h = height if i == rows - 1 else (i + 1) * h_step
                start_w = j * w_step
                end_w = width if j == cols - 1 else (j + 1) * w_step
                mask = torch.zeros(1, 1, height, width, device=device)
                mask[:, :, start_h:end_h, start_w:end_w] = 1.0
                block_feature = x * mask
                blocks.append(block_feature)
                masks.append(mask)
                center_y = (start_h + end_h) / 2
                center_x = (start_w + end_w) / 2
                positions.append([center_y, center_x])
        while len(blocks) < num_blocks:
            blocks.append(blocks[-1].clone())
            masks.append(masks[-1].clone())
            positions.append(positions[-1].copy())
        return (blocks[:num_blocks], masks[:num_blocks], positions[:num_blocks])

    def _apply_global_position_encoding(self, x, adaptive_blocks, adaptive_masks, block_positions_list):
        batch_size, channels, height, width = x.shape
        device = x.device
        try:
            batch_positions = []
            for positions in block_positions_list:
                if positions:
                    positions_tensor = torch.FloatTensor(positions).to(device)
                    batch_positions.append(positions_tensor)
                else:
                    default_positions = torch.FloatTensor([[height // 2, width // 2]]).to(device)
                    batch_positions.append(default_positions)
            max_blocks = max((len(pos) for pos in batch_positions)) if batch_positions else 1
            padded_positions = torch.zeros(batch_size, max_blocks, 2, device=device)
            for b, positions in enumerate(batch_positions):
                if len(positions) > 0:
                    num_positions = min(len(positions), max_blocks)
                    padded_positions[b, :num_positions] = positions[:num_positions]
            enhanced_features, position_info = self.position_encoder(x, padded_positions)
            if enhanced_features.shape == x.shape:
                combined_features = torch.cat([x, enhanced_features], dim=1)
                final_features = self.global_context_fusion(combined_features)
                final_features = x + final_features
            else:
                final_features = x
            return final_features
        except Exception as e:
            return x

    def _create_simple_labels(self, height, width, num_blocks):
        labels = np.zeros((height, width), dtype=np.int32)
        if num_blocks == 4:
            h_mid = height // 2
            w_mid = width // 2
            labels[:h_mid, :w_mid] = 0
            labels[:h_mid, w_mid:] = 1
            labels[h_mid:, :w_mid] = 2
            labels[h_mid:, w_mid:] = 3
        elif num_blocks == 6:
            h_step = height // 2
            w_step = width // 3
            for i in range(2):
                for j in range(3):
                    start_h = i * h_step
                    end_h = height if i == 1 else (i + 1) * h_step
                    start_w = j * w_step
                    end_w = width if j == 2 else (j + 1) * w_step
                    labels[start_h:end_h, start_w:end_w] = i * 3 + j
        else:
            rows = int(np.sqrt(num_blocks))
            cols = (num_blocks + rows - 1) // rows
            h_step = height // rows
            w_step = width // cols
            for i in range(rows):
                for j in range(cols):
                    if i * cols + j < num_blocks:
                        start_h = i * h_step
                        end_h = height if i == rows - 1 else (i + 1) * h_step
                        start_w = j * w_step
                        end_w = width if j == cols - 1 else (j + 1) * w_step
                        labels[start_h:end_h, start_w:end_w] = i * cols + j
        return labels

    def _simple_grid_blocks(self, x, num_blocks, direction_features, texture_features):
        batch_size, channels, height, width = x.shape
        device = x.device
        blocks = []
        masks = []
        for b in range(batch_size):
            current_blocks = []
            current_masks = []
            rows = int(np.sqrt(num_blocks))
            cols = (num_blocks + rows - 1) // rows
            h_step = height // rows
            w_step = width // cols
            for i in range(rows):
                for j in range(cols):
                    if len(current_blocks) >= num_blocks:
                        break
                    start_h = i * h_step
                    end_h = height if i == rows - 1 else (i + 1) * h_step
                    start_w = j * w_step
                    end_w = width if j == cols - 1 else (j + 1) * w_step
                    mask = torch.zeros(1, 1, height, width, device=device)
                    mask[:, :, start_h:end_h, start_w:end_w] = 1.0
                    block_feature = x[b:b + 1] * mask
                    current_blocks.append(block_feature)
                    current_masks.append(mask)
            while len(current_blocks) < num_blocks:
                current_blocks.append(current_blocks[-1].clone())
                current_masks.append(current_masks[-1].clone())
            blocks.append(current_blocks)
            masks.append(current_masks)
        enhanced_features = x
        return (blocks, masks, direction_features, texture_features, enhanced_features)

class ImprovedPyramidAdaptiveFusion(nn.Module):

    def __init__(self, channels, num_levels=4, reduction=8):
        super(ImprovedPyramidAdaptiveFusion, self).__init__()
        self.channels = channels
        self.num_levels = num_levels
        self.reduction = reduction
        self.base_level_weights = nn.Parameter(torch.ones(num_levels) / num_levels)
        self.level_importance_evaluators = nn.ModuleList([nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(channels, max(channels // reduction, 1), 1), nn.PReLU(), nn.Conv2d(max(channels // reduction, 1), 1, 1), nn.Sigmoid()) for _ in range(num_levels)])
        self.inter_level_attention = nn.ModuleList([nn.Sequential(nn.Conv2d(channels, channels // 4, 1), nn.PReLU(), nn.Conv2d(channels // 4, 1, 1), nn.Sigmoid()) for _ in range(num_levels)])
        self.feature_refinement = nn.Sequential(nn.Conv2d(channels, channels, 3, padding=1), nn.PReLU(), DualResidualBlock(channels), nn.Conv2d(channels, channels, 1))
        self.temperature = nn.Parameter(torch.tensor(1.0))

    def forward(self, pyramid_features):
        if not pyramid_features:
            return (None, None)
        target_size = pyramid_features[0].shape[2:]
        batch_size = pyramid_features[0].shape[0]
        aligned_features = []
        for feat in pyramid_features[:self.num_levels]:
            if feat.shape[2:] != target_size:
                feat_aligned = F.interpolate(feat, size=target_size, mode='bilinear', align_corners=False)
            else:
                feat_aligned = feat
            aligned_features.append(feat_aligned)
        while len(aligned_features) < self.num_levels:
            aligned_features.append(aligned_features[-1].clone())
        importance_weights = []
        for i, feat in enumerate(aligned_features):
            importance = self.level_importance_evaluators[i](feat)
            importance_weights.append(importance)
        inter_level_weights = []
        for i, feat in enumerate(aligned_features):
            inter_weight = self.inter_level_attention[i](feat)
            inter_level_weights.append(inter_weight)
        base_weights = torch.softmax(self.base_level_weights / self.temperature, dim=0)
        final_weighted_features = []
        for i, (feat, importance, inter_weight) in enumerate(zip(aligned_features, importance_weights, inter_level_weights)):
            combined_weight = base_weights[i] * importance * inter_weight
            weighted_feat = feat * combined_weight
            final_weighted_features.append(weighted_feat)
        fused_feature = torch.stack(final_weighted_features, dim=0).sum(dim=0)
        refined_feature = self.feature_refinement(fused_feature)
        final_output = fused_feature + refined_feature
        weight_info = {'base_weights': base_weights.detach().cpu(), 'importance_weights': [w.detach().cpu() for w in importance_weights], 'inter_level_weights': [w.detach().cpu() for w in inter_level_weights], 'temperature': self.temperature.detach().cpu()}
        return (final_output, weight_info)

class EnhancedDualPyramidBackboneImproved(nn.Module):

    def __init__(self, channels=200, num_levels=4):
        super(EnhancedDualPyramidBackboneImproved, self).__init__()
        self.channels = channels
        self.num_levels = num_levels
        self.encoder_pyramid = nn.ModuleList()
        for i in range(num_levels):
            if i == 0:
                encoder_block = nn.Sequential(nn.Conv2d(channels, channels, 3, padding=1), nn.PReLU(), DualResidualBlock(channels), nn.Conv2d(channels, channels, 3, padding=1), nn.PReLU())
            else:
                encoder_block = nn.Sequential(nn.Conv2d(channels, channels, 3, stride=2, padding=1), nn.PReLU(), DualResidualBlock(channels), nn.Conv2d(channels, channels, 3, padding=1), nn.PReLU())
            self.encoder_pyramid.append(encoder_block)
        self.decoder_pyramid = nn.ModuleList()
        for i in range(num_levels - 1):
            decoder_block = nn.Sequential(nn.ConvTranspose2d(channels, channels, 4, stride=2, padding=1), nn.PReLU(), DualResidualBlock(channels), nn.Conv2d(channels, channels, 3, padding=1), nn.PReLU())
            self.decoder_pyramid.append(decoder_block)
        self.cross_pyramid_connections = nn.ModuleList()
        for i in range(num_levels - 1):
            connection = nn.Sequential(nn.Conv2d(channels * 2, channels, 1), nn.PReLU(), DualResidualBlock(channels), nn.Conv2d(channels, channels, 3, padding=1), nn.PReLU())
            self.cross_pyramid_connections.append(connection)
        self.encoder_fusion = ImprovedPyramidAdaptiveFusion(channels, num_levels)
        self.decoder_fusion = ImprovedPyramidAdaptiveFusion(channels, num_levels)
        self.final_fusion = nn.Sequential(nn.Conv2d(channels * 2, channels, 1), nn.PReLU(), DualResidualBlock(channels), nn.Conv2d(channels, channels, 3, padding=1), nn.PReLU(), nn.Conv2d(channels, channels, 3, padding=1))
        self.feature_enhancement = nn.Sequential(nn.Conv2d(channels, channels, 3, padding=1), nn.PReLU(), DualResidualBlock(channels), nn.Conv2d(channels, channels, 1))

    def forward(self, x):
        batch_size, channels, height, width = x.shape
        encoder_features = []
        current_feat = x
        for encoder_block in self.encoder_pyramid:
            current_feat = encoder_block(current_feat)
            encoder_features.append(current_feat)
        decoder_features = []
        current_feat = encoder_features[-1]
        decoder_features.append(current_feat)
        for i, decoder_block in enumerate(self.decoder_pyramid):
            current_feat = decoder_block(current_feat)
            if i < len(encoder_features) - 1:
                encoder_level = encoder_features[-(i + 2)]
                if current_feat.shape[2:] != encoder_level.shape[2:]:
                    current_feat = F.interpolate(current_feat, size=encoder_level.shape[2:], mode='bilinear', align_corners=False)
                combined = torch.cat([current_feat, encoder_level], dim=1)
                current_feat = self.cross_pyramid_connections[i](combined)
            decoder_features.append(current_feat)
        encoder_fused, encoder_weights = self.encoder_fusion(encoder_features)
        decoder_fused, decoder_weights = self.decoder_fusion(decoder_features)
        target_size = (height, width)
        if encoder_fused.shape[2:] != target_size:
            encoder_fused = F.interpolate(encoder_fused, size=target_size, mode='bilinear', align_corners=False)
        if decoder_fused.shape[2:] != target_size:
            decoder_fused = F.interpolate(decoder_fused, size=target_size, mode='bilinear', align_corners=False)
        final_combined = torch.cat([encoder_fused, decoder_fused], dim=1)
        final_output = self.final_fusion(final_combined)
        enhanced_output = self.feature_enhancement(final_output)
        final_output = final_output + enhanced_output
        return (final_output, {'encoder_weights': encoder_weights, 'decoder_weights': decoder_weights})

class Conv3x3(nn.Module):

    def __init__(self, in_dim, out_dim, kernel_size, stride, dilation=1):
        super(Conv3x3, self).__init__()
        reflect_padding = int(dilation * (kernel_size - 1) / 2)
        self.reflection_pad = nn.ReflectionPad2d(reflect_padding)
        self.conv2d = nn.Conv2d(in_dim, out_dim, kernel_size, stride, dilation=dilation, bias=False)

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        return out

class ColorCalibrationModule(nn.Module):

    def __init__(self, planes=31):
        super(ColorCalibrationModule, self).__init__()
        self.color_mapper = nn.Sequential(nn.Conv2d(planes, planes, 1), nn.PReLU(), nn.Conv2d(planes, planes, 1), nn.Sigmoid())
        self.color_bias = nn.Parameter(torch.zeros(1, planes, 1, 1))
        self.channel_consistency = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(planes, planes // 4, 1), nn.ReLU(), nn.Conv2d(planes // 4, planes, 1), nn.Sigmoid())
        self.calibration_strength = nn.Parameter(torch.tensor(0.1))

    def forward(self, x):
        mapped = self.color_mapper(x)
        calibrated = mapped + self.color_bias
        consistency_weight = self.channel_consistency(calibrated)
        adjusted = calibrated * consistency_weight
        output = x + self.calibration_strength * adjusted
        return output

class EnhancedPyramidDualResidualFusionWoodGrainHSI(nn.Module):

    def __init__(self, inplanes=3, planes=31, channels=200, num_pyramid_levels=4):
        super(EnhancedPyramidDualResidualFusionWoodGrainHSI, self).__init__()
        self.inplanes = inplanes
        self.planes = planes
        self.channels = channels
        self.num_pyramid_levels = num_pyramid_levels
        self.input_conv = Conv3x3(inplanes, channels, 3, 1)
        self.input_prelu = nn.PReLU()
        self.freq_processor = AdaptiveFrequencyLearning(channels, reduction=4)
        self.lbp_extractor = EnhancedLBPExtractor(radius=1, n_points=8, channels=channels)
        self.wood_analyzer = AdaptiveWoodGrainAnalyzer(channels)
        self.enhanced_dual_pyramid_backbone = EnhancedDualPyramidBackboneImproved(channels=channels, num_levels=num_pyramid_levels)
        self.output_conv = Conv3x3(channels, planes, 3, 1)
        self.long_residual = LongResidualConnection(inplanes, planes)
        self.color_calibration = ColorCalibrationModule(planes)
        self.final_enhancement = nn.Sequential(nn.Conv2d(planes, planes, kernel_size=3, padding=1), nn.PReLU(), DualResidualBlock(planes))

    def forward(self, x):
        original_input = x
        feat = self.input_prelu(self.input_conv(x))
        feat = self.freq_processor(feat)
        lbp_feat = self.lbp_extractor(feat)
        feat = feat + lbp_feat
        try:
            blocks, masks, direction_features, texture_features, enhanced_feat = self.wood_analyzer.create_adaptive_texture_blocks(feat, num_blocks=6)
            feat = enhanced_feat
        except Exception as e:
            pass
        feat, fusion_weights = self.enhanced_dual_pyramid_backbone(feat)
        output = self.output_conv(feat)
        output = self.long_residual(original_input, output)
        output = self.color_calibration(output)
        output = self.final_enhancement(output)
        return output

TGFSNet = EnhancedPyramidDualResidualFusionWoodGrainHSI

def build_model(**kwargs):
    return TGFSNet(**kwargs)
