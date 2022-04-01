"""Stereo matching."""
import numpy as np
from scipy.signal import convolve2d



class Solution:
    def __init__(self):
        pass

    @staticmethod
    def ssd_distance(left_image: np.ndarray,
                     right_image: np.ndarray,
                     win_size: int,
                     dsp_range: int) -> np.ndarray:
        """Compute the SSDD distances tensor.

        Args:
            left_image: Left image of shape: HxWx3, and type np.double64.
            right_image: Right image of shape: HxWx3, and type np.double64.
            win_size: Window size odd integer.
            dsp_range: Half of the disparity range. The actual range is
            -dsp_range, -dsp_range + 1, ..., 0, 1, ..., dsp_range.

        Returns:
            A tensor of the sum of squared differences for every pixel in a
            window of size win_size X win_size, for the 2*dsp_range + 1
            possible disparity values. The tensor shape should be:
            HxWx(2*dsp_range+1).
        """
        print('ssd_distance')
        num_of_rows, num_of_cols = left_image.shape[0], left_image.shape[1]
        disparity_values = range(-dsp_range, dsp_range + 1)
        ssdd_tensor = np.zeros((num_of_rows,
                                num_of_cols,
                                len(disparity_values)))

        right_lr_pad = int(np.abs(np.max(disparity_values)) + np.floor(win_size / 2))
        left_lrud_pad = int(np.floor(win_size / 2))
        right_image_pad = np.pad(right_image, ((left_lrud_pad, left_lrud_pad), (right_lr_pad, right_lr_pad), (0, 0)))
        left_image_pad = np.pad(left_image, ((left_lrud_pad, left_lrud_pad), (left_lrud_pad, left_lrud_pad), (0, 0)))

        for row_indx in range(left_lrud_pad, left_image_pad.shape[0] - left_lrud_pad):
            for col_indx in range(left_lrud_pad, left_image_pad.shape[1] - left_lrud_pad):
                value_left_image = left_image_pad[row_indx - left_lrud_pad:row_indx + left_lrud_pad + 1,
                                   col_indx - left_lrud_pad:col_indx + left_lrud_pad + 1, :]
                dis_indx = 0
                for dis_val in disparity_values:
                    col_in_right_image = col_indx + dsp_range + dis_val;
                    value_right_image = right_image_pad[row_indx - left_lrud_pad:row_indx + left_lrud_pad + 1,
                                        col_in_right_image - left_lrud_pad:col_in_right_image + left_lrud_pad + 1, :]
                    ssdd_tensor[row_indx - left_lrud_pad, col_indx - left_lrud_pad, dis_indx] = np.sum(
                        (value_left_image - value_right_image) ** 2)
                    dis_indx = dis_indx + 1

        ssdd_tensor -= ssdd_tensor.min()
        ssdd_tensor /= ssdd_tensor.max()
        ssdd_tensor *= 255.0
        return ssdd_tensor

    @staticmethod
    def naive_labeling(ssdd_tensor: np.ndarray) -> np.ndarray:
        """Estimate a naive depth estimation from the SSDD tensor.

        Args:
            ssdd_tensor: A tensor of the sum of squared differences for every
            pixel in a window of size win_size X win_size, for the
            2*dsp_range + 1 possible disparity values.

        Evaluate the labels in a naive approach. Each value in the
        result tensor should contain the disparity matching minimal ssd (sum of
        squared difference).

        Returns:
            Naive labels HxW matrix.
        """
        # you can erase the label_no_smooth initialization.
        # label_no_smooth = np.zeros((ssdd_tensor.shape[0], ssdd_tensor.shape[1]))

        label_no_smooth = np.argmin(ssdd_tensor, axis=2)

        return label_no_smooth

    @staticmethod
    def dp_grade_slice(c_slice: np.ndarray, p1: float, p2: float) -> np.ndarray:
        """Calculate the scores matrix for slice c_slice.

        Calculate the scores slice which for each column and disparity value
        states the score of the best route. The scores slice is of shape:
        (2*dsp_range + 1)xW.

        Args:
            c_slice: A slice of the ssdd tensor.
            p1: penalty for taking disparity value with 1 offset.
            p2: penalty for taking disparity value more than 2 offset.
        Returns:
            Scores slice which for each column and disparity value states the
            score of the best route.
        """
        c_slice = c_slice.T
        num_labels, num_of_cols = c_slice.shape[0], c_slice.shape[1]
        l_slice = np.zeros((num_labels, num_of_cols))


        l_slice[:, 0] = c_slice[:, 0]
        for col_slice in range(1, num_of_cols):
            for d_slice in range(0, num_labels):
                if d_slice - 2 < 0:
                    Mp2 = np.min(l_slice[d_slice + 2:num_labels, col_slice - 1])
                else:
                    Mp2 = np.min(np.hstack(
                        (l_slice[d_slice + 2:num_labels, col_slice - 1], l_slice[0:d_slice - 2, col_slice - 1])))
                if d_slice + 1 > num_labels - 1:
                    ld_next = np.inf
                else:
                    ld_next = l_slice[d_slice + 1, col_slice - 1]
                if d_slice - 1 < 0:
                    ld_prev = np.inf
                else:
                    ld_prev = l_slice[d_slice - 1, col_slice - 1]
                M = np.min((l_slice[d_slice, col_slice - 1], p1 + np.min((ld_prev, ld_next)), p2 + Mp2))
                l_slice[d_slice, col_slice] = c_slice[d_slice, col_slice] + M - np.min(l_slice[:, col_slice - 1])

        return l_slice

    def dp_labeling(self,
                    ssdd_tensor: np.ndarray,
                    p1: float,
                    p2: float) -> np.ndarray:
        """Estimate a depth map using Dynamic Programming.

        (1) Call dp_grade_slice on each row slice of the ssdd tensor.
        (2) Store each slice in a corresponding l tensor (of shape as ssdd).
        (3) Finally, for each pixel in l (along each row and column), choose
        the best disparity value. That is the disparity value which
        corresponds to the lowest l value in that pixel.

        Args:
            ssdd_tensor: A tensor of the sum of squared differences for every
            pixel in a window of size win_size X win_size, for the
            2*dsp_range + 1 possible disparity values.
            p1: penalty for taking disparity value with 1 offset.
            p2: penalty for taking disparity value more than 2 offset.
        Returns:
            Dynamic Programming depth estimation matrix of shape HxW.
        """
        l = np.zeros_like(ssdd_tensor)

        for c_silce_indx in range(0, ssdd_tensor.shape[0]):
            c_slice = ssdd_tensor[c_silce_indx, :, :]
            l_slice = self.dp_grade_slice(c_slice, p1, p2)
            l[c_silce_indx, :, :] = l_slice.T

        return self.naive_labeling(l)
    
    @staticmethod
    def slices_per_dir(ssdd_tensor, direction):
        number_of_direction = 8
        start = -(ssdd_tensor.shape[0] - 1)
        end = ssdd_tensor.shape[1] - 1
        dict_dire = {}
        dict_dire_indices = {}
        indices_mat = np.arange(0, (ssdd_tensor.shape[0]) * (ssdd_tensor.shape[1])).reshape(ssdd_tensor.shape[0],
                                                                                            ssdd_tensor.shape[1])
        dire = direction % 4
        if (dire == 1):
            'east or west'
            dict_dire[str(dire)] = []  # east
            dict_dire[str(dire + 4)] = []  # west
            dict_dire_indices[str(dire)] = []  # east
            dict_dire_indices[str(dire + 4)] = []  # west
            for x in range (ssdd_tensor.shape[0]):
                dict_dire[str(dire)].append(ssdd_tensor[x])
                dict_dire_indices[str(dire)].append(indices_mat[x])
                dict_dire[str(dire + 4)].append((np.fliplr(ssdd_tensor))[x])
                dict_dire_indices[str(dire + 4)].append((np.fliplr(indices_mat))[x])

        if (dire == 3):
            'south or north'
            dict_dire[str(dire)] = []  # south
            dict_dire[str(dire + 4)] = []  # north
            dict_dire_indices[str(dire)] = []  # south
            dict_dire_indices[str(dire + 4)] = []  # north
            for y in range (ssdd_tensor.shape[1]):
                new_tensor = np.rot90(ssdd_tensor)
                new_indices_mat = np.rot90(indices_mat)
                dict_dire[str(dire)].append(new_tensor[y])
                dict_dire_indices[str(dire)].append(new_indices_mat[y])
                dict_dire[str(dire + 4)].append(np.fliplr(new_tensor)[y])
                dict_dire_indices[str(dire + 4)].append(np.fliplr(new_indices_mat)[y])


        if (dire == 2):
            'south east or north west'
            dict_dire[str(dire)] = []  # south east
            dict_dire[str(dire + 4)] = []  # north west
            dict_dire_indices[str(dire)] = []  # south east
            dict_dire_indices[str(dire + 4)] = []  # north west
            for offset in range(start, end):
                dict_dire[str(dire)].append(ssdd_tensor.diagonal(offset=offset))
                dict_dire_indices[str(dire)].append(indices_mat.diagonal(offset=offset))
                dict_dire[str(dire + 4)].append(np.fliplr(ssdd_tensor.diagonal(offset=offset)))
                imat =indices_mat.diagonal(offset=offset)
                dict_dire_indices[str(dire + 4)].append(np.fliplr(imat.reshape((1,len(imat)))))

        if (dire == 0):
           'south west or north east'
           dict_dire[str(dire + 4)] = []  # south west
           dict_dire[str(dire + 8)] = []  # north east
           dict_dire_indices[str(dire + 4)] = []  # south west
           dict_dire_indices[str(dire + 8)] = []  # north east
           for offset in range(start, end):
               new_tensor = np.fliplr(ssdd_tensor)
               new_indices_mat = np.fliplr(indices_mat)
               dict_dire[str(dire + 4)].append((new_tensor.diagonal(offset=offset)))
               dict_dire_indices[str(dire + 4)].append(new_indices_mat.diagonal(offset=offset))
               dict_dire[str(dire + 8)].append(np.fliplr(new_tensor.diagonal(offset)))
               imat = new_indices_mat.diagonal(offset)
               dict_dire_indices[str(dire + 8)].append(np.fliplr(imat.reshape((1,len(imat)))))

        return dict_dire[str(direction)], dict_dire_indices[str(direction)], dict_dire[str(direction + 4)], dict_dire_indices[str(direction + 4)]
    
    
    def dp_labeling_per_direction(self,
            ssdd_tensor: np.ndarray,
            p1: float,
            p2: float) -> dict:
        """Return a dictionary of directions to a Dynamic Programming
        etimation of depth.
    
        For each direction in 1, ..., 8, calculate scores tensors
        according to dp_grade_slice and the method which allows you to
        extract slices along each direction.
    
        You may use helper methods (functions) that you write on your own.
        We found `np.diagonal` to be very helpful to extract diagonal slices.
        `np.unravel_index` might be helpful if you're thinking in MATLAB
        notations: it's the ind2sub equivalent.
    
        Args:
            ssdd_tensor: A tensor of the sum of squared differences for
            every pixel in a window of size win_size X win_size, for the
            2*dsp_range + 1 possible disparity values.
            p1: penalty for taking disparity value with 1 offset.
            p2: penalty for taking disparity value more than 2 offset.
    
        Returns:
            Dictionary int->np.ndarray which maps each direction to the
            corresponding dynamic programming estimation of depth based on
            that direction.
        """
        num_of_directions = 8
        l = np.zeros_like(ssdd_tensor)
        opp_l = np.zeros_like(ssdd_tensor)
        direction_to_slice = {}

    
        for direction in range(1, int(num_of_directions / 2) + 1):
            cur_slices, cur_indices, opp_cur_slices, opp_cur_indices = self.slices_per_dir(ssdd_tensor, direction)
                
            dim = len(cur_slices)
            for c_silce_indx in range(0, dim):
                c_slice = cur_slices[c_silce_indx]
                if c_slice.shape[0] == 41:
                    c_slice = c_slice.T
                if np.shape(c_slice) == (1,41):
                    l_slice = c_slice.T
                else:
                    l_slice = self.dp_grade_slice(c_slice, p1, p2)
                raw, col = np.unravel_index(cur_indices[c_silce_indx], (ssdd_tensor.shape[0], ssdd_tensor.shape[1]))
                l[raw,col,:] = l_slice.T


    
                opp_c_slice = opp_cur_slices[c_silce_indx]
                if opp_c_slice.shape[0] == 41:
                    opp_c_slice = opp_c_slice.T
                if np.shape(opp_c_slice) == (1,41):
                    opp_l_slice = opp_c_slice.T
                else:
                    opp_l_slice = self.dp_grade_slice(opp_c_slice, p1, p2)

                opp_raw, opp_col = np.unravel_index(opp_cur_indices[c_silce_indx], (ssdd_tensor.shape[0], ssdd_tensor.shape[1]))
                opp_l[opp_raw,opp_col,:] = opp_l_slice.T

            direction_to_slice[direction] = self.naive_labeling(l)
            direction_to_slice[direction + 4] = self.naive_labeling(opp_l)
    
        return direction_to_slice
    
    
    def sgm_labeling(self, ssdd_tensor: np.ndarray, p1: float, p2: float):
    
        """Estimate the depth map according to the SGM algorithm.
    
        For each direction in 1, ..., 8, calculate scores tensors
        according to dp_grade_slice and the method which allows you to
        extract slices along each direction.
    
        You may use helper methods (functions) that you write on your own.
        We found `np.diagonal` to be very helpful to extract diagonal slices.
        `np.unravel_index` might be helpful if you're thinking in MATLAB
        notations: it's the ind2sub equivalent.
    
        Args:
            ssdd_tensor: A tensor of the sum of squared differences for
            every pixel in a window of size win_size X win_size, for the
            2*dsp_range + 1 possible disparity values.
            p1: penalty for taking disparity value with 1 offset.
            p2: penalty for taking disparity value more than 2 offset.
    
        Returns:
            Semi-Global Mapping depth estimation matrix of shape HxW.
        """
       
        num_of_directions = 8
        l = np.zeros_like(ssdd_tensor)
        opp_l = np.zeros_like(ssdd_tensor)
        l_final = np.zeros_like(ssdd_tensor)
        direction_to_slice = {}
        direction_to_vote = {}

    
        for direction in range(1, int(num_of_directions / 2) + 1):
            cur_slices, cur_indices, opp_cur_slices, opp_cur_indices = self.slices_per_dir(ssdd_tensor, direction)
                
            dim = len(cur_slices)
            for c_silce_indx in range(0, dim):
                c_slice = cur_slices[c_silce_indx]
                if c_slice.shape[0] == 41:
                    c_slice = c_slice.T
                if np.shape(c_slice) == (1,41):
                    l_slice = c_slice.T
                else:
                    l_slice = self.dp_grade_slice(c_slice, p1, p2)
                raw, col = np.unravel_index(cur_indices[c_silce_indx], (ssdd_tensor.shape[0], ssdd_tensor.shape[1]))
                l[raw,col,:] = l_slice.T


    
                opp_c_slice = opp_cur_slices[c_silce_indx]
                if opp_c_slice.shape[0] == 41:
                    opp_c_slice = opp_c_slice.T
                if np.shape(opp_c_slice) == (1,41):
                    opp_l_slice = opp_c_slice.T
                else:
                    opp_l_slice = self.dp_grade_slice(opp_c_slice, p1, p2)

                opp_raw, opp_col = np.unravel_index(opp_cur_indices[c_silce_indx], (ssdd_tensor.shape[0], ssdd_tensor.shape[1]))
                opp_l[opp_raw,opp_col,:] = opp_l_slice.T
  
            l_final = l_final + l + opp_l

        return self.naive_labeling(l_final/8)

    
    @staticmethod
    def ncc_distance(left_image: np.ndarray,
                     right_image: np.ndarray,
                     win_size: int,
                     dsp_range: int) -> np.ndarray:        
        
        print('ncc_distance')
        num_of_rows, num_of_cols = left_image.shape[0], left_image.shape[1]
        disparity_values = range(-dsp_range, dsp_range+1)
        nccd_tensor = np.zeros((num_of_rows,
                                num_of_cols,
                                len(disparity_values)))
        right_lr_pad = int(np.abs(np.max(disparity_values))+np.floor(win_size/2))
        left_lrud_pad = int(np.floor(win_size/2))
        right_image_pad = np.pad(right_image,((left_lrud_pad,left_lrud_pad),(right_lr_pad,right_lr_pad),(0,0)))
        left_image_pad = np.pad(left_image,((left_lrud_pad,left_lrud_pad),(left_lrud_pad,left_lrud_pad),(0,0)))
        
        for row_indx in range(left_lrud_pad,left_image_pad.shape[0]-left_lrud_pad):
            for col_indx in range(left_lrud_pad,left_image_pad.shape[1]-left_lrud_pad):
                value_left_image = left_image_pad[row_indx-left_lrud_pad:row_indx+left_lrud_pad+1,col_indx-left_lrud_pad:col_indx+left_lrud_pad+1,:]
                dis_indx  = 0
                for dis_val in disparity_values:
                    col_in_right_image = col_indx + dsp_range + dis_val;
                    value_right_image = right_image_pad[row_indx-left_lrud_pad:row_indx+left_lrud_pad+1, col_in_right_image-left_lrud_pad:col_in_right_image+left_lrud_pad+1 ,:]
                    N = len(value_right_image.flatten())
                    mean_left = np.mean(value_left_image)
                    mean_right = np.mean(value_right_image)
                    var_left = np.sum((value_left_image - mean_left)**2)/N
                    var_right = np.sum((value_right_image - mean_right)**2)/N
                    nccd_tensor[row_indx-left_lrud_pad,col_indx-left_lrud_pad,dis_indx] = 1 - np.sum((value_left_image - mean_left) * (value_right_image - mean_right)) / (N * np.sqrt(var_left*var_right + np.finfo(float).eps)) # NCC
                    dis_indx = dis_indx + 1
        
        
        nccd_tensor -= nccd_tensor.min()
        nccd_tensor /= nccd_tensor.max()
        nccd_tensor *= 255.0
        return nccd_tensor
        
    
    
    
    
    
    
    
    

