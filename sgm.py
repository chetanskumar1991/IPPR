import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.transform import rescale
from skimage.util import view_as_windows


def dp_chain(g, f, m):
  # for s in disparity array, get min value
    '''
        g: unary costs with shape (H,W,D), data term, in our case-cost volume
        f: pairwise costs with shape (H,W,D,D)
        m: messages with shape (H,W,D)
    '''
    # TODO
    return

def shift(im, dx, dy):
    n,m = im.shape
    bigim = np.zeros((3*n,3*m),im.dtype)
    bigim[n:2*n,m:2*m] = im
    x = n - dx
    y = m - dy
    return bigim[x : x + n, y : y + m]

def compute_cost_volume_sad(left_image, right_image, D, radius):
    """
    Sum of Absolute Differences (SAD) cost volume
    :param left_image: Left input image of size (H,W)
    :param right_image: Right input image of size (H,W)
    :param D: maximal disparity
    :param radius: Radius of the filter
    :return: cost volume of size (H,W,D)
    """
    H = left_image.shape[0]
    W = right_image.shape[1]
    cv_sad = np.zeros((H, W, D))

    for d in range(0, D):
        #Translate image by d, and compute similarity for that slice of cost volume.
        tr_right_image = shift(right_image, 0, d)
        diff_im = np.abs(left_image - tr_right_image) ##Absolute difference
        padded_diff_im = np.zeros((diff_im.shape[0] + 2 * radius , diff_im.shape[1] + 2 * radius))
        padded_diff_im[radius : radius + diff_im.shape[0], radius : radius + diff_im.shape[1]] = diff_im
        padded_diff_im_windows = view_as_windows(padded_diff_im, (radius * 2, radius * 2))

        padded_diff_im_windows = padded_diff_im_windows.reshape((padded_diff_im_windows.shape[0], padded_diff_im_windows.shape[1], -1))
        cv_sad[:, :, d] = np.sum(padded_diff_im_windows, axis = 2)[0 : H, 0: W]
        print("Computed SAD for disparity slice, ", d)

    return cv_sad

def compute_cost_volume_ssd(left_image, right_image, D, radius):
    """
    Sum of Squared Differences (SSD) cost volume
    :param left_image: Left input image of size (H,W)
    :param right_image: Right input image of size (H,W)
    :param D: maximal disparity
    :param radius: Radius of the filter
    :return: cost volume of size (H,W,D)
    """
    H = left_image.shape[0]
    W = right_image.shape[1]
    cv_ssd = np.zeros((H, W, D))

    for d in range(0, D):
        #Translate image by d, and compute similarity for that slice of cost volume.
        tr_right_image = shift(right_image, 0, d)
        diff_im = (left_image - tr_right_image) ** 2 #Squared difference.
        padded_diff_im = np.zeros((diff_im.shape[0] + 2 * radius , diff_im.shape[1] + 2 * radius))
        padded_diff_im[radius : radius + diff_im.shape[0], radius : radius + diff_im.shape[1]] = diff_im
        padded_diff_im_windows = view_as_windows(padded_diff_im, (radius * 2, radius * 2))

        padded_diff_im_windows = padded_diff_im_windows.reshape((padded_diff_im_windows.shape[0], padded_diff_im_windows.shape[1], -1))
        cv_ssd[:, :, d] = np.sum(padded_diff_im_windows, axis = 2)[0 : H, 0: W]
        print("Computed SSD for disparity slice, ", d)

    return cv_ssd


def compute_cost_volume_ncc(left_image, right_image, D, radius):
    """
    Normalized Cross Correlation (NCC) cost volume
    :param left_image: Left input image of size (H,W)
    :param right_image: Right input image of size (H,W)
    :param D: maximal disparity
    :param radius: Radius of the filter
    :return: cost volume of size (H,W,D)
    """
    H = left_image.shape[0]
    W = right_image.shape[1]
    cv_ncc = np.zeros((H, W, D))

    padded_left_image = np.zeros((left_image.shape[0] + 2 * radius , left_image.shape[1] + 2 * radius))
    padded_left_image[radius : radius + left_image.shape[0], radius : radius + left_image.shape[1]] = left_image
    padded_left_image_windows = view_as_windows(padded_left_image, (radius * 2, radius * 2))
    padded_left_image_windows = padded_left_image_windows.reshape((padded_left_image_windows.shape[0], padded_left_image_windows.shape[1], -1))
    pv = padded_left_image_windows
    p_mean = np.mean(pv, axis = 2)
    p_mean_3d = np.repeat(p_mean[:, :, np.newaxis], pv.shape[2], axis = 2)
    p = pv - p_mean_3d

    for d in range(0, D):
        #Translate image by d, and compute similarity for that slice of cost volume.
        tr_right_image = shift(right_image, 0, d)
        padded_tr_right_image = np.zeros((tr_right_image.shape[0] + 2 * radius , tr_right_image.shape[1] + 2 * radius))
        padded_tr_right_image[radius : radius + tr_right_image.shape[0], radius : radius + tr_right_image.shape[1]] = tr_right_image
        padded_tr_right_image_windows = view_as_windows(padded_tr_right_image, (radius * 2, radius * 2))
        padded_tr_right_image_windows = padded_tr_right_image_windows.reshape((padded_tr_right_image_windows.shape[0], padded_tr_right_image_windows.shape[1], -1))
        qv = padded_tr_right_image_windows

        q_mean = np.mean(qv, axis = 2)
        q_mean_3d = np.repeat(q_mean[:, :, np.newaxis], qv.shape[2], axis = 2)
        q = qv - q_mean_3d

        pq = p * q
        p2q2 = (p ** 2) * (q ** 2)

        cv_ncc[:, :, d] = (np.sum(pq, axis = 2) / np.sqrt(np.sum(p2q2, axis = 2)))[0: H, 0: W]

        print("Computed NCC for disparity slice, ", d)

    return cv_ncc

def calculate_traversal_cost(L1, L2, dist):
    if dist == 0:
        return 0
    elif dist == 1:
        return L1
    else:
        return L2

def get_pairwise_costs(H, W, D, weights=None):
    L1 = 1
    L2 = 10
    disparity_size = len(D)
    pairwise_costs = np.ones((H, W, disparity_size, disparity_size))


    for h in range(H): # row-wise
      for w in range(W): # compare prev neighbour column pixel
        for d2 in range(D): # current disparity iteration
          for d1 in range(D): # prev pixel disparity iteration
            if weights==None:
              pairwise_costs[h][w][d2][d1] =  calculate_traversal_cost(L1, L2, abs(d1 - d2))
            else:
              pairwise_costs[h][w][d2][d1] =  weights[d2][d1]*calculate_traversal_cost(L1, L2, abs(d1 - d2))
    """
    :param H: height of input image
    :param W: width of input image
    :param D: maximal disparity
    :param weights: edge-dependent weights (necessary to implement the bonus task)
    :return: pairwise_costs of shape (H,W,D,D)
             Note: If weight=None, then each spatial position gets exactly the same pairwise costs.
             In this case the array of shape (D,D) can be broadcasted to (H,W,D,D) by using np.broadcast_to(..).
    """
    # TODO

    return pairwise_costs


def compute_sgm(cv, f):
    """
    Compute the SGM
    :param cv: cost volume of shape (H,W,D)
    :param f: Pairwise costs of shape (H,W,D,D)
    :return: pixel wise disparity map of shape (H,W)
    """
    Hr, Wc, D = cv.shape
    print(Hr, Wc, D)
    m = np.ones((Hr, Wc, D))
    b = np.ones((Hr, Wc, D))
    optimal_path = np.ones((D, Hr, Wc))
    resultant_image = np.ones((Hr, Wc))
    directions = ['l', 'r', 'u', 'd']

    for h in range(Hr): # row-wise
      for w in range(Wc): # compare prev neighbour column pixel
        for t in range(D): # current disparity iteration
          m_i_s = [0]
          for s in range(D): # prev pixel disparity iteration
            sum_of_messages = 0
            if w == 0 or h==0: # first column only has base energy
              # TODO: Add the edge cases
              x = 0
            else:
              for a in directions:
                if a == 'l':
                  he = h
                  we = w - 1
                elif a == 'r':
                  he = h
                  we = w + 1
                elif a == 'u':
                  he = h + 1
                  we = w
                elif a == 'd':
                  he = h + 1
                  we = w

                sum_of_messages = sum_of_messages + m[h][w][s][a]

                # TODO: How to calculate messages in node i
              b[h][w][s] = cv[h][w][s] + sum_of_messages
              m_i_s = m_i_s.append[m[h][w][s][a] + f[h][w][s][t] + cv[h][w][s]]
            m[he][we][t][a] = min(m_i_s)
      resultant_image[h][w] = b[h][w].argmin()

    # TODO
    return resultant_image

# TODO: Check if accX is correct

def calculate_accX(computed_disparities, ground_truth_disparities, occlusion_mask, X):
  H, W = computed_disparities.shape
  counter = 0
  Z = 0

  for i in range(computed_disparities.shape[0]):
    for j in range(computed_disparities.shape[1]):
      if occlusion_mask[i, j] > 0:
        Z = Z + 1
        if abs(computed_disparities[i, j] - ground_truth_disparities[i,j]) <= X:
          counter = counter + 1


  acc = counter/Z

def main():
    # Load input images
    im0 = imread("Images/Adirondack_left.png")
    im1 = imread("Images/Adirondack_right.png")

    im0g = rgb2gray(im0)
    im1g = rgb2gray(im1)

    # Plot input images
    plt.figure(figsize=(8,4))
    plt.subplot(121), plt.imshow(im0g, cmap='gray'), plt.title('Left')
    plt.subplot(122), plt.imshow(im1g, cmap='gray'), plt.title('Right')
    plt.tight_layout()

    # Use either SAD, NCC or SSD to compute the cost volume
#    cv = compute_cost_volume_sad(im0g, im1g, 64, 5)
#    cv = compute_cost_volume_ssd(im0g, im1g, 64, 5)
    cv = compute_cost_volume_ncc(im0g, im1g, 64, 5)

#    shimg = shift(im0g, 50, 0)

    #Compute winner takes all.
    disp_wta = np.argmin(cv, axis = 2)

    # Compute pairwise costs
    H, W, D = cv.shape
    f = get_pairwise_costs(H, W, D)

    # Compute SGM
    disp = compute_sgm(cv, f)

    # Plot result
    plt.figure()
    plt.imshow(disp_wta)
    plt.show()


if __name__== "__main__":
    main()
