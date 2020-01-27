import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.transform import rescale
from skimage.util import view_as_windows
from scipy.ndimage import uniform_filter


def dp_chain(g, f, m, a):
  # eqn 8
    '''
        g: unary costs with shape (H,W,D)
        f: pairwise costs with shape (H,W,D,D)
        m: messages with shape (H,W,D)
    '''
    # a = direction, added by Shalini

    H, W, D = g.shape
    if(a == 0):
      # for up to down a=0
      for hr in range(H):
        if hr == 0:
          m[hr, :, :] = np.zeros(m[hr, :, :].shape)
        else:
          for t in range(D):
            m[hr, :, t] = np.amin(m[hr - 1, :, :] + f[hr - 1,:, :, t] + g[hr - 1, :, :], axis=1) #Is the axis correct

    elif(a == 1):
      # for down to up a=1
      for hr in range(H - 1, 0, -1):
        if hr == H - 1:
          m[hr, :, :] = np.zeros(m[hr, :, :].shape)
        else:
          for t in range(D):
            m[hr, :, t] = np.amin(m[hr + 1, :, :] + f[hr + 1,:, :, t] + g[hr + 1, :, :], axis=1)

    elif(a == 2):
      # for left to right a=2
      for wc in range(W):
        if wc == 0:
          m[:, wc, :] = np.zeros(m[:, wc, :].shape)
        else:
          for t in range(D):
            m[:, wc, t] = np.amin(m[:, wc - 1, :] + f[:,wc - 1, :, t] + g[:, wc - 1, :], axis=1)

    elif(a == 3):
      # for right to left a=3
      for wc in range(W - 1, 0, -1):
        if wc == W - 1:
          m[:, wc, :] = np.zeros(m[:, wc, :].shape)
        else:
          for t in range(D):
            m[:, wc, t] = np.amin(m[:, wc + 1, :] + f[:,wc + 1, :, t] + g[:, wc + 1, :], axis=1)

    return m

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


def compute_mean(image, filter_size):
    mean_image = uniform_filter(image, filter_size, mode='constant')
    return mean_image


def get_pairwise_costs(H, W, D, weights=None):
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
    pairwise = np.ones((H, W, D, D))
    if(weights is None):
        L1 = 2
        L2 = 4
        dline = np.linspace(0, D, D)
        print(dline.shape)
        dxx, dyy = np.meshgrid(dline, dline)
        print(dxx.shape)
        print(dyy.shape)
        dgrid = np.abs(dxx - dyy)
        print(dgrid.shape)
        dgrid[(dgrid == 1)] = L1
        dgrid[(dgrid > 1)] = L2
        print(H, W)
        for h in range(H):
          for w in range(W):
            pairwise[h, w, :, :] = dgrid

        return pairwise

    return None


def compute_sgm(cv, f):
    # eq 9
    """
    Compute the SGM
    :param cv: cost volume of shape (H,W,D)
    :param f: Pairwise costs of shape (H,W,D,D)
    :return: pixel wise disparity map of shape (H,W)
    """
    # TODO
    m = np.zeros((cv.shape))
    H, W, D = cv.shape
    d = np.zeros((H, W))

    down = dp_chain(cv, f, m, 0)
    print('loaded down')
    up = dp_chain(cv, f, m, 1)
    print('loaded up')
    right = dp_chain(cv, f, m, 2)
    print('loaded right')
    left = dp_chain(cv, f, m, 3)
    print('loaded left')

    for h in range(H):
      for w in range(W):
        d[h, w] = np.argmin(cv[h, w, :] + down[h, w, :] + up[h, w, :] + right[h, w, :] + left[h, w, :], axis=0)

    return d

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

  return acc

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
    #cv = compute_cost_volume_sad(im0g, im1g, 64, 5)
    #cv = compute_cost_volume_ssd(im0g, im1g, 64, 5)
    #cv = compute_cost_volume_ncc(im0g, im1g, 64, 5)
    #np.save('cv_ssd', cv)
    cv = np.load('cv.npy')
    cv = -1*cv
    #cv = np.load('cv_sad.npy')
    print('loaded cv')
#    shimg = shift(im0g, 50, 0)

    #Compute winner takes all.
    disp_wta = np.argmin(cv, axis = 2)

    # Compute pairwise costs
    H, W, D = cv.shape
    #f = get_pairwise_costs(H, W, D)
    f = np.load('pairwise.npy')
    print('loaded pairwise')
    #np.save('pairwise', f)

    # Compute SGM
    disp = compute_sgm(cv, f)

    # Plot result
    plt.figure()
    plt.imshow(disp)
    plt.imshow(disp_wta)
    plt.show()


if __name__== "__main__":
    main()
