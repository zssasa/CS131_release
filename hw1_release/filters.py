import numpy as np


def conv_nested(image, kernel):
    """A naive implementation of convolution filter.

    This is a naive implementation of convolution using 4 nested for-loops.
    This function computes convolution of an image with a kernel and outputs
    the result that has the same shape as the input image.

    Args:
        image: numpy array of shape (Hi, Wi)
        kernel: numpy array of shape (Hk, Wk)

    Returns:
        out: numpy array of shape (Hi, Wi)
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))

    H = Hk // 2
    W = Wk // 2
    for i1 in range(Hi):
        for j1 in range(Wi):
            for i2 in range(Hk):
                for j2 in range(Wk):
                    i = i2 - H
                    j = j2 - W
                    if i1-i<0 or j1-j<0 or i1-i>=Hi or j1-j>=Wi:
                        continue
                    out[i1, j1] += kernel[i2, j2]*image[i1-i, j1-j]
    return out

def zero_pad(image, pad_height, pad_width):
    """ Zero-pad an image.

    Ex: a 1x1 image [[1]] with pad_height = 1, pad_width = 2 becomes:

        [[0, 0, 0, 0, 0],
         [0, 0, 1, 0, 0],
         [0, 0, 0, 0, 0]]         of shape (3, 5)

    Args:
        image: numpy array of shape (H, W)
        pad_width: width of the zero padding (left and right padding)
        pad_height: height of the zero padding (bottom and top padding)

    Returns:
        out: numpy array of shape (H+2*pad_height, W+2*pad_width)
    """

    H, W = image.shape
    out = None
    out = np.zeros((H+2*pad_height, W+2*pad_width))
    out[pad_height:H+pad_height, pad_width:W+pad_width] = image
    return out


def conv_fast(image, kernel):
    """ An efficient implementation of convolution filter.

    This function uses element-wise multiplication and np.sum()
    to efficiently compute weighted sum of neighborhood at each
    pixel.

    Hints:
        - Use the zero_pad function you implemented above
        - There should be two nested for-loops
        - You may find np.flip() and np.sum() useful

    Args:
        image: numpy array of shape (Hi, Wi)
        kernel: numpy array of shape (Hk, Wk)

    Returns:
        out: numpy array of shape (Hi, Wi)
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))

    k = np.flip(np.flip(kernel, 1), 0)
    padding_image = zero_pad(image, Hk//2, Wk//2)
    for i in range(Hi):
        for j in range(Wi):
            out[i, j] = np.sum(np.multiply(padding_image[i:i+Hk, j:j+Wk], k))

    return out

def conv_faster(image, kernel):
    """
    Args:
        image: numpy array of shape (Hi, Wi)
        kernel: numpy array of shape (Hk, Wk)

    Returns:
        out: numpy array of shape (Hi, Wi)
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))

    k = np.flip(np.flip(kernel, 1), 0)
    padding_image = zero_pad(image, Hk // 2, Wk // 2)
    image_transform = np.zeros((Hi*Wi, Hk*Wk))
    for i in range(Hi):
        for j in range(Wi):
            image_transform[i*Wi+j,:] = padding_image[i:i+Hk, j:j+Wk].reshape(1, -1)
    out = image_transform.dot(k.reshape(-1, 1))
    out = out.reshape(Hi, Wi)
    return out

def cross_correlation(f, g):
    """ Cross-correlation of f and g

    Hint: use the conv_fast function defined above.

    Args:
        f: numpy array of shape (Hf, Wf)
        g: numpy array of shape (Hg, Wg)

    Returns:
        out: numpy array of shape (Hf, Wf)
    """

    out = None
    k = np.flip(np.flip(g, 0), 1)
    out = conv_fast(f, k)

    return out

def zero_mean_cross_correlation(f, g):
    """ Zero-mean cross-correlation of f and g

    Subtract the mean of g from g so that its mean becomes zero

    Args:
        f: numpy array of shape (Hf, Wf)
        g: numpy array of shape (Hg, Wg)

    Returns:
        out: numpy array of shape (Hf, Wf)
    """

    out = None
    g = g - np.mean(g)
    k = np.flip(np.flip(g, 0), 1)
    out = conv_fast(f, k)

    return out

def normalized_cross_correlation(f, g):
    """ Normalized cross-correlation of f and g

    Normalize the subimage of f and the template g at each step
    before computing the weighted sum of the two.

    Args:
        f: numpy array of shape (Hf, Wf)
        g: numpy array of shape (Hg, Wg)

    Returns:
        out: numpy array of shape (Hf, Wf)
    """

    out = None
    Hi, Wi = f.shape
    Hk, Wk = g.shape
    out = np.zeros((Hi, Wi))

    g = (g - np.mean(g)) / np.std(g)
    # padding_image = zero_pad(f, Hk // 2, Wk // 2)
    # for i in range(Hi):
    #     for j in range(Wi):
    #         tmp = padding_image[i:i + Hk, j:j + Wk]
    #         tmp = (tmp-np.mean(tmp))/np.std(tmp)
    #         out[i, j] = np.sum(np.multiply(tmp, g))

    padding_image = zero_pad(f, Hk // 2, Wk // 2)
    image_transform = np.zeros((Hi * Wi, Hk * Wk))
    for i in range(Hi):
        for j in range(Wi):
            tmp = padding_image[i:i + Hk, j:j + Wk]
            tmp = (tmp - np.mean(tmp)) / np.std(tmp)
            image_transform[i * Wi + j, :] = tmp.reshape(1, -1)
    out = image_transform.dot(g.reshape(-1, 1))
    out = out.reshape(Hi, Wi)

    return out
