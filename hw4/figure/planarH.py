import numpy as np
import cv2


def computeH(x1, x2):
    # Q2.2.1
    # Compute the homography between two sets of points
    A = np.empty(shape=(0, 9))
    N = len(x1)
    for i in range(N):
        row1 = np.array([x2[i, 0], x2[i, 1], 1, 0, 0, 0, -x1[i, 0] * x2[i, 0], -x1[i, 0] * x2[i, 1], -x1[i, 0]])
        row2 = np.array([0, 0, 0, x2[i, 0], x2[i, 1], 1, -x2[i, 0] * x1[i, 1], -x1[i, 1] * x2[i, 1], -x1[i, 1]])
        A = np.vstack((A, row1))
        A = np.vstack((A, row2))

    u, s, vh = np.linalg.svd(A)
    H2to1 = np.reshape(vh[8], (3, 3))

    return H2to1


def computeH_norm(x1, x2):
    # Q2.2.2
    # Compute the centroid of the points
    centroid_x1 = np.mean(x1, axis=0)
    centroid_x2 = np.mean(x2, axis=0)

    # Shift the origin of the points to the centroid
    shift_x1 = x1 - centroid_x1
    shift_x2 = x2 - centroid_x2

    # Normalize the points so that the largest distance from the origin is equal to sqrt(2)
    dist_x1 = np.linalg.norm(shift_x1, axis=1)
    dist_x2 = np.linalg.norm(shift_x2, axis=1)
    max_dist_x1 = np.max(dist_x1)
    max_dist_x2 = np.max(dist_x2)
    scale1 = np.sqrt(2) / max_dist_x1
    scale2 = np.sqrt(2) / max_dist_x2
    norm_x1 = shift_x1 * scale1
    norm_x2 = shift_x2 * scale2

    # Similarity transform 1
    T1 = np.array(([[scale1, 0, -centroid_x1[0] * scale1],
                    [0, scale1, -centroid_x1[1] * scale1],
                    [0, 0, 1]]))

    # Similarity transform 2
    T2 = np.array(([[scale2, 0, -centroid_x2[0] * scale2],
                    [0, scale2, -centroid_x2[1] * scale2],
                    [0, 0, 1]]))

    # Compute homography
    H = computeH(norm_x1, norm_x2)

    # Denormalization
    H2to1 = np.linalg.inv(T1) @ H @ T2

    return H2to1


def computeH_ransac(locs1, locs2, opts):
    # Q2.2.3
    # Compute the best fitting homography given a list of matching points
    max_iters = opts.max_iters  # the number of iterations to run RANSAC for
    inlier_tol = opts.inlier_tol  # the tolerance value for considering a point to be an inlier

    # initialization variables
    N = len(locs1)
    best_counts = 0
    bestH2to1 = []
    inliers = []

    for i in range(max_iters):
        counts = 0
        inliers_iters = []

        assert len(locs1) >= 4
        assert len(locs1) == len(locs2)

        # generate 4 pairs of random points from locs1 and locs2
        indexes = np.random.choice(len(locs1), 4, replace=False)
        locs1_1 = locs1[indexes[0]]
        locs1_2 = locs1[indexes[1]]
        locs1_3 = locs1[indexes[2]]
        locs1_4 = locs1[indexes[3]]
        locs2_1 = locs2[indexes[0]]
        locs2_2 = locs2[indexes[1]]
        locs2_3 = locs2[indexes[2]]
        locs2_4 = locs2[indexes[3]]

        # put the points into array for compute H norm
        x1 = np.array([locs1_1, locs1_2, locs1_3, locs1_4])
        x2 = np.array([locs2_1, locs2_2, locs2_3, locs2_4])
        H = computeH_norm(x1, x2)

        # check the computed H through the all data points
        for j in range(N):
            point1 = np.dot(H, np.append(locs2[j], 1))  # make it (x2, y2, 1) homogeneous
            point1 = np.array([point1[0] / point1[2], point1[1] / point1[2]])  # make it (x2, y2) homogeneous
            point2 = np.array(locs1[j])  # (x1, y1)

            # calculating Euclidean distance between the two points
            error = np.linalg.norm(point1 - point2)

            # compare error with the inlier tolerance
            if error <= inlier_tol:
                counts = counts + 1
                inliers_iters.append(1)
            else:
                inliers_iters.append(0)

        # after 1 iteration, check the results is best or not
        if counts > best_counts:
            best_counts = counts
            bestH2to1 = H
            inliers = inliers_iters

    return bestH2to1, inliers


def compositeH(H2to1, template, img):
    # Create a composite image after warping the template image on top
    # of the image using the homography

    # Note that the homography we compute is from the image to the template;
    # x_template = H2to1*x_photo
    # For warping the template to the image, we need to invert it.

    # Create mask of same size as template
    white_mask = 255 * np.ones((template.shape[0], template.shape[1]), dtype=np.uint8)

    # Warp mask by appropriate homography
    white_mask_wrapped = cv2.warpPerspective(white_mask, H2to1, (img.shape[1], img.shape[0]))
    black_mask_wrapped = cv2.bitwise_not(white_mask_wrapped)

    # Warp template by appropriate homography
    template_wrapped = cv2.warpPerspective(template, H2to1, (img.shape[1], img.shape[0]))

    # Use mask to combine the warped template and the image
    img_masked = cv2.bitwise_or(img, template_wrapped, mask=black_mask_wrapped)
    composite_img = cv2.bitwise_or(template_wrapped, img_masked)

    return composite_img
