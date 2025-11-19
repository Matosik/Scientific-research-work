# geom_align_robust.py
import os
import cv2
import numpy as np

EPS = 1e-8

def save_error_map(error_map: np.ndarray, image_path: str):
    base, _ = os.path.splitext(image_path)
    npy_path = base + "_err.npy"
    np.save(npy_path, error_map)
    return npy_path

def load_error_map_for_image(image_path: str):
    base, _ = os.path.splitext(image_path)
    npy_path = base + "_err.npy"
    if os.path.exists(npy_path):
        return np.load(npy_path)
    return None

def _create_detector(prefer_sift=True):
    if prefer_sift:
        try:
            sift = cv2.SIFT_create()
            return sift, True
        except Exception:
            pass
    # fallback ORB
    orb = cv2.ORB_create(nfeatures=3000)
    return orb, False

def detect_and_match(img1, img2, prefer_sift=True, ratio_thresh=0.75, max_matches=100, debug_save=None):
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY) if img1.ndim == 3 else img1
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY) if img2.ndim == 3 else img2

    detector, is_sift = _create_detector(prefer_sift)
    kp1, des1 = detector.detectAndCompute(gray1, None)
    kp2, des2 = detector.detectAndCompute(gray2, None)

    if des1 is None or des2 is None or len(kp1) == 0 or len(kp2) == 0:
        return np.zeros((0,2),dtype=np.float32), np.zeros((0,2),dtype=np.float32), {'kps1': len(kp1), 'kps2': len(kp2), 'matches': 0, 'debug_img': None}

    if is_sift:
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches_all = flann.knnMatch(des1, des2, k=2)
    else:
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        matches_all = bf.knnMatch(des1, des2, k=2)

    good = []
    for m_n in matches_all:
        if len(m_n) < 2:
            continue
        m, n = m_n[0], m_n[1]
        if m.distance < ratio_thresh * n.distance:
            good.append(m)
    good = sorted(good, key=lambda x: x.distance)[:max_matches]

    pts1 = np.array([kp1[m.queryIdx].pt for m in good], dtype=np.float32) if len(good)>0 else np.zeros((0,2),dtype=np.float32)
    pts2 = np.array([kp2[m.trainIdx].pt for m in good], dtype=np.float32) if len(good)>0 else np.zeros((0,2),dtype=np.float32)

    debug_img_path = None
    if debug_save is not None and len(good)>0:
        vis = cv2.drawMatches(img1, kp1, img2, kp2, good[:200], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        cv2.imwrite(debug_save, vis)
        debug_img_path = debug_save

    info = {'kps1': len(kp1), 'kps2': len(kp2), 'raw_matches': len(good), 'debug_img': debug_img_path, 'used_detector': 'SIFT' if is_sift else 'ORB'}
    return pts1, pts2, info

def sample_weights_from_error_map(points, error_map, mode='nearest', scale=1.0):
    if error_map is None:
        return np.ones(points.shape[0], dtype=np.float32)
    H, W = error_map.shape[:2]
    xs = np.clip(np.round(points[:,0]).astype(int), 0, W-1)
    ys = np.clip(np.round(points[:,1]).astype(int), 0, H-1)
    sampled = error_map[ys, xs].astype(np.float32)
    weights = 1.0 / (sampled * scale + EPS)
    weights = weights / (np.max(weights) + EPS)
    return weights

def refine_affine_leastsq(pts_src, pts_dst, weights=None):

    N = pts_src.shape[0]
    if N < 3:
        raise ValueError("Need at least 3 points to estimate affine.")
    
    A = np.zeros((2*N, 6), dtype=np.float64)
    b = np.zeros((2*N,), dtype=np.float64)

    for i in range(N):
        x, y = pts_src[i]
        xp, yp = pts_dst[i]
        A[2*i]   = [x, y, 0, 0, 1, 0]
        A[2*i+1] = [0, 0, x, y, 0, 1]
        b[2*i]   = xp
        b[2*i+1] = yp

    if weights is None:
        W = np.eye(2*N)
    else:
        w2 = np.repeat(weights, 2)
        W = np.diag(w2)

    ATA = A.T @ W @ A
    ATb = A.T @ W @ b
    params, _, _, _ = np.linalg.lstsq(ATA + EPS*np.eye(6), ATb, rcond=None)
    M = params.reshape(2,3)
    return M

def refine_homography_weighted(pts_src, pts_dst, weights=None):
    N = pts_src.shape[0]
    if N < 4:
        raise ValueError("Need at least 4 points for homography.")
    A = []

    for i in range(N):
        x, y = pts_src[i]
        u, v = pts_dst[i]
        s = 1.0 if weights is None else weights[i]
        sqrt_s = np.sqrt(s)
        row1 = np.array([-x, -y, -1, 0, 0, 0, u*x, u*y, u]) * sqrt_s
        row2 = np.array([0, 0, 0, -x, -y, -1, v*x, v*y, v]) * sqrt_s
        A.append(row1)
        A.append(row2)

    A = np.vstack(A)
    U, S, Vt = np.linalg.svd(A)
    h = Vt[-1, :]
    H = h.reshape(3,3)
    H = H / (H[2,2] + EPS)
    return H

def warp_image(img, M, model='affine', dsize=None, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0)):
    if dsize is None:
        h, w = img.shape[:2]
        dsize = (w, h)
    if model == 'affine':
        return cv2.warpAffine(img, M, dsize, flags=cv2.INTER_LINEAR, borderMode=borderMode, borderValue=borderValue)
    else:
        return cv2.warpPerspective(img, M, dsize, flags=cv2.INTER_LINEAR, borderMode=borderMode, borderValue=borderValue)

def align_images(img_src, img_dst, err_src=None, err_dst=None, prefer_sift=True,
                 model='affine', debug_dir=None, ratio_thresh=0.75, ransac_thresh=3.0):

    if img_src is None or img_dst is None:
        raise FileNotFoundError("Не удалось прочитать одно из изображений.")

    err1 = err_src
    err2 = err_dst

    debug_matches_path = None
    if debug_dir is not None:
        os.makedirs(debug_dir, exist_ok=True)
        debug_matches_path = os.path.join(debug_dir, "matches.png")

    pts1, pts2, info = detect_and_match(img_src, img_dst, prefer_sift=prefer_sift, ratio_thresh=ratio_thresh, debug_save=debug_matches_path)
    debug_info = {'detector': info.get('used_detector'), 'kps1': info.get('kps1'), 'kps2': info.get('kps2'), 'matches_after_ratio': info.get('raw_matches')}

    if pts1.shape[0] < 3:
        raise RuntimeError(f"Недостаточно соответствий после детекции ({pts1.shape[0]}). Попробуй повысить текстуру сцены или указать ручные точки.")

    if model == 'affine':
        # estimate affine with RANSAC using OpenCV
        M_ransac, inliers_mask = cv2.estimateAffine2D(pts2, pts1, method=cv2.RANSAC, ransacReprojThreshold=ransac_thresh, maxIters=2000, confidence=0.995)
        if M_ransac is None:
            raise RuntimeError("RANSAC для аффинного преобразования не нашёл модель.")
        inliers_mask = inliers_mask.ravel().astype(bool)
        num_inliers = np.sum(inliers_mask)
        debug_info.update({'ransac_inliers': int(num_inliers)})

        src_in = pts2[inliers_mask]  
        dst_in = pts1[inliers_mask]

        w_src = sample_weights_from_error_map(dst_in, err1)  
        w_dst = sample_weights_from_error_map(src_in, err2)  

        weights = (w_src + w_dst) / 2.0
        M_refined = refine_affine_leastsq(src_in, dst_in, weights=weights)
        warped = warp_image(img_dst, M_refined, model='affine', dsize=(img_src.shape[1], img_src.shape[0]))
        warped_err = None
        if err2 is not None:
            warped_err = cv2.warpAffine(err2.astype(np.float32), M_refined, (img_src.shape[1], img_src.shape[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=np.nan)
        return M_refined, warped, warped_err, debug_info

    elif model == 'homography':
        H, mask = cv2.findHomography(pts2, pts1, method=cv2.RANSAC, ransacReprojThreshold=ransac_thresh, maxIters=2000, confidence=0.995)
        if H is None:
            raise RuntimeError("RANSAC для гомографии не нашёл модель.")
        mask = mask.ravel().astype(bool)
        num_inliers = np.sum(mask)
        debug_info.update({'ransac_inliers': int(num_inliers)})

        src_in = pts2[mask]
        dst_in = pts1[mask]

        w_src = sample_weights_from_error_map(dst_in, err1)
        w_dst = sample_weights_from_error_map(src_in, err2)

        weights = (w_src + w_dst) / 2.0

        H_refined = refine_homography_weighted(src_in, dst_in, weights=weights)
        warped = warp_image(img_dst, H_refined, model='homography', dsize=(img_src.shape[1], img_src.shape[0]))
        warped_err = None

        if err2 is not None:
            warped_err = cv2.warpPerspective(err2.astype(np.float32), H_refined, (img_src.shape[1], img_src.shape[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=np.nan)
        return H_refined, warped, warped_err, debug_info
    
    else:
        raise ValueError("model must be 'affine' or 'homography'")
