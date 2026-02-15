"""
Геометрическое согласование изображений (Image Registration)
=============================================================

Поддерживает несколько методов:
- ECC (Enhanced Correlation Coefficient) — intensity-based, для простых/малых изображений
- Phase Correlation — быстрый, только сдвиг
- ORB — feature-based, для сложных изображений с текстурой

Структура модуля:
-----------------
ОСНОВНЫЕ ФУНКЦИИ:
    register_images()      — главная входная функция (все методы)
    compute_similarity()   — метрики качества (MSE, PSNR, NCC)

ВНУТРЕННИЕ ФУНКЦИИ (по методам):
    [ECC]   _register_ecc()           — основная ECC регистрация
    [ECC]   _ecc_with_pyramid()       — ECC с пирамидой для лучшей сходимости
    [PHASE] _register_phase_correlation() — фазовая корреляция
    [ORB]   _register_orb()           — feature-based с ORB детектором

УТИЛИТЫ:
    [ALL]   to_grayscale()            — конвертация в grayscale
    [ALL]   extract_transform_params() — извлечение параметров из матрицы
"""

import cv2
import numpy as np
from typing import Tuple, Optional, Literal


# =============================================================================
# ОСНОВНЫЕ ФУНКЦИИ
# =============================================================================

def register_images(
    ref_img: np.ndarray,
    mov_img: np.ndarray,
    method: Literal["ecc", "phase", "orb"] = "ecc",
    transform_type: Literal["translation", "euclidean", "affine", "homography"] = "affine",
    num_iterations: int = 5000,
    termination_eps: float = 1e-10
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    [ВСЕ МЕТОДЫ] Главная входная функция для геометрического согласования.
    
    Выполняет согласование mov_img относительно ref_img.
    
    Параметры:
    ----------
    ref_img : np.ndarray
        Референсное изображение (grayscale)
    mov_img : np.ndarray
        Изображение для согласования (grayscale)
    method : str
        Метод согласования:
        - "ecc"   : Enhanced Correlation Coefficient 
                    → для простых/маленьких изображений, малых смещений
        - "phase" : Phase Correlation 
                    → только сдвиг, очень быстрый
        - "orb"   : Feature-based с ORB детектором 
                    → для сложных изображений с текстурой, больших смещений
    transform_type : str
        Тип преобразования (для ECC и ORB):
        - "translation" : только сдвиг (tx, ty) — 2 параметра
        - "euclidean"   : сдвиг + поворот — 3 параметра
        - "affine"      : полное аффинное — 6 параметров
        - "homography"  : перспективное — 8 параметров
    num_iterations : int
        [ECC] Макс. количество итераций оптимизации
    termination_eps : float
        [ECC] Критерий сходимости (порог изменения)
    
    Возвращает:
    -----------
    aligned_img : np.ndarray
        Согласованное изображение
    transform_matrix : np.ndarray
        Матрица преобразования (2x3 или 3x3)
    info : dict
        Информация о согласовании:
        - translation_x, translation_y : сдвиг в пикселях
        - rotation_deg : угол поворота в градусах
        - scale_avg : средний масштаб
        - correlation_coefficient : [ECC] коэфф. корреляции
        - keypoints_ref, keypoints_mov : [ORB] кол-во ключевых точек
        - phase_response : [PHASE] отклик фазовой корреляции
    
    Пример:
    -------
    >>> ref = load_grayscale("reference.png")
    >>> mov = load_grayscale("moving.png")
    >>> aligned, matrix, info = register_images(ref, mov, method="ecc")
    >>> print(f"Сдвиг: ({info['translation_x']:.1f}, {info['translation_y']:.1f})")
    """
    
    # Проверка входных данных
    if ref_img is None or mov_img is None:
        raise ValueError("Изображения не могут быть None")
    
    # Преобразование в grayscale если нужно
    ref_gray = to_grayscale(ref_img)
    mov_gray = to_grayscale(mov_img)
    
    # Приведение к одному размеру
    if ref_gray.shape != mov_gray.shape:
        mov_gray = cv2.resize(mov_gray, (ref_gray.shape[1], ref_gray.shape[0]))
    
    # Выбор метода
    if method == "ecc":
        aligned, matrix, info = _register_ecc(
            ref_gray, mov_gray, transform_type, num_iterations, termination_eps
        )
    elif method == "phase":
        aligned, matrix, info = _register_phase_correlation(ref_gray, mov_gray)
    elif method == "orb":
        aligned, matrix, info = _register_orb(ref_gray, mov_gray, transform_type)
    else:
        raise ValueError(f"Неизвестный метод: {method}")
    
    info['method'] = method
    info['transform_type'] = transform_type
    
    return aligned, matrix, info


def load_grayscale(path: str) -> np.ndarray:
    """
    [ВСЕ МЕТОДЫ] Загружает изображение в grayscale формате.
    
    Параметры:
    ----------
    path : str
        Путь к файлу изображения
    
    Возвращает:
    -----------
    np.ndarray
        Grayscale изображение (H, W), dtype=uint8
    
    Raises:
    -------
    FileNotFoundError
        Если файл не найден или не удалось загрузить
    """
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Не удалось загрузить: {path}")
    return img


def compute_similarity(img1: np.ndarray, img2: np.ndarray) -> dict:
    """
    [ВСЕ МЕТОДЫ] Вычисляет метрики сходства между двумя изображениями.
    
    Используется для оценки качества согласования (до/после).
    
    Параметры:
    ----------
    img1, img2 : np.ndarray
        Сравниваемые изображения (grayscale)
    
    Возвращает:
    -----------
    dict с метриками:
        - mse  : Mean Squared Error (↓ лучше)
        - psnr : Peak Signal-to-Noise Ratio в dB (↑ лучше)
        - ncc  : Normalized Cross-Correlation [-1, 1] (↑ лучше, 1 = идеально)
    
    Пример:
    -------
    >>> before = compute_similarity(ref, mov)
    >>> after = compute_similarity(ref, aligned)
    >>> print(f"MSE: {before['mse']:.1f} → {after['mse']:.1f}")
    """
    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
    
    img1_f = img1.astype(np.float64)
    img2_f = img2.astype(np.float64)
    
    # MSE (Mean Squared Error)
    mse = np.mean((img1_f - img2_f) ** 2)
    
    # PSNR (Peak Signal-to-Noise Ratio)
    psnr = 10 * np.log10(255**2 / mse) if mse > 0 else float('inf')
    
    # NCC (Normalized Cross-Correlation)
    img1_norm = img1_f - np.mean(img1_f)
    img2_norm = img2_f - np.mean(img2_f)
    denom = np.sqrt(np.sum(img1_norm**2) * np.sum(img2_norm**2))
    ncc = np.sum(img1_norm * img2_norm) / denom if denom > 0 else 0.0
    
    return {'mse': mse, 'psnr': psnr, 'ncc': ncc}


# =============================================================================
# МЕТОД ECC (Enhanced Correlation Coefficient)
# =============================================================================

def _register_ecc(
    ref_gray: np.ndarray,
    mov_gray: np.ndarray,
    transform_type: str,
    num_iterations: int,
    termination_eps: float
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    [ECC] Intensity-based регистрация методом Enhanced Correlation Coefficient.
    
    Принцип работы:
    ---------------
    Итеративно оптимизирует матрицу преобразования M, максимизируя
    корреляцию между ref и warp(mov, M).
    
    Преимущества:
    - Работает на простых изображениях без текстуры
    - Точен для малых смещений
    - Не требует детекции ключевых точек
    
    Недостатки:
    - Может не сойтись при больших смещениях (>15-20%)
    - Чувствителен к изменениям освещения
    
    Параметры:
    ----------
    ref_gray, mov_gray : np.ndarray
        Входные grayscale изображения
    transform_type : str
        Тип преобразования: "translation", "euclidean", "affine", "homography"
    num_iterations : int
        Максимальное число итераций
    termination_eps : float
        Порог сходимости
    
    Возвращает:
    -----------
    aligned, warp_matrix, info
    """
    
    # Определение типа преобразования
    warp_mode_map = {
        "translation": cv2.MOTION_TRANSLATION,
        "euclidean": cv2.MOTION_EUCLIDEAN,
        "affine": cv2.MOTION_AFFINE,
        "homography": cv2.MOTION_HOMOGRAPHY
    }
    warp_mode = warp_mode_map.get(transform_type, cv2.MOTION_AFFINE)
    
    # Инициализация матрицы преобразования (единичная)
    if warp_mode == cv2.MOTION_HOMOGRAPHY:
        warp_matrix = np.eye(3, 3, dtype=np.float32)
    else:
        warp_matrix = np.eye(2, 3, dtype=np.float32)
    
    # Критерий остановки
    criteria = (
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
        num_iterations,
        termination_eps
    )
    
    # Нормализация изображений для лучшей сходимости
    ref_norm = cv2.normalize(ref_gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.float32)
    mov_norm = cv2.normalize(mov_gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.float32)
    
    # Применение гауссова размытия для устойчивости к шуму
    ref_blur = cv2.GaussianBlur(ref_norm, (3, 3), 0)
    mov_blur = cv2.GaussianBlur(mov_norm, (3, 3), 0)
    
    # Поиск преобразования методом ECC
    try:
        cc, warp_matrix = cv2.findTransformECC(
            ref_blur, mov_blur, warp_matrix, warp_mode, criteria
        )
    except cv2.error as e:
        # Fallback 1: пирамида изображений
        try:
            cc, warp_matrix = _ecc_with_pyramid(
                ref_blur, mov_blur, warp_mode, criteria
            )
        except (cv2.error, ValueError):
            # Fallback 2: упрощение до translation
            warp_matrix = np.eye(2, 3, dtype=np.float32)
            simpler_criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 1000, 1e-6)
            try:
                cc, warp_matrix = cv2.findTransformECC(
                    ref_blur, mov_blur, warp_matrix, cv2.MOTION_TRANSLATION, simpler_criteria
                )
            except cv2.error:
                # Fallback 3: phase correlation
                shift, response = cv2.phaseCorrelate(
                    ref_blur.astype(np.float64), mov_blur.astype(np.float64)
                )
                warp_matrix = np.array([[1, 0, -shift[0]], [0, 1, -shift[1]]], dtype=np.float32)
                cc = response
    
    # Применение преобразования
    height, width = ref_gray.shape
    if warp_mode == cv2.MOTION_HOMOGRAPHY:
        aligned = cv2.warpPerspective(
            mov_gray, warp_matrix, (width, height),
            flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_REPLICATE
        )
    else:
        aligned = cv2.warpAffine(
            mov_gray, warp_matrix, (width, height),
            flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_REPLICATE
        )
    
    # Извлечение параметров
    info = extract_transform_params(warp_matrix, warp_mode == cv2.MOTION_HOMOGRAPHY)
    info['correlation_coefficient'] = cc
    
    return aligned, warp_matrix, info


def _ecc_with_pyramid(
    ref_img: np.ndarray,
    mov_img: np.ndarray,
    warp_mode: int,
    criteria: tuple,
    num_levels: int = 3
) -> Tuple[float, np.ndarray]:
    """
    [ECC] ECC с пирамидой изображений (coarse-to-fine).
    
    Вспомогательная функция для улучшения сходимости ECC.
    
    Принцип:
    --------
    1. Создаёт пирамиду (уменьшенные версии изображений)
    2. Начинает с самого маленького уровня (грубая оценка)
    3. Уточняет на каждом следующем уровне
    
    Это помогает избежать локальных минимумов при больших смещениях.
    
    Параметры:
    ----------
    ref_img, mov_img : np.ndarray
        Входные изображения (float32)
    warp_mode : int
        Режим OpenCV (MOTION_TRANSLATION, MOTION_AFFINE, etc.)
    criteria : tuple
        Критерий остановки
    num_levels : int
        Количество уровней пирамиды (default=3)
    
    Возвращает:
    -----------
    cc : float
        Коэффициент корреляции
    warp_matrix : np.ndarray
        Найденная матрица преобразования
    """
    
    # Создание пирамид
    ref_pyr = [ref_img]
    mov_pyr = [mov_img]
    
    for _ in range(num_levels - 1):
        ref_pyr.append(cv2.pyrDown(ref_pyr[-1]))
        mov_pyr.append(cv2.pyrDown(mov_pyr[-1]))
    
    # Инициализация матрицы и коэффициента корреляции
    if warp_mode == cv2.MOTION_HOMOGRAPHY:
        warp_matrix = np.eye(3, 3, dtype=np.float32)
    else:
        warp_matrix = np.eye(2, 3, dtype=np.float32)
    
    cc = 0.0  # Инициализация коэффициента корреляции
    
    # Итерация от грубого к точному (coarse-to-fine)
    for level in range(num_levels - 1, -1, -1):
        ref_level = ref_pyr[level]
        mov_level = mov_pyr[level]
        
        try:
            cc, warp_matrix = cv2.findTransformECC(
                ref_level, mov_level, warp_matrix, warp_mode, criteria
            )
        except cv2.error as e:
            if level == 0:
                raise ValueError(f"ECC не сошёлся: {e}")
        
        # Масштабирование матрицы для следующего уровня
        if level > 0:
            if warp_mode == cv2.MOTION_HOMOGRAPHY:
                warp_matrix[0, 2] *= 2
                warp_matrix[1, 2] *= 2
                warp_matrix[2, 0] /= 2
                warp_matrix[2, 1] /= 2
            else:
                warp_matrix[0, 2] *= 2
                warp_matrix[1, 2] *= 2
    
    return cc, warp_matrix


# =============================================================================
# МЕТОД PHASE CORRELATION
# =============================================================================

def _register_phase_correlation(
    ref_gray: np.ndarray,
    mov_gray: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    [PHASE] Регистрация методом фазовой корреляции.
    
    Принцип работы:
    ---------------
    Использует теорему о сдвиге Фурье: сдвиг в пространственной области
    соответствует линейному изменению фазы в частотной области.
    
    Преимущества:
    - Очень быстрый (O(n log n))
    - Точный для чистых сдвигов
    - Устойчив к шуму
    
    Недостатки:
    - Только сдвиг (translation), нет поворота/масштаба
    - Не работает при значительных различиях в яркости
    
    Параметры:
    ----------
    ref_gray, mov_gray : np.ndarray
        Входные grayscale изображения
    
    Возвращает:
    -----------
    aligned : np.ndarray
        Согласованное изображение
    warp_matrix : np.ndarray
        Матрица сдвига (2x3)
    info : dict
        translation_x, translation_y, phase_response
    """
    
    # Phase correlation через FFT
    shift, response = cv2.phaseCorrelate(
        ref_gray.astype(np.float64),
        mov_gray.astype(np.float64)
    )
    
    # Матрица сдвига
    tx, ty = shift
    warp_matrix = np.array([
        [1, 0, -tx],
        [0, 1, -ty]
    ], dtype=np.float32)
    
    # Применение сдвига
    height, width = ref_gray.shape
    aligned = cv2.warpAffine(
        mov_gray, warp_matrix, (width, height),
        borderMode=cv2.BORDER_REPLICATE
    )
    
    info = {
        'translation_x': -tx,
        'translation_y': -ty,
        'phase_response': response,
        'rotation_deg': 0,
        'scale_avg': 1.0
    }
    
    return aligned, warp_matrix, info


# =============================================================================
# МЕТОД ORB (Feature-based)
# =============================================================================

def _register_orb(
    ref_gray: np.ndarray,
    mov_gray: np.ndarray,
    transform_type: str,
    max_features: int = 1000,
    good_match_percent: float = 0.3
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    [ORB] Feature-based регистрация с ORB детектором.
    
    Принцип работы:
    ---------------
    1. Детекция ключевых точек (углы, края) с ORB
    2. Вычисление дескрипторов для каждой точки
    3. Сопоставление точек между изображениями
    4. Оценка матрицы преобразования с RANSAC
    
    Преимущества:
    - Работает при больших смещениях/поворотах
    - Устойчив к частичным окклюзиям
    - Не требует хорошей инициализации
    
    Недостатки:
    - Требует текстуру/характерные точки на изображении
    - Не работает на простых изображениях (мало особенностей)
    - Менее точен для субпиксельных смещений
    
    Параметры:
    ----------
    ref_gray, mov_gray : np.ndarray
        Входные grayscale изображения
    transform_type : str
        "affine" или "homography"
    max_features : int
        Макс. количество ключевых точек для детекции
    good_match_percent : float
        Доля лучших совпадений для использования
    
    Возвращает:
    -----------
    aligned, warp_matrix, info
    
    Raises:
    -------
    ValueError
        Если недостаточно ключевых точек (используйте ECC)
    """
    
    # Улучшение контраста для лучшей детекции особенностей
    ref_enhanced = cv2.equalizeHist(ref_gray)
    mov_enhanced = cv2.equalizeHist(mov_gray)
    
    # ORB детектор
    orb = cv2.ORB_create(nfeatures=max_features)
    
    keypoints_ref, descriptors_ref = orb.detectAndCompute(ref_enhanced, None)
    keypoints_mov, descriptors_mov = orb.detectAndCompute(mov_enhanced, None)
    
    if descriptors_ref is None or descriptors_mov is None:
        raise ValueError(
            "Не удалось найти ключевые точки. "
            "Попробуйте method='ecc' для простых изображений."
        )
    
    if len(keypoints_ref) < 4 or len(keypoints_mov) < 4:
        raise ValueError(
            f"Недостаточно ключевых точек: ref={len(keypoints_ref)}, mov={len(keypoints_mov)}. "
            "Попробуйте method='ecc'."
        )
    
    # Сопоставление дескрипторов (Brute-Force с Hamming distance)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(descriptors_mov, descriptors_ref)
    matches = sorted(matches, key=lambda x: x.distance)
    
    num_good = max(int(len(matches) * good_match_percent), 4)
    good_matches = matches[:num_good]
    
    if len(good_matches) < 4:
        raise ValueError(f"Недостаточно совпадений: {len(good_matches)}")
    
    # Извлечение координат совпавших точек
    points_mov = np.float32([keypoints_mov[m.queryIdx].pt for m in good_matches])
    points_ref = np.float32([keypoints_ref[m.trainIdx].pt for m in good_matches])
    
    # Вычисление матрицы преобразования с RANSAC
    height, width = ref_gray.shape
    use_homography = (transform_type == "homography")
    
    if use_homography:
        warp_matrix, mask = cv2.findHomography(points_mov, points_ref, cv2.RANSAC, 5.0)
        aligned = cv2.warpPerspective(mov_gray, warp_matrix, (width, height))
        inliers = np.sum(mask) if mask is not None else len(good_matches)
    else:
        warp_matrix, mask = cv2.estimateAffinePartial2D(points_mov, points_ref, cv2.RANSAC)
        aligned = cv2.warpAffine(mov_gray, warp_matrix, (width, height))
        inliers = np.sum(mask) if mask is not None else len(good_matches)
    
    info = extract_transform_params(warp_matrix, use_homography)
    info.update({
        'keypoints_ref': len(keypoints_ref),
        'keypoints_mov': len(keypoints_mov),
        'matches': len(matches),
        'good_matches': len(good_matches),
        'inliers': int(inliers)
    })
    
    return aligned, warp_matrix, info


# =============================================================================
# УТИЛИТЫ (для всех методов)
# =============================================================================

def to_grayscale(img: np.ndarray) -> np.ndarray:
    """
    [ВСЕ МЕТОДЫ] Конвертация изображения в grayscale.
    
    Если изображение уже grayscale — возвращает копию.
    
    Параметры:
    ----------
    img : np.ndarray
        Входное изображение (BGR или grayscale)
    
    Возвращает:
    -----------
    np.ndarray
        Grayscale изображение (H, W)
    """
    if len(img.shape) == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img.copy()


def extract_transform_params(matrix: np.ndarray, is_homography: bool) -> dict:
    """
    [ВСЕ МЕТОДЫ] Извлекает параметры преобразования из матрицы.
    
    Декомпозирует матрицу преобразования на интерпретируемые параметры:
    сдвиг, поворот, масштаб, сдвиг (shear).
    
    Параметры:
    ----------
    matrix : np.ndarray
        Матрица преобразования (2x3 или 3x3)
    is_homography : bool
        True если матрица 3x3 (гомография)
    
    Возвращает:
    -----------
    dict с параметрами:
        - translation_x, translation_y : сдвиг в пикселях
        - scale_x, scale_y, scale_avg : масштаб
        - rotation_deg : угол поворота в градусах
        - shear_deg : угол сдвига в градусах
    """
    params = {}
    
    if is_homography:
        a, b, tx = matrix[0, 0], matrix[0, 1], matrix[0, 2]
        c, d, ty = matrix[1, 0], matrix[1, 1], matrix[1, 2]
    else:
        a, b, tx = matrix[0, 0], matrix[0, 1], matrix[0, 2]
        c, d, ty = matrix[1, 0], matrix[1, 1], matrix[1, 2]
    
    # Сдвиг (translation)
    params['translation_x'] = tx
    params['translation_y'] = ty
    
    # Масштаб (scale)
    scale_x = np.sqrt(a**2 + c**2)
    scale_y = np.sqrt(b**2 + d**2)
    params['scale_x'] = scale_x
    params['scale_y'] = scale_y
    params['scale_avg'] = (scale_x + scale_y) / 2
    
    # Поворот (rotation)
    params['rotation_deg'] = np.arctan2(c, a) * 180 / np.pi
    
    # Сдвиг (shear)
    params['shear_deg'] = np.arctan2(b, d) * 180 / np.pi + params['rotation_deg']
    
    return params


