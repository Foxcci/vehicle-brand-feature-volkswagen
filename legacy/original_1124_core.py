"""Verbatim functions from cells 3 and 4 of the supplied 11_24 notebook."""
import cv2
import numpy as np

def extract_red_contour(image_path):
    """从图片中提取最大的红色轮廓，返回 Nx2 的坐标数组（x, y）"""
    img = cv2.imread(image_path)          # BGR
    if img is None:
        raise FileNotFoundError(image_path)

    # 转 HSV，提取红色（两段）
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_red1 = np.array([0, 80, 80])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 80, 80])
    upper_red2 = np.array([180, 255, 255])

    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = cv2.bitwise_or(mask1, mask2)

    # 形态学闭运算，连成一条线
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    # 找轮廓
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if len(contours) == 0:
        raise RuntimeError("未找到轮廓: " + image_path)

    # ✅ 这里改成 lambda，显式传 closed=True
    contour = max(contours, key=lambda c: cv2.arcLength(c, True))

    contour = contour[:, 0, :]    # (N, 1, 2) -> (N, 2)
    return contour.astype(np.float64)

def resample_contour(contour, num_points=100):
    """沿弧长等间距重采样轮廓为 num_points 个点"""
    pts = contour
    # 计算每段长度
    diff = np.diff(pts, axis=0)
    seg_lens = np.sqrt((diff**2).sum(axis=1))
    # 闭合：最后一个点到第一个点
    last_seg = np.linalg.norm(pts[0] - pts[-1])
    seg_lens = np.append(seg_lens, last_seg)

    cumlen = np.insert(np.cumsum(seg_lens), 0, 0.0)
    total_len = cumlen[-1]

    # 等间距采样位置（不包含 total_len）
    target_lens = np.linspace(0, total_len, num_points, endpoint=False)

    resampled = []
    for t in target_lens:
        # 找到所在区间
        idx = np.searchsorted(cumlen, t, side='right') - 1
        if idx >= len(pts):
            idx = len(pts) - 1
        # 当前段起点和终点
        p0 = pts[idx]
        if idx == len(pts) - 1:
            p1 = pts[0]
        else:
            p1 = pts[idx + 1]
        # 当前段内的插值比例
        seg_start = cumlen[idx]
        seg_end = cumlen[idx + 1]
        if seg_end == seg_start:
            alpha = 0.0
        else:
            alpha = (t - seg_start) / (seg_end - seg_start)
        p = (1 - alpha) * p0 + alpha * p1
        resampled.append(p)

    return np.array(resampled)

def gm11_predict_next(x0):
    """
    GM(1,1) 灰色预测，输入长度为 n>=4 的一维序列 x0（numpy 数组），
    返回下一步 x0(n+1) 的预测值。
    """
    x0 = np.array(x0, dtype=np.float64).flatten()
    n = len(x0)
    if n < 4:
        raise ValueError("GM(1,1) 至少需要 4 个数据点")

    # 1-AGO
    x1 = np.cumsum(x0)

    # 背景值
    z1 = 0.5 * (x1[1:] + x1[:-1])  # 长度 n-1

    # 构造 B, Y
    B = np.column_stack((-z1, np.ones(n-1)))
    Y = x0[1:].reshape(-1, 1)

    # 最小二乘估计参数
    BTB = B.T @ B
    if np.linalg.det(BTB) == 0:
        # 退化时简单返回最后一个值（防止数值问题）
        return x0[-1]
    theta = np.linalg.inv(BTB) @ B.T @ Y
    a, b = theta[0, 0], theta[1, 0]

    # 时间响应
    def x1_hat(k):
        return (x0[0] - b / a) * np.exp(-a * (k - 1)) + b / a

    # 预测下一步 k = n+1
    x1_next = x1_hat(n + 1)
    x1_n = x1_hat(n)
    x0_next = x1_next - x1_n
    return x0_next
