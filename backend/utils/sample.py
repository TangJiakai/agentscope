import random
import math
import matplotlib.pyplot as plt

def generate_points_sampling(k, radius_ratio=0.4, initial_center_dist=0.12, canvas_size=1.0):
    """
    根据指定的参数生成随机选择的圆并绘制在画布上。

    参数：
    - k：需要绘制的圆的数量。
    - radius_ratio：半径占圆心距的比例，默认为0.4。
    - initial_center_dist：初始圆心距，默认为0.12。
    - canvas_size：画布的大小，默认为1.0（即1x1的画布）。

    返回：
    - selected_positions：选择的圆的圆心坐标列表。
    - radius：圆的半径。
    """
    # 计算圆心之间的距离
    def calculate_n(center_dist, canvas_size):
        """
        根据圆心距计算画布上能放置的最大圆数
        """
        # 每行能放的圆的个数
        circles_per_row = int(canvas_size // center_dist)
        # 每列能放的圆的个数
        circles_per_col = int(canvas_size // center_dist)
        
        # 总的圆数
        return circles_per_row * circles_per_col, circles_per_row, circles_per_col

    def generate_grid_positions(circles_per_row, circles_per_col, center_dist):
        """
        根据每行每列能放的圆数和圆心距生成所有圆的坐标
        """
        positions = []
        for i in range(circles_per_row):
            for j in range(circles_per_col):
                # 计算圆心位置
                x = (i + 0.5) * center_dist
                y = (j + 0.5) * center_dist
                positions.append((x, y))
        return positions

    def plot_points(points, radius):
        """
        画出选择的圆
        """
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # 设置坐标范围为画布大小
        ax.set_xlim(0, canvas_size)
        ax.set_ylim(0, canvas_size)
    
        # 画出选择的圆，使用实心填充
        for point in points:
            circle = plt.Circle(point, radius, color='r', fill=True)  # 修改为实心圆
            ax.add_patch(circle)
            # ax.plot(point[0], point[1], 'ro')  # 红色点标记圆心
    
        ax.set_aspect('equal')
        plt.title(f'随机选择的 {len(points)} 个圆')
        plt.show()

    # 初始化参数
    center_dist = initial_center_dist

    # 计算初始的最大圆数n
    n, circles_per_row, circles_per_col = calculate_n(center_dist, canvas_size)
    
    # 如果 k > n，需要调整圆心距，缩短圆心距来放更多圆
    while k > n:
        center_dist *= 0.9  # 缩短圆心距
        n, circles_per_row, circles_per_col = calculate_n(center_dist, canvas_size)
    
    # 如果 k 远小于 n，可以增大圆心距，使圆之间间隔更大
    while k / n < 0.5 and center_dist * 1.1 < canvas_size / min(circles_per_row, circles_per_col):
        center_dist *= 1.1  # 增加圆心距
        n, circles_per_row, circles_per_col = calculate_n(center_dist, canvas_size)
    
    # 根据圆心距计算圆的半径
    radius = center_dist * radius_ratio

    # 生成所有圆的坐标
    positions = generate_grid_positions(circles_per_row, circles_per_col, center_dist)

    # 检查圆是否在画布内（考虑半径）
    valid_positions = []
    for pos in positions:
        x, y = pos
        if radius <= x <= canvas_size - radius and radius <= y <= canvas_size - radius:
            valid_positions.append(pos)

    # 更新可用的圆数量
    n = len(valid_positions)

    # 如果可用的圆少于需要的数量，调整参数
    while k > n:
        center_dist *= 0.9  # 缩短圆心距
        radius = center_dist * radius_ratio
        n, circles_per_row, circles_per_col = calculate_n(center_dist, canvas_size)
        positions = generate_grid_positions(circles_per_row, circles_per_col, center_dist)
        valid_positions = []
        for pos in positions:
            x, y = pos
            if radius <= x <= canvas_size - radius and radius <= y <= canvas_size - radius:
                valid_positions.append(pos)
        n = len(valid_positions)

    # 随机选择k个圆
    selected_positions = random.sample(valid_positions, k)

    # 画出这些随机选择的圆
    plot_points(selected_positions, radius)

    # print(f"最终圆心距: {center_dist}")
    # print(f"最终半径: {radius}")
    # print(f"可用的圆的数量: {n}")
    # print(f"选择的圆的数量: {k}")

    return selected_positions, radius

# 调用函数，生成并绘制100个实心圆
selected_positions, raius = generate_points_sampling(k=30, radius_ratio=0.4, initial_center_dist=0.12, canvas_size=1.0)
# print(selected_positions, radius)
