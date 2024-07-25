from PIL import Image, ImageDraw
import random
import os

def create_random_polyline(width, height, num_points=3):
    """在给定区域范围内生成随机点，并确保这些点不共线。"""
    points = []
    while len(points) < num_points:
        new_point = (random.randint(0, width-1), random.randint(0, height-1))
        points.append(new_point)
        # 如果点数多于等于3，检查是否共线
        if len(points) >= 3:
            # 取最后三个点
            p1, p2, p3 = points[-3:]
            # 计算共线性，即面积为0
            if (p2[0] - p1[0]) * (p3[1] - p1[1]) == (p3[0] - p1[0]) * (p2[1] - p1[1]):
                # 如果共线，移除最后一个点
                points.pop()
    return points

def draw_polyline_on_image(width, height, points):
    """在图像上根据点绘制折线。"""
    image = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(image)
    draw.line(points, fill='black', width=1)
    return image

def main():
    width, height = 32, 32
    num_images = 100
    directory = 'output_images'
    
    # 创建保存图片的目录
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    # 生成100张图片
    for i in range(num_images):
        polyline = create_random_polyline(width, height)
        image = draw_polyline_on_image(width, height, polyline)
        image_filepath = os.path.join(directory, f'image_{i+1:03d}.png')
        image.save(image_filepath)

    print(f"{num_images} images generated in '{directory}' directory.")

if __name__ == '__main__':
    main()