import pywebio
import nibabel
import io
from matplotlib import pyplot as plt
import PIL.Image

import torch

import random
def generate_random_str(target_length=32):
    random_str = ''
    base_str = 'ABCDEFGHIGKLMNOPQRSTUVWXYZabcdefghigklmnopqrstuvwxyz0123456789'
    length = len(base_str) - 1
    for i in range(target_length):
        random_str += base_str[random.randint(0, length)]
    return random_str


def zlzheimer_diagnostic_system(is_demo=False):
    """
    基于 PyWebIO 和 PyTorch 的阿尔兹海默智能诊断系统
    """
    def show_img(img): 
        """
        接收二进制图像，用于展示图像
        """
        pywebio.output.popup("加载中", [  
                pywebio.output.put_loading(), 
        ])
        pywebio.output.popup("上传的图像",[ 
            pywebio.output.put_image(img), 
        ])

    from pyecharts.charts import Bar

    def draw_chart(x_label,data):
        """
        输入 x 轴标签和数据列表，返回图表的 HTML 文本
        """
        chart = (
            Bar()
            .add_xaxis(x_label) # 设置 x 轴标签列表
            .add_yaxis("output_value", data) # 传入数据列表和标签
            .set_global_opts(title_opts={"text": "模型输出", "subtext": ""},) # text 的值为图表标题
        )
        return chart.render_notebook() # 返回图表对象的 html 文本

    while 1:
        pywebio.output.clear()
        pywebio.output.put_markdown("# 基于 PyWebIO 和 PyTorch 的阿尔兹海默智能诊断系统")
        pywebio.output.put_warning('识别结果仅供参考')
        pywebio.output.put_button('使用示例图像', onclick=lambda :zlzheimer_diagnostic_system(is_demo=True))
        # input_img = pywebio.input.file_upload(label="上传图片", accept=[".jpg", ".png", ".jpeg"])
        nii_path = "./demodata/demo.nii"
        if not is_demo:
            input_img = pywebio.input.file_upload(label="上传图像", accept=[".nii"])
            pywebio.output.popup("加载中", [
                pywebio.output.put_loading(),
            ])
            input_img = io.BytesIO(input_img['content'])
            nii_path = "./uploaded_img/" + generate_random_str() + ".nii"
            with open(nii_path, 'wb') as file:
                file.write(input_img.getvalue())  # 保存到本地

        if is_demo:
            pywebio.output.popup("加载中", [
                pywebio.output.put_loading(),
            ])
            is_demo = False

        img = nibabel.load(nii_path)
        img = img.get_fdata()
        print(img.shape)
        # (166, 256, 256, 1)

        torch.no_grad()
        test_model = torch.load("./myModel_109.pth", map_location=torch.device('cpu'))
        test_model.eval()

        processed_img = torch.from_numpy(img)
        processed_img = processed_img.squeeze()
        processed_img = processed_img.reshape(1, -1, 256, 256)
        processed_img = processed_img[:, 0:160, :, :].float()
        processed_img = processed_img.reshape((1, 1, -1, 256, 256))

        output = None
        with torch.no_grad():
            output = test_model(processed_img)
        ans_y = output.squeeze().tolist()
        print(ans_y)

        from datasets import LABEL_LIST
        if min(ans_y) < 0:
            m = min(ans_y)
            for i in range(len(ans_y)):
                ans_y[i] -= 1.2 * m
        chart_html = draw_chart(LABEL_LIST, ans_y)
        pywebio.output.put_html(chart_html)

        ans = LABEL_LIST[output.argmax(1).item()]
        if ans == 'AD':
            ans += '（阿尔茨海默病）'
        elif ans == 'CN':
            ans += '（认知正常）'
        elif ans == 'MCI':
            ans += '（轻度认知障碍）'
        elif ans == 'EMCI':
            ans += '（早期轻度认知障碍）'
        elif ans == 'LMCI':
            ans += '（晚期轻度认知障碍）'
        show_result = [pywebio.output.put_markdown("诊断为：\n # " + ans),
                       pywebio.output.put_warning('结果仅供参考'),]
        pywebio.output.popup(title='AI识别结果', content=show_result)

        while 1:
            act = pywebio.input.actions(' ', ['查看图像', '上传新图像'], )
            if act == '上传新图像':
                pywebio.output.clear()
                break
            dim = pywebio.input.radio('查看视角', ['X', 'Y', 'Z'])
            max_index = 0
            if dim == 'X':
                max_index = img.shape[0]
            if dim == 'Y':
                max_index = img.shape[1]
            if dim == 'Z':
                max_index = img.shape[2]
            index = pywebio.input.slider("查看层数", max_value=max_index, step=1)

            if dim == 'X':
                plt.imshow(img[index, :, :, :], cmap='gray')
            if dim == 'Y':
                plt.imshow(img[:, index, :, :], cmap='gray')
            if dim == 'Z':
                plt.imshow(img[:, :, index, :], cmap='gray')

            plt.axis('off')
            png_path = './uploaded_img/' + generate_random_str() + '.png'
            plt.savefig(png_path, bbox_inches='tight', pad_inches=0)
            show_img(PIL.Image.open(png_path))


if __name__ == "__main__":
    pywebio.platform.start_server(
        applications=[zlzheimer_diagnostic_system, ], # applications 参数为一个可迭代对象（此处是列表），里面放要运行的主函数。
        auto_open_webbrowser=False, # 不自动打开浏览器
        cdn=False, # 不使用 cdn
        debug=True, # 可以看到报错
        port=8080  # 运行在 8080 端口
    )
    # start_server 函数启动一个 http 服务器

