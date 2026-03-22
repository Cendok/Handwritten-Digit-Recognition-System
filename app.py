from datetime import timedelta
import torch
from torchvision import transforms
from PIL import Image
from flask import jsonify, render_template, request, Flask
from pth import CNN

# 先加载并设置你的数字识别模型
device = "cuda" if torch.cuda.is_available() else "cpu"
model = CNN().to(device)  # 初始化你的模型
model.load_state_dict(torch.load('./mnist_cnn_model.pth'))  # 加载训练好的模型
model.eval()  # 设置为评估模式

# 定义转换操作
transform = transforms.Compose([
    transforms.Grayscale(),  # 转为灰度图像
    transforms.Resize((28, 28)),  # 调整大小，符合MNIST要求
    transforms.ToTensor(),  # 转换为Tensor
])

def load_digit_image(image_path):
    image = Image.open(image_path)
    image = transform(image)  # 应用预处理
    image = image.unsqueeze(0).to(device)  # 增加batch维度并移至设备
    return image

def predict_digit(image_tensor):
    with torch.no_grad():
        output = model(image_tensor)  # 进行预测
        _, predicted = torch.max(output.data, 1)  # 获取预测结果
    return predicted.item()  # 返回预测的数字


#设置成能找到静态文件的路径，CSS、JavaScript、图像等，static_url_path="/static"
app = Flask(__name__, static_url_path="/static")

# 自动重载模板文件
app.jinja_env.auto_reload = True
app.config['TEMPLATES_AUTO_RELOAD'] = True
# 设置静态文件缓存过期时间
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = timedelta(seconds=1)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

@app.route('/', methods=['GET', 'POST'])
def root():
    return render_template("index.html")

# predict()函数必须跟在这段下面
# 路径名称"/predict"必须跟定义的函数一致def predict()
@app.route("/predict", methods=["GET", "POST"])
@torch.no_grad()

def predict():
    info = {}
    try:
        image_file = request.files["file0"]  # 从前端获取文件
        img_bytes = image_file.read()  # 读取文件内容
        image_path = './number/digit1.png'  # 保存路径
        with open(image_path, 'wb') as f:
            f.write(img_bytes)  # 保存图像文件

        # 加载并处理图像
        digit_image = load_digit_image(image_path)
        predicted_digit = predict_digit(digit_image)  # 预测数字

        info["result"] = f"预测的数字是：{predicted_digit}"  # 返回结果
    except Exception as e:
        info["err"] = str(e)
    return jsonify(info)  # 返回json格式结果

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=1235)
#app.run(debug=True, host="0.0.0.0", port=1235)
#关闭调试，否则无限循环训练，无法打开网页