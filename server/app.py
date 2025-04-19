from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import uuid
from werkzeug.utils import secure_filename
from mobilenet_classifier import classifier
import base64
from PIL import Image
import io
import json

app = Flask(__name__)
CORS(app)  # 允许跨域请求

# 配置上传文件存储路径
UPLOAD_FOLDER = 'uploads'
CLASSIFIED_FOLDER = 'classified'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

# 确保目录存在
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(os.path.join(CLASSIFIED_FOLDER, 'cat'), exist_ok=True)
os.makedirs(os.path.join(CLASSIFIED_FOLDER, 'dog'), exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 限制上传文件大小为16MB

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'ok'})

@app.route('/api/classify', methods=['POST'])
def classify_image():
    try:
        # 确保请求包含 JSON 数据
        if not request.is_json:
            return jsonify({'error': 'Content-Type must be application/json'}), 400

        data = request.get_json()
        if 'image' not in data:
            return jsonify({'error': 'No image data provided'}), 400
            
        # 处理Base64图像
        base64_image = data['image']
        # 去除Base64前缀
        if ',' in base64_image:
            base64_image = base64_image.split(',')[1]
            
        # 解码Base64并创建PIL图像
        try:
            image_data = base64.b64decode(base64_image)
            image = Image.open(io.BytesIO(image_data))
        except Exception as e:
            return jsonify({'error': f'Invalid base64 image data: {str(e)}'}), 400
        
        # 使用模型进行预测
        result = classifier.predict_image(image)

        # 输出结果
        print('result', result)
        
        return jsonify({
            'filename': 'uploaded_image.jpg',
            'class': result['class'],
            'confidence': result['confidence']
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/batch-classify', methods=['POST'])
def batch_classify():
    try:
        data = request.json
        input_dir = data.get('input_dir', '')
        
        if not input_dir or not os.path.exists(input_dir):
            return jsonify({'error': 'Invalid input directory'}), 400
            
        # 设置输出目录
        output_cat_dir = data.get('output_cat_dir', os.path.join(CLASSIFIED_FOLDER, 'cat'))
        output_dog_dir = data.get('output_dog_dir', os.path.join(CLASSIFIED_FOLDER, 'dog'))
        
        # 批量分类
        results = classifier.batch_classify(input_dir, output_cat_dir, output_dog_dir)
        
        return jsonify(results)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# 获取分类历史记录API
@app.route('/api/history', methods=['GET'])
def get_history():
    try:
        # 在实际应用中，这里应该从数据库读取历史记录
        # 这里我们简单地返回一个模拟的历史记录列表
        return jsonify({
            'status': 'success',
            'data': []  # 这里应该是从数据库读取的历史记录
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
