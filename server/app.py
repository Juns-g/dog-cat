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
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024  # 限制上传文件大小为16MB

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/api/classify', methods=['POST'])
def classify_image():
    try:
        if not request.is_json:
            return jsonify({'error': 'Content-Type must be application/json'}), 400

        data = request.get_json()
        if 'image' not in data:
            return jsonify({'error': 'No image data provided'}), 400
            
        base64_image = data['image']
        if ',' in base64_image:
            base64_image = base64_image.split(',')[1]
            
        try:
            image_data = base64.b64decode(base64_image)
            image = Image.open(io.BytesIO(image_data))
        except Exception as e:
            return jsonify({'error': f'Invalid base64 image data: {str(e)}'}), 400
        
        # 使用模型进行预测
        result = classifier.predict_image(image)

        print('model classifier result', result)
        
        return jsonify({
            'class': result['class'],
            'confidence': result['confidence'],
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

        print('batch classify results', results)
        return jsonify(results)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
