from flask import Flask, request, jsonify, send_from_directory
import openai
import os
import io
import base64
import uuid
from datetime import datetime
from PIL import Image
from urllib.parse import urlparse
import requests
import google.generativeai as genai
from google.genai.types import GenerateImagesConfig

app = Flask(__name__)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_IMAGE_MODEL = "imagen-3.0-generate-001"

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
IDEOGRAM_API_KEY = os.getenv("IDEOGRAM_API_KEY")
IDEOGRAM_API_URL = "https://api.ideogram.ai/v1/ideogram-v3/generate"
UPLOAD_FOLDER = 'generated_images'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

from google.cloud import storage

# Cấu hình GCS
GCS_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME")
GCS_CREDENTIALS_JSON = "./sun-production.json" 

def upload_to_gcs(local_file_path, destination_blob_name=None):
    """Upload file lên GCS và trả về public URL"""
    try:
        # Tạo client
        storage_client = storage.Client.from_service_account_json(GCS_CREDENTIALS_JSON)
        bucket = storage_client.bucket(GCS_BUCKET_NAME)

        # Luôn lưu vào folder 'history_redesign/'
        filename = os.path.basename(local_file_path)
        destination_blob_name = destination_blob_name or f"history_redesign/{filename}"

        # Tên file trên GCS
        if not destination_blob_name:
            destination_blob_name = os.path.basename(local_file_path)

        blob = bucket.blob(destination_blob_name)
        blob.upload_from_filename(local_file_path)

        # Thiết lập quyền public
        blob.make_public()
        os.remove(local_file_path)
        return blob.public_url
    except Exception as e:
        print(f"Upload to GCS failed: {e}")
        return None


def base64_to_image_file(b64_data, filename=None):
    """Chuyển base64 thành file ảnh và lưu local"""
    try:
        # Tạo filename nếu không có
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            unique_id = str(uuid.uuid4())[:8]
            filename = f"generated_{timestamp}_{unique_id}.png"
        
        # Đường dẫn đầy đủ
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        
        # Decode base64
        image_data = base64.b64decode(b64_data)
        
        # Mở và xử lý ảnh bằng PIL
        image = Image.open(io.BytesIO(image_data))
        
        # Chuyển sang RGB nếu cần (để lưu JPEG)
        if image.mode in ('RGBA', 'P'):
            # Tạo background trắng cho RGBA
            background = Image.new('RGB', image.size, (255, 255, 255))
            if image.mode == 'RGBA':
                background.paste(image, mask=image.split()[-1])
            else:
                background.paste(image)
            image = background
        
        # Lưu ảnh
        image.save(filepath, format='PNG', quality=95, optimize=True)
        
        print(f"Saved image to: {filepath}")
        return filepath
        
    except Exception as e:
        print(f"Error saving base64 to file: {str(e)}")
        return None

def validate_image_url(url):
    """Kiểm tra URL ảnh có hợp lệ không"""
    try:
        parsed = urlparse(url)
        if not parsed.scheme or not parsed.netloc:
            return False
        valid_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp']
        path = parsed.path.lower()
        
        has_valid_extension = any(path.endswith(ext) for ext in valid_extensions)
        trusted_domains = ['i.ibb.co', 'imgur.com', 'i.imgur.com', 'cdn.discordapp.com']
        is_trusted_domain = any(domain in parsed.netloc for domain in trusted_domains)
        
        return has_valid_extension or is_trusted_domain
    except:
        return False

def download_image(image_url):
    """Download ảnh từ URL và trả về base64 string với nhiều phương pháp fallback"""
    
    # Danh sách các phương pháp download khác nhau
    methods = [
        # Method 1: Standard request with longer timeout
        {
            'timeout': 60,
            'verify': True,
            'stream': True,
            'allow_redirects': True,
            'headers': {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                'Accept': 'image/webp,image/apng,image/*,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.9',
                'Accept-Encoding': 'gzip, deflate, br',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1',
            }
        },
        # Method 2: Disable SSL verification
        {
            'timeout': 60,
            'verify': False,
            'stream': True,
            'allow_redirects': True,
            'headers': {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                'Accept': 'image/*,*/*;q=0.8',
            }
        },
        # Method 3: Simple request without stream
        {
            'timeout': 90,
            'verify': False,
            'stream': False,
            'allow_redirects': True,
            'headers': {
                'User-Agent': 'Python-requests/2.31.0',
                'Accept': '*/*',
            }
        },
        # Method 4: Minimal headers
        {
            'timeout': 120,
            'verify': False,
            'stream': False,
            'allow_redirects': True,
            'headers': {}
        }
    ]
    
    last_error = None
    
    for i, method in enumerate(methods, 1):
        try:
            print(f"Trying download method {i}/{len(methods)}...")
            
            # Tạo session mới cho mỗi method
            session = requests.Session()
            
            # Cài đặt timeout cho adapter
            from requests.adapters import HTTPAdapter
            from urllib3.util.retry import Retry
            
            retry_strategy = Retry(
                total=3,
                backoff_factor=1,
                status_forcelist=[429, 500, 502, 503, 504],
            )
            
            adapter = HTTPAdapter(max_retries=retry_strategy)
            session.mount("http://", adapter)
            session.mount("https://", adapter)
            
            # Cập nhật headers
            session.headers.update(method['headers'])
            
            # Thực hiện request
            response = session.get(
                image_url,
                timeout=method['timeout'],
                verify=method['verify'],
                stream=method['stream'],
                allow_redirects=method['allow_redirects']
            )
            
            response.raise_for_status()
            
            # Kiểm tra content type
            content_type = response.headers.get('content-type', '').lower()
            print(f"Content-Type: {content_type}")
            
            # Đọc dữ liệu ảnh
            image_data = response.content
            print(f"Downloaded {len(image_data)} bytes")
            
            if len(image_data) < 1024:  # Nhỏ hơn 1KB có thể là lỗi
                raise Exception("Downloaded file too small, might be an error page")
            
            # Mở ảnh bằng PIL để xác thực
            try:
                image = Image.open(io.BytesIO(image_data))
                print(f"Image opened successfully: {image.size}, mode: {image.mode}")
                
                # Chuyển về RGB nếu cần
                if image.mode != 'RGB':
                    if image.mode == 'RGBA':
                        background = Image.new('RGB', image.size, (255, 255, 255))
                        background.paste(image, mask=image.split()[-1])
                        image = background
                    else:
                        image = image.convert('RGB')
                
                # Resize nếu ảnh quá lớn (để tránh lỗi với OpenAI API)
                max_size = 2048
                if image.size[0] > max_size or image.size[1] > max_size:
                    image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
                    print(f"Image resized to: {image.size}")
                
                # Lưu lại thành bytes
                img_byte_arr = io.BytesIO()
                image.save(img_byte_arr, format='JPEG', quality=85, optimize=True)
                img_byte_arr.seek(0)
                
                # Chuyển thành base64
                base64_string = base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')
                print(f"Base64 string length: {len(base64_string)}")
                
                return f"data:image/jpeg;base64,{base64_string}"
                
            except Exception as e:
                raise Exception(f"Cannot process image with PIL: {str(e)}")
                
        except Exception as e:
            last_error = str(e)
            print(f"Method {i} failed: {last_error}")
            continue
    
    # Nếu tất cả methods đều thất bại
    raise Exception(f"All download methods failed. Last error: {last_error}")

def describe_image_with_gpt4o_2D(base64_image):
    """Sử dụng GPT-4o để mô tả ảnh chi tiết"""
    client = openai.OpenAI(api_key=OPENAI_API_KEY)
    
    try:
        analysis_prompt = """'**Role:**
You are a professional AI visual designer with expertise in design analysis and prompt engineering for high-fidelity image generation.

**Task:**
Analyze the given image and generate a detailed visual analysis followed by a complete prompt that can recreate the design exactly as it appears.

**Analysis Instructions:**
1. Extract and explicitly describe ONLY the visual design elements:
   - **Typography and text**: any phrases, text, or captions with **exact wording** and **relative position**
   - **Graphics and illustrations**: characters, icons, symbols, decorative elements that are part of the design
   - **Visual layout**: spatial arrangement, composition, hierarchy of design elements
   - **Colors and style**: artistic style, color palette, visual treatment
   - **Design mood**: the aesthetic tone conveyed by the visual elements

2. Generate a complete visual description that preserves all design elements exactly as they appear. Do not suggest improvements or additions.

**CRITICAL CONSTRAINTS:**
- Describe ONLY the graphic design content (text, logos, illustrations, patterns)
- Do NOT mention any physical objects, containers, or products (t-shirts, mugs, signs, etc.)
- Do NOT describe backgrounds that are not part of the design itself
- Do NOT make suggestions or recommendations about the design
- Do NOT comment on what could be improved or added
- Do NOT provide design advice or critique
- ONLY describe what you see, exactly as it appears
- Focus exclusively on the visual design elements that would be reproduced
- For OpenAI image generation, ALWAYS include "on a transparent background" at the end of the Final Prompt section

**Format Requirements:**
- Use the exact section structure below
- Do NOT use JSON format
- Include ALL design elements in comprehensive detail
- Quote all text exactly as written

**Required Response Format:**

### 🔍 Analysis
[Analysis of the design content only: typography, graphic elements, layout, colors, and visual composition - NO physical objects or containers]

### 🎨 Final Prompt
[Visual description focused on the design elements: text, graphics, colors, style, and composition - exclude any physical containers. For transparent backgrounds, end with: "on a transparent background"]

**Critical Requirements:**
- Preserve EXACT text wording and positioning
- Include ALL visual design elements from the image
- Focus ONLY on the design content itself
- NO hex codes or technical specifications
- NO mention of physical objects or containers
- NO suggestions, recommendations, or improvements
- NO commentary about what could be added or changed
- ONLY describe what IS in the image, not what COULD BE
- Focus on complete visual fidelity to the design elements only
"""
        
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text", 
                            "text": analysis_prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": base64_image}
                        }
                    ]
                }
            ],
            max_tokens=800
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        raise Exception(f"Cannot analyze image with GPT-4o: {str(e)}")

def describe_image_with_gpt4o_3D(base64_image):
    """Sử dụng GPT-4o để mô tả ảnh chi tiết"""
    client = openai.OpenAI(api_key=OPENAI_API_KEY)
    
    try:
        analysis_prompt = """'**Role:**
You are a professional AI visual designer with expertise in 3D design analysis and prompt engineering for high-fidelity, three-dimensional image generation.
**Task:**
Analyze the given image and generate a detailed 3D visual analysis followed by a complete prompt that can recreate the design with accurate three-dimensional depth, lighting, and perspective, exactly as it appears.
**Analysis Instructions:**
- Extract and explicitly describe ONLY the 3D visual design elements:
- Typography and text: any phrases, text, or captions with exact wording, depth effects (e.g., embossing, extrusion), and relative 3D positioning
- Graphics and illustrations: characters, icons, symbols, decorative elements, focusing on 3D modeling, textures, and depth
- Visual layout: spatial arrangement, composition, layering, and perspective of design elements
- Colors and style: artistic style, color palette, shading, reflections, and 3D lighting effects
- Design mood: the aesthetic tone conveyed by the 3D elements, materials, and light
- Generate a complete 3D visual description that preserves all design elements exactly as they appear, emphasizing three-dimensional qualities (e.g., volume, perspective, realistic lighting).
Do not suggest improvements or additions.

**CRITICAL CONSTRAINTS:**
- Describe ONLY the 3D design content (text, logos, illustrations, 3D patterns)
- Do NOT mention physical objects or containers (t-shirts, mugs, signs, etc.)
- Do NOT describe backgrounds that are not part of the design itself
- Do NOT make suggestions or recommendations about the design
- Do NOT comment on what could be improved or added
- Do NOT provide design advice or critique
- ONLY describe what you see, focusing on 3D characteristics
- For OpenAI image generation, ALWAYS include "in a realistic 3D style, on a transparent background" at the end of the Final Prompt section
Format Requirements:
- Use the exact section structure below
- Do NOT use JSON format
- Include ALL design elements in comprehensive detail
- Quote all text exactly as written
- Emphasize 3D geometry, textures, and lighting
Required Response Format:

**🔍 Analysis**
[3D analysis of the design content only: typography, 3D graphic elements, layout, colors, depth, and visual composition – NO physical objects or containers]
🎨 Final Prompt
[3D visual description focused on the design elements: text, graphics, colors, style, perspective, and 3D depth – exclude any physical containers. For transparent backgrounds, end with: "in a realistic 3D style, on a transparent background"]
**Critical Requirements:**
- Preserve EXACT text wording and positioning
- Include ALL 3D visual design elements from the image
- Focus ONLY on the 3D design content itself
- NO hex codes or technical specifications
- NO mention of physical objects or containers
- NO suggestions, recommendations, or improvements
- NO commentary about what could be added or changed
- ONLY describe what IS in the image, not what COULD BE
- Highlight 3D depth, realistic lighting, shadows, and perspective
"""
        
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text", 
                            "text": analysis_prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": base64_image}
                        }
                    ]
                }
            ],
            max_tokens=800
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        raise Exception(f"Cannot analyze image with GPT-4o: {str(e)}")
    
def generate_dalle_prompt(image_description):
    """Sử dụng GPT-4o để tạo prompt tối ưu cho DALL-E"""
    client = openai.OpenAI(api_key=OPENAI_API_KEY)
    
    try:
        prompt_generation_request = f"""Dựa trên mô tả ảnh này, hãy tạo một prompt tối ưu cho DALL-E để vẽ lại ảnh giống y hệt:

Mô tả ảnh: {image_description}

Yêu cầu tạo prompt:
- Ngắn gọn nhưng đầy đủ chi tiết quan trọng
- Tập trung vào composition, colors, lighting, style
- Sử dụng từ khóa hiệu quả cho DALL-E
- Độ dài 200-300 từ
- Format: detailed description, art style, quality modifiers

Chỉ trả về prompt, không giải thích."""
        
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": prompt_generation_request
                }
            ],
            max_tokens=400
        )
        
        return response.choices[0].message.content.strip()
        
    except Exception as e:
        raise Exception(f"Cannot generate DALL-E prompt: {str(e)}")

def create_local_url(filepath, base_url="http://localhost:5000"):
    """Tạo URL local cho file ảnh"""
    if filepath and os.path.exists(filepath):
        filename = os.path.basename(filepath)
        return f"{base_url}/images/{filename}"
    return None

def generate_image(prompt, base64_image,n):
    """Sử dụng DALL-E để tạo ảnh từ prompt và hình ảnh tham chiếu"""
    client = openai.OpenAI(api_key=OPENAI_API_KEY)
    
    try:
        image_bytes = base64.b64decode(base64_image.split(',')[1])
        image_file = io.BytesIO(image_bytes)
        image_file.name = "reference_image.jpg"
        
        response = client.images.edit(
            model="gpt-image-1",
            image=image_file,  
            prompt=prompt,
            size="1024x1024",
            quality="auto",  
            n=n
        )
        
        base64_images = [img.b64_json for img in response.data if hasattr(img, 'b64_json')]
        return base64_images
        
    except Exception as e:
        raise Exception(f"Cannot generate image with DALL-E: {str(e)}")


# Ví dụ: validate_image_url
def validate_image_url(url):
    try:
        parsed = urlparse(url)
        if not parsed.scheme or not parsed.netloc:
            return False
        valid_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp']
        path = parsed.path.lower()
        has_valid_extension = any(path.endswith(ext) for ext in valid_extensions)
        trusted_domains = ['i.ibb.co', 'imgur.com', 'i.imgur.com', 'cdn.discordapp.com']
        is_trusted = any(domain in parsed.netloc for domain in trusted_domains)
        return has_valid_extension or is_trusted
    except:
        return False

# xử lý ideogram
def _call_ideogram(files_form):
    headers = {"Api-Key": IDEOGRAM_API_KEY}
    r = requests.post(IDEOGRAM_API_URL, headers=headers, files=files_form, timeout=300)
    r.raise_for_status()
    return r.json()

def _prepare_reference_files_from_urls(urls):
    """Tải tối đa 3 URL ảnh và đóng gói dạng multipart để gửi lên Ideogram."""
    refs = []
    for idx, u in enumerate(urls[:3]):
        resp = requests.get(u, timeout=180)
        resp.raise_for_status()
        content_type = resp.headers.get("Content-Type", "image/png")
        filename = f"ref_{idx}.png" if "png" in content_type else f"ref_{idx}.jpg"
        refs.append(("style_reference_images", (filename, io.BytesIO(resp.content), content_type)))
    return refs

# === API 1: Sinh prompt từ ảnh ===
@app.route('/gen_prompt', methods=['POST'])
def generate_prompt_api():
    data = request.get_json()
    image_url = data.get('image_url')
    style = data.get('style_type')

    if not image_url or not validate_image_url(image_url):
        return jsonify({"error": "Missing or invalid image_url"}), 400

    try:
        base64_image = download_image(image_url)
        if style == "2D":
            image_description = describe_image_with_gpt4o_2D(base64_image)
        else:
            image_description = describe_image_with_gpt4o_3D(base64_image)
        dalle_prompt = generate_dalle_prompt(image_description)
        return jsonify({
            "prompt": dalle_prompt
        }), 200
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

# === API 2: Tạo ảnh từ prompt and url===
@app.route('/generate_image', methods=['POST'])
def generate_image_api():
    data = request.get_json()
    prompt = data.get('prompt')
    image_url = data.get('image_url')
    n = data.get('image_count')

    if not prompt:
        return jsonify({"error": "Missing prompt"}), 400

    try:
        base64_image = None
        if image_url:
            if not validate_image_url(image_url):
                return jsonify({"error": "Invalid image_url"}), 400
            base64_image = download_image(image_url)

        base64_images = generate_image(prompt,base64_image,n)

        public_urls = []
        for b64 in base64_images:
            local_path = base64_to_image_file(b64)
            if local_path:
                gcs_url = upload_to_gcs(local_path)
                if gcs_url:
                    public_urls.append(gcs_url)
        return jsonify({
            "urls": public_urls,
        }), 200
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

# === generate_image_from_prompt ===
@app.route('/generate_image_from_prompt', methods=['POST'])
def generate_image_from_prompt():
    try:
        data = request.get_json()
        prompt = data.get("prompt")
        n = data.get('image_count')
        if not prompt:
            return jsonify({"error": "Missing prompt"}), 400

        client = openai.OpenAI(api_key=OPENAI_API_KEY)
        response = client.images.generate(
            model="gpt-image-1",
            prompt=prompt,
            size="1024x1024",
            quality="auto",
            n=n
        )
        base64_images = [img.b64_json for img in response.data if hasattr(img, 'b64_json')]
        public_urls = []
        for b64 in base64_images:
            local_path = base64_to_image_file(b64)
            if local_path:
                gcs_url = upload_to_gcs(local_path)
                if gcs_url:
                    public_urls.append(gcs_url)
        return jsonify({
            "urls": public_urls,
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500
# === Serve local images if needed ===
@app.route("/upload_cropped_image", methods=["POST"])
def upload_cropped_image():
    try:
        data = request.get_json()
        base64_image = data.get("image_base64", "")
        if "," in base64_image:
            base64_image = base64_image.split(",")[1]  # Remove prefix like data:image/png;base64,...
        local_path = base64_to_image_file(base64_image)
        public_url = upload_to_gcs(local_path)
        if public_url:
            return jsonify({"url": public_url})
        else:
            return jsonify({"error": "Upload failed"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/ideogram/generate", methods=["POST"])
def ideogram_generate():
    try:
        # ---- Đọc input (JSON hoặc multipart) ----
        DEFAULT_RENDERING_SPEED = "TURBO"        # tốc độ
        DEFAULT_ASPECT_RATIO   = "16x9"          # tỉ lệ
        DEFAULT_NEGATIVE_PROMPT = "no text, no watermark"
        image_reference_urls = []
        uploaded_files = []

        if request.is_json:
            data = request.get_json(force=True)
            prompt = data.get("prompt")
            num_images = data.get("num_images")
            image_reference_urls = data.get("image_references", []) or []
        else:
            prompt = request.form.get("prompt")
            num_images = request.form.get("image_count")
            uploaded_files = request.files.getlist("image_reference_images")

        if not prompt or not num_images:
            return jsonify({"error": "prompt and num_images are required"}), 400

        # ---- Build form-data gửi Ideogram ----
        files_list = [
            ("prompt", (None, prompt)),
            ("num_images", (None, str(num_images))),
            ("rendering_speed", (None, DEFAULT_RENDERING_SPEED)),
            ("aspect_ratio", (None, DEFAULT_ASPECT_RATIO)),
            ("negative_prompt", (None, DEFAULT_NEGATIVE_PROMPT)),
        ]

        # Nếu có reference (URL hoặc file) thì thêm, ngược lại thì không thêm gì
        if uploaded_files:
            for f in uploaded_files[:3]:
                files_list.append(("style_reference_images", (f.filename, f.stream, f.mimetype)))
        elif image_reference_urls:
            files_list.extend(_prepare_reference_files_from_urls(image_reference_urls))

        # ---- Gọi Ideogram ----
        ideogram_json = _call_ideogram(files_list)
        image_urls = [item.get("url") for item in ideogram_json.get("data", []) if item.get("url")]

        return jsonify({"images": image_urls})

    except requests.HTTPError as e:
        status = e.response.status_code if e.response is not None else 500
        try:
            detail = e.response.json()
        except Exception:
            detail = e.response.text if e.response is not None else str(e)
        return jsonify({"error": "Ideogram API error", "detail": detail}), status
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/gemini/generate", methods=["POST"])
def gemini_generate():
    client = genai.Client(api_key=GEMINI_API_KEY)
    try:
        if not request.is_json:
            return jsonify({"error": "Content-Type must be application/json"}), 400

        data = request.get_json(force=True)
        prompt = data.get("prompt")
        num_images = int(data.get("num_images", 1))

        # Bạn có thể gửi 1 trong 2 dạng:
        # 1) "image_references": ["https://...","https://..."]
        # 2) "image_url": "https://..." (đơn lẻ)
        image_reference_urls = data.get("image_references") or []
        if not image_reference_urls and data.get("image_url"):
            image_reference_urls = [data["image_url"]]

        if not prompt or num_images < 1:
            return jsonify({"error": "prompt and num_images are required"}), 400

        # Convert URL -> bytes/base64 cho Gemini
        references = []
        for u in image_reference_urls[:3]:
            try:
                references.append(references(u))
            except Exception as e:
                print("Download reference failed:", u, e)

        cfg = GenerateImagesConfig(number_of_images=num_images)

        # Một số phiên bản SDK chưa hỗ trợ 'references'.
        # Nếu bạn gặp lỗi TypeError, bỏ tham số 'references' đi.
        resp = client.images.generate(
            model=GEMINI_IMAGE_MODEL,
            prompt=prompt,
            config=cfg,
            references=references if references else None,
        )

        # Chuẩn hóa output thành data URL
        out = []
        for img in getattr(resp, "images", []):
            if hasattr(img, "bytes_base64") and img.bytes_base64:
                out.append(f"data:image/png;base64,{img.bytes_base64}")
            elif hasattr(img, "data") and img.data:
                b64 = base64.b64encode(img.data).decode("utf-8")
                out.append(f"data:image/png;base64,{b64}")

        return jsonify({"images": out})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
