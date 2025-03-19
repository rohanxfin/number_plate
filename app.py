import os
import io
import cv2
import numpy as np
import torch
import logging
import concurrent.futures
import base64
from flask import Flask, request, jsonify, Response
from flask_cors import CORS
from ultralytics import YOLO
import time

# ----------------------------
# Setup Logging and Device
# ----------------------------
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Using device: {device}")

# ----------------------------
# Global Model Initialization
# ----------------------------
NUMPLATE_MODEL_PATH = "number_plate_blur.pt"  # Update with your model path
DEFAULT_LOGO_PATH = "logo.png"                 # Update with your default logo path

try:
    numplate_yolo_model = YOLO(NUMPLATE_MODEL_PATH)
    numplate_yolo_model.to(device)
    logger.info("YOLO model loaded successfully.")
except Exception as e:
    logger.error("Failed to load YOLO model: " + str(e))
    raise e

# ----------------------------
# Helper Functions
# ----------------------------
def order_points(pts):
    """
    Order the four points in the order: top-left, top-right, bottom-right, bottom-left.
    """
    try:
        x_sorted = pts[np.argsort(pts[:, 0]), :]
        left_most = x_sorted[:2, :]
        right_most = x_sorted[2:, :]
        left_most = left_most[np.argsort(left_most[:, 1]), :]
        (tl, bl) = left_most
        right_most = right_most[np.argsort(right_most[:, 1]), :]
        (tr, br) = right_most
        return np.array([tl, tr, br, bl], dtype=np.float32)
    except Exception as e:
        logger.error("Error ordering points: " + str(e))
        raise e

def process_image(image_file, custom_logo_file=None):
    """
    Process a single image:
      - Detect the number plate using YOLO.
      - Overlay the logo (custom if provided, otherwise default) onto the detected region.
      - Return the JPEG-encoded result.
    """
    try:
        file_bytes = image_file.read()
        if not file_bytes:
            raise ValueError("No data found in the image file.")
        np_img = np.frombuffer(file_bytes, np.uint8)
        original_image = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
        if original_image is None:
            raise ValueError("Invalid image provided or unable to decode.")
    except Exception as e:
        logger.error("Error reading the image file: " + str(e))
        raise e

    try:
        # Detect the number plate using YOLO.
        results = numplate_yolo_model.predict(source=original_image, conf=0.05)
        if not results or len(results) == 0:
            raise Exception("No detection results returned from model.")
        # Extract bounding box data (update index if needed).
        tensor_data = results[0].obb.xyxyxyxy[0].numpy()
    except Exception as e:
        logger.error("Error during YOLO prediction: " + str(e))
        raise Exception("Error during prediction: " + str(e))
    
    try:
        # Order the quadrilateral points.
        quad_points = np.array(tensor_data, dtype=np.float32)
        ordered_quad = order_points(quad_points)
        x_coords = ordered_quad[:, 0]
        y_coords = ordered_quad[:, 1]
        w = int(np.max(x_coords) - np.min(x_coords))
        h = int(np.max(y_coords) - np.min(y_coords))
        if w <= 0 or h <= 0:
            raise ValueError("Invalid bounding box dimensions.")
    except Exception as e:
        logger.error("Error processing bounding box: " + str(e))
        raise e

    try:
        # Load the logo: use custom if provided; otherwise use the default.
        if custom_logo_file:
            logo_bytes = custom_logo_file.read()
            if not logo_bytes:
                raise ValueError("Custom logo file is empty.")
            logo_np = np.frombuffer(logo_bytes, np.uint8)
            logo = cv2.imdecode(logo_np, cv2.IMREAD_UNCHANGED)
            if logo is None:
                raise ValueError("Invalid custom logo provided.")
        else:
            if not os.path.exists(DEFAULT_LOGO_PATH):
                raise ValueError("Default logo not found at path: " + DEFAULT_LOGO_PATH)
            logo = cv2.imread(DEFAULT_LOGO_PATH, cv2.IMREAD_UNCHANGED)
            if logo is None:
                raise ValueError("Failed to load the default logo image.")
    except Exception as e:
        logger.error("Error loading logo: " + str(e))
        raise e

    try:
        # Resize the logo to fit the bounding box while maintaining aspect ratio.
        logo_height, logo_width = logo.shape[:2]
        scale = min(w / logo_width, h / logo_height)
        new_width = max(1, int(logo_width * scale))
        new_height = max(1, int(logo_height * scale))
        resized_logo = cv2.resize(logo, (new_width, new_height), interpolation=cv2.INTER_AREA)
    except Exception as e:
        logger.error("Error resizing logo: " + str(e))
        raise e

    try:
        # Ensure the logo image has an alpha channel.
        if resized_logo.shape[2] == 3:
            resized_logo = cv2.cvtColor(resized_logo, cv2.COLOR_BGR2BGRA)
    except Exception as e:
        logger.error("Error processing logo alpha channel: " + str(e))
        raise e

    try:
        # Create a white background (RGBA) for the bounding box.
        white_image = np.zeros((h, w, 4), dtype=np.uint8)
        white_image[:, :] = [255, 255, 255, 255]
        # Center the resized logo on the white background.
        x_center = (w - new_width) // 2
        y_center = (h - new_height) // 2
        white_image[y_center:y_center+new_height, x_center:x_center+new_width] = resized_logo
    except Exception as e:
        logger.error("Error preparing the logo overlay: " + str(e))
        raise e

    try:
        # Compute perspective transform and warp the logo.
        src_pts = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
        M = cv2.getPerspectiveTransform(src_pts, ordered_quad)
        warped_logo = cv2.warpPerspective(white_image, M, (original_image.shape[1], original_image.shape[0]))
    except Exception as e:
        logger.error("Error during perspective transform: " + str(e))
        raise e

    try:
        # Blend the warped logo onto the original image.
        warped_alpha = warped_logo[:, :, 3]
        original_image_rgba = cv2.cvtColor(original_image, cv2.COLOR_BGR2BGRA)
        mask = warped_alpha > 0
        original_image_rgba[mask] = warped_logo[mask]
        result_image = cv2.cvtColor(original_image_rgba, cv2.COLOR_BGRA2BGR)
    except Exception as e:
        logger.error("Error blending logo with original image: " + str(e))
        raise e

    try:
        # Encode the result as a JPEG image.
        success, encoded_image = cv2.imencode('.jpg', result_image)
        if not success:
            raise Exception("Image encoding failed.")
    except Exception as e:
        logger.error("Error encoding the final image: " + str(e))
        raise e

    return encoded_image.tobytes()

def worker_process_image(args):
    """
    Worker function for processing images in the /numberplate-removal endpoint.
    Returns a tuple (filename, base64 encoded image, status).
    If processing fails, the original image is returned with status "failed".
    """
    filename, image_bytes, logo_bytes = args
    try:
        processed_bytes = process_image(io.BytesIO(image_bytes), io.BytesIO(logo_bytes) if logo_bytes else None)
        processed_b64 = base64.b64encode(processed_bytes).decode('utf-8')
        return (filename, processed_b64, "success")
    except Exception as e:
        logger.error(f"Error processing image {filename}: " + str(e))
        # Return the original image (in base64) if processing fails.
        original_b64 = base64.b64encode(image_bytes).decode('utf-8')
        return (filename, original_b64, "failed")

# ----------------------------
# Flask App and Endpoints
# ----------------------------
app = Flask(__name__)
CORS(app)

@app.route("/", methods=["GET"])
def index():
    try:
        return "application number plater removal is up and running", 200
    except Exception as e:
        logger.error("Error in root endpoint: " + str(e))
        return jsonify({"error": "Internal server error"}), 500

@app.route("/test", methods=["POST"])
def test_endpoint():
    try:
        image_file = request.files.get("images")
        if not image_file:
            return jsonify({"error": "No image file provided"}), 400
        custom_logo_file = request.files.get("logo") if request.files.get("logo") and request.files.get("logo").filename != "" else None
        processed_bytes = process_image(image_file, custom_logo_file)
        return Response(processed_bytes, mimetype='image/jpeg')
    except Exception as e:
        logger.error("Error in /test endpoint: " + str(e))
        return jsonify({"error": str(e)}), 500

@app.route("/numberplate-removal", methods=["POST"])
def numberplate_removal():
    try:
        data = request.get_json(force=True)
        if not data or "images" not in data:
            return jsonify({"error": "Invalid JSON data. 'images' key missing."}), 400
        
        images_data = data["images"]
        logo_bytes = None
        if "logo" in data and data["logo"].get("base64"):
            try:
                logo_bytes = base64.b64decode(data["logo"]["base64"])
            except Exception as e:
                logger.error("Invalid logo base64 data: " + str(e))
                return jsonify({"error": "Invalid logo base64 data."}), 400
        
        tasks = []
        for img in images_data:
            if "filename" not in img or "base64" not in img:
                logger.warning("Skipping an image entry due to missing keys.")
                continue
            try:
                image_bytes = base64.b64decode(img["base64"])
            except Exception as e:
                logger.error(f"Invalid base64 data for {img.get('filename', 'unknown')}: " + str(e))
                continue
            tasks.append((img["filename"], image_bytes, logo_bytes))
        
        if not tasks:
            return jsonify({"error": "No valid images to process."}), 400
        
        results = []
        chunk_size = 3  # Process in chunks of 3.
        for i in range(0, len(tasks), chunk_size):
            chunk = tasks[i:i+chunk_size]
            with concurrent.futures.ProcessPoolExecutor(max_workers=2) as executor:
                futures = [executor.submit(worker_process_image, task) for task in chunk]
                for future in concurrent.futures.as_completed(futures):
                    filename, b64_result, status = future.result()
                    results.append({
                        "filename": filename,
                        "base64": b64_result,
                        "status": status
                    })
            time.sleep(2)  # Increased delay between chunks.
        
        return jsonify({"images": results})
    except Exception as e:
        logger.error("Error in /numberplate-removal endpoint: " + str(e))
        return jsonify({"error": "Internal server error: " + str(e)}), 500

if __name__ == "__main__":
    try:
        app.run(host="0.0.0.0", port=8080)
    except Exception as e:
        logger.error("Failed to start the Flask app: " + str(e))



# import os
# import io
# import cv2
# import numpy as np
# import torch
# import logging
# import concurrent.futures
# import base64
# from flask import Flask, request, jsonify
# from flask_cors import CORS
# from ultralytics import YOLO
# import time

# # ----------------------------
# # Setup Logging and Device
# # ----------------------------
# logger = logging.getLogger(__name__)
# logging.basicConfig(level=logging.INFO)
# device = "cuda" if torch.cuda.is_available() else "cpu"
# logger.info(f"Using device: {device}")

# # ----------------------------
# # Global Model Initialization
# # ----------------------------
# NUMPLATE_MODEL_PATH = "number_plate_blur.pt"  # Update with your model path
# DEFAULT_LOGO_PATH = "logo.png"                 # Update with your default logo path
# numplate_yolo_model = YOLO(NUMPLATE_MODEL_PATH)
# numplate_yolo_model.to(device)

# # ----------------------------
# # Helper Functions
# # ----------------------------
# def order_points(pts):
#     """
#     Order the four points in the order: top-left, top-right, bottom-right, bottom-left.
#     """
#     x_sorted = pts[np.argsort(pts[:, 0]), :]
#     left_most = x_sorted[:2, :]
#     right_most = x_sorted[2:, :]
#     left_most = left_most[np.argsort(left_most[:, 1]), :]
#     (tl, bl) = left_most
#     right_most = right_most[np.argsort(right_most[:, 1]), :]
#     (tr, br) = right_most
#     return np.array([tl, tr, br, bl], dtype=np.float32)

# def process_image(image_file, custom_logo_file=None):
#     """
#     Process a single image:
#       - Detect the number plate quadrilateral using YOLO.
#       - Overlay the logo (custom if provided, otherwise default) onto the detected region.
#       - Return the JPEG-encoded result.
#     """
#     # Read and decode the input image
#     file_bytes = image_file.read()
#     np_img = np.frombuffer(file_bytes, np.uint8)
#     original_image = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
#     if original_image is None:
#         raise ValueError("Invalid image provided.")
    
#     # Detect the number plate using YOLO
#     try:
#         results = numplate_yolo_model.predict(source=original_image, conf=0.05)
#         tensor_data = results[0].obb.xyxyxyxy[0].numpy()
#     except Exception as e:
#         raise Exception("Error during prediction: " + str(e))
    
#     # Order the quadrilateral points
#     quad_points = np.array(tensor_data, dtype=np.float32)
#     ordered_quad = order_points(quad_points)
#     x_coords = ordered_quad[:, 0]
#     y_coords = ordered_quad[:, 1]
#     w = int(np.max(x_coords) - np.min(x_coords))
#     h = int(np.max(y_coords) - np.min(y_coords))
    
#     # Load the logo: custom if provided; otherwise the default logo
#     if custom_logo_file:
#         logo_bytes = custom_logo_file.read()
#         logo_np = np.frombuffer(logo_bytes, np.uint8)
#         logo = cv2.imdecode(logo_np, cv2.IMREAD_UNCHANGED)
#         if logo is None:
#             raise ValueError("Invalid custom logo provided.")
#     else:
#         logo = cv2.imread(DEFAULT_LOGO_PATH, cv2.IMREAD_UNCHANGED)
#         if logo is None:
#             raise ValueError("Default logo not found.")
    
#     # Resize logo to fit the bounding box while maintaining aspect ratio
#     logo_height, logo_width = logo.shape[:2]
#     scale = min(w / logo_width, h / logo_height)
#     new_width = int(logo_width * scale)
#     new_height = int(logo_height * scale)
#     resized_logo = cv2.resize(logo, (new_width, new_height), interpolation=cv2.INTER_AREA)
    
#     # Ensure the logo image has an alpha channel
#     if resized_logo.shape[2] == 3:
#         resized_logo = cv2.cvtColor(resized_logo, cv2.COLOR_BGR2BGRA)
    
#     # Create a white background (RGBA) with the same size as the bounding box
#     white_image = np.zeros((h, w, 4), dtype=np.uint8)
#     white_image[:, :] = [255, 255, 255, 255]
    
#     # Center the resized logo on the white background
#     x_center = (w - new_width) // 2
#     y_center = (h - new_height) // 2
#     white_image[y_center:y_center+new_height, x_center:x_center+new_width] = resized_logo
    
#     # Define source points and compute the perspective transform
#     src_pts = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
#     M = cv2.getPerspectiveTransform(src_pts, ordered_quad)
#     warped_logo = cv2.warpPerspective(white_image, M, (original_image.shape[1], original_image.shape[0]))
    
#     # Blend the warped logo onto the original image
#     warped_alpha = warped_logo[:, :, 3]
#     original_image_rgba = cv2.cvtColor(original_image, cv2.COLOR_BGR2BGRA)
#     mask = warped_alpha > 0
#     original_image_rgba[mask] = warped_logo[mask]
#     result_image = cv2.cvtColor(original_image_rgba, cv2.COLOR_BGRA2BGR)
    
#     # Encode the result as a JPEG image
#     success, encoded_image = cv2.imencode('.jpg', result_image)
#     if not success:
#         raise Exception("Image encoding failed.")
#     return encoded_image.tobytes()

# # ----------------------------
# # Flask App and Endpoints
# # ----------------------------
# app = Flask(__name__)
# CORS(app)

# @app.route("/remove-number-pate", methods=["POST"])
# def process_images():
#     images = request.files.getlist("images")
#     if not images:
#         return jsonify({"error": "No image files provided"}), 400

#     custom_logo_file = request.files.get("logo")
#     custom_logo_bytes = custom_logo_file.read() if custom_logo_file and custom_logo_file.filename != "" else None

#     results = {}
    
#     def process_single(image_file):
#         try:
#             image_bytes = image_file.read()
#             image_io = io.BytesIO(image_bytes)
#             logo_io = io.BytesIO(custom_logo_bytes) if custom_logo_bytes else None
#             processed = process_image(image_io, logo_io)
#             # Encode the processed image to a base64 string for transmission in JSON
#             encoded = base64.b64encode(processed).decode('utf-8')
#             return (image_file.filename, encoded)
#         except Exception as e:
#             return (image_file.filename, f"Error: {str(e)}")
    
#     # Split images into chunks of 5
#     chunks = [images[i:i + 5] for i in range(0, len(images), 5)]
    
#     results = {}

#     # Process images in sets of 5
#     for chunk in chunks:
#         with concurrent.futures.ThreadPoolExecutor() as executor:
#             future_to_file = {executor.submit(process_single, img): img for img in chunk}
#             for future in concurrent.futures.as_completed(future_to_file):
#                 filename, result_data = future.result()
#                 results[filename] = result_data
#         time.sleep(1)  # Add a small delay between chunks to prevent CPU overload

#     return jsonify(results)

# if __name__ == "__main__":
#     app.run(host="0.0.0.0", port=8080)






# import os
# import io
# import cv2
# import numpy as np
# import torch
# import logging
# import concurrent.futures
# import base64
# from flask import Flask, request, jsonify, send_file
# from flask_cors import CORS
# from ultralytics import YOLO

# # ----------------------------
# # Setup Logging and Device
# # ----------------------------
# logger = logging.getLogger(__name__)
# logging.basicConfig(level=logging.INFO)
# device = "cuda" if torch.cuda.is_available() else "cpu"
# logger.info(f"Using device: {device}")

# # ----------------------------
# # Global Model Initialization
# # ----------------------------
# NUMPLATE_MODEL_PATH = "number_plate_blur.pt"  # Update with your model path
# DEFAULT_LOGO_PATH = "logo.png"                 # Update with your default logo path
# numplate_yolo_model = YOLO(NUMPLATE_MODEL_PATH)
# numplate_yolo_model.to(device)

# # ----------------------------
# # Helper Functions
# # ----------------------------
# def order_points(pts):
#     """
#     Order the four points in the order: top-left, top-right, bottom-right, bottom-left.
#     """
#     x_sorted = pts[np.argsort(pts[:, 0]), :]
#     left_most = x_sorted[:2, :]
#     right_most = x_sorted[2:, :]
#     left_most = left_most[np.argsort(left_most[:, 1]), :]
#     (tl, bl) = left_most
#     right_most = right_most[np.argsort(right_most[:, 1]), :]
#     (tr, br) = right_most
#     return np.array([tl, tr, br, bl], dtype=np.float32)

# def process_image(image_file, custom_logo_file=None):
#     """
#     Process a single image:
#       - Detect the number plate quadrilateral using YOLO.
#       - Overlay the logo (custom if provided, otherwise default) onto the detected region.
#       - Return the JPEG-encoded result.
#     """
#     # Read and decode the input image
#     file_bytes = image_file.read()
#     np_img = np.frombuffer(file_bytes, np.uint8)
#     original_image = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
#     if original_image is None:
#         raise ValueError("Invalid image provided.")
    
#     # Detect the number plate using YOLO
#     try:
#         results = numplate_yolo_model.predict(source=original_image, conf=0.05)
#         tensor_data = results[0].obb.xyxyxyxy[0].numpy()
#     except Exception as e:
#         raise Exception("Error during prediction: " + str(e))
    
#     # Order the quadrilateral points
#     quad_points = np.array(tensor_data, dtype=np.float32)
#     ordered_quad = order_points(quad_points)
#     x_coords = ordered_quad[:, 0]
#     y_coords = ordered_quad[:, 1]
#     w = int(np.max(x_coords) - np.min(x_coords))
#     h = int(np.max(y_coords) - np.min(y_coords))
    
#     # Load the logo: custom if provided; otherwise the default logo
#     if custom_logo_file:
#         logo_bytes = custom_logo_file.read()
#         logo_np = np.frombuffer(logo_bytes, np.uint8)
#         logo = cv2.imdecode(logo_np, cv2.IMREAD_UNCHANGED)
#         if logo is None:
#             raise ValueError("Invalid custom logo provided.")
#     else:
#         logo = cv2.imread(DEFAULT_LOGO_PATH, cv2.IMREAD_UNCHANGED)
#         if logo is None:
#             raise ValueError("Default logo not found.")
    
#     # Resize logo to fit the bounding box while maintaining aspect ratio
#     logo_height, logo_width = logo.shape[:2]
#     scale = min(w / logo_width, h / logo_height)
#     new_width = int(logo_width * scale)
#     new_height = int(logo_height * scale)
#     resized_logo = cv2.resize(logo, (new_width, new_height), interpolation=cv2.INTER_AREA)
    
#     # Ensure the logo image has an alpha channel
#     if resized_logo.shape[2] == 3:
#         resized_logo = cv2.cvtColor(resized_logo, cv2.COLOR_BGR2BGRA)
    
#     # Create a white background (RGBA) with the same size as the bounding box
#     white_image = np.zeros((h, w, 4), dtype=np.uint8)
#     white_image[:, :] = [255, 255, 255, 255]
    
#     # Center the resized logo on the white background
#     x_center = (w - new_width) // 2
#     y_center = (h - new_height) // 2
#     white_image[y_center:y_center+new_height, x_center:x_center+new_width] = resized_logo
    
#     # Define source points and compute the perspective transform
#     src_pts = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
#     M = cv2.getPerspectiveTransform(src_pts, ordered_quad)
#     warped_logo = cv2.warpPerspective(white_image, M, (original_image.shape[1], original_image.shape[0]))
    
#     # Blend the warped logo onto the original image
#     warped_alpha = warped_logo[:, :, 3]
#     original_image_rgba = cv2.cvtColor(original_image, cv2.COLOR_BGR2BGRA)
#     mask = warped_alpha > 0
#     original_image_rgba[mask] = warped_logo[mask]
#     result_image = cv2.cvtColor(original_image_rgba, cv2.COLOR_BGRA2BGR)
    
#     # Encode the result as a JPEG image
#     success, encoded_image = cv2.imencode('.jpg', result_image)
#     if not success:
#         raise Exception("Image encoding failed.")
#     return encoded_image.tobytes()

# # ----------------------------
# # Flask App and Endpoints
# # ----------------------------
# app = Flask(__name__)
# CORS(app)

# @app.route("/")
# def home():
#     return "Welcome to the number plate processing API!"

# # Normal endpoint: Processes a single image with an optional logo
# @app.route("/number-plate-removal", methods=["POST"])
# def number_plate_removal():
#     if "image" not in request.files:
#         return jsonify({"error": "No image file provided"}), 400
#     image_file = request.files["image"]
#     if image_file.filename == "":
#         return jsonify({"error": "No file selected for image"}), 400

#     custom_logo_file = request.files.get("logo")
#     try:
#         processed_image_bytes = process_image(image_file, custom_logo_file)
#     except Exception as e:
#         return jsonify({"error": str(e)}), 500

#     return send_file(io.BytesIO(processed_image_bytes), mimetype="image/jpeg",
#                      as_attachment=True, download_name=f"processed_{image_file.filename}")

# # Bulk endpoint: Processes multiple images and returns a JSON mapping the original file names
# # to the processed image (base64 encoded) or an error message.
# @app.route("/number-plate-removal/bulk", methods=["POST"])
# def bulk_number_plate_removal():
#     images = request.files.getlist("images")
#     if not images:
#         return jsonify({"error": "No image files provided"}), 400

#     custom_logo_file = request.files.get("logo")
#     custom_logo_bytes = custom_logo_file.read() if custom_logo_file and custom_logo_file.filename != "" else None

#     results = {}

#     def process_single(image_file):
#         try:
#             image_bytes = image_file.read()
#             image_io = io.BytesIO(image_bytes)
#             logo_io = io.BytesIO(custom_logo_bytes) if custom_logo_bytes else None
#             processed = process_image(image_io, logo_io)
#             # Encode the processed image to a base64 string for transmission in JSON
#             encoded = base64.b64encode(processed).decode('utf-8')
#             return (image_file.filename, encoded)
#         except Exception as e:
#             return (image_file.filename, f"Error: {str(e)}")

#     with concurrent.futures.ThreadPoolExecutor() as executor:
#         future_to_file = {executor.submit(process_single, img): img for img in images}
#         for future in concurrent.futures.as_completed(future_to_file):
#             filename, result_data = future.result()
#             results[filename] = result_data

#     return jsonify(results)

# if __name__ == "__main__":
#     app.run(host="0.0.0.0", port=8080)





# import os
# import io
# import cv2
# import numpy as np
# import torch
# import logging
# from flask import Flask, request, jsonify, send_file
# from flask_cors import CORS
# from ultralytics import YOLO
# from PIL import Image

# # Set up logging
# logger = logging.getLogger(__name__)
# logging.basicConfig(level=logging.INFO)

# # Decide device based on CUDA availability
# device = "cuda" if torch.cuda.is_available() else "cpu"
# logger.info(f"Using device: {device}")

# # ----------------------------
# # Global model initializations
# # ----------------------------

# # For number-plate removal
# NUMPLATE_MODEL_PATH = "number_plate_blur.pt"  # Update with your number-plate detection model
# LOGO_PATH = "logo.png"                         # Update with your logo image path
# numplate_yolo_model = YOLO(NUMPLATE_MODEL_PATH)
# numplate_yolo_model.to(device)

# # ----------------------------
# # Helper Functions
# # ----------------------------

# def order_points(pts):
#     """
#     Order the four points in the order: top-left, top-right, bottom-right, bottom-left.
#     """
#     x_sorted = pts[np.argsort(pts[:, 0]), :]
#     left_most = x_sorted[:2, :]
#     right_most = x_sorted[2:, :]
#     left_most = left_most[np.argsort(left_most[:, 1]), :]
#     (tl, bl) = left_most
#     right_most = right_most[np.argsort(right_most[:, 1]), :]
#     (tr, br) = right_most
#     return np.array([tl, tr, br, bl], dtype=np.float32)

# app = Flask(__name__)
# CORS(app)


# @app.route("/")
# def home():
#     return "Welcome to the number plate processing API!"

# @app.route("/number-plate-removal", methods=["POST"])
# def number_plate_removal():
#     if "image" not in request.files:
#         return jsonify({"error": "No image file provided"}), 400
#     file = request.files["image"]
#     if file.filename == "":
#         return jsonify({"error": "No file selected"}), 400

#     original_filename = file.filename

#     # Read the uploaded image from the request into a NumPy array
#     file_bytes = file.read()
#     np_img = np.frombuffer(file_bytes, np.uint8)
#     original_image = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
#     if original_image is None:
#         return jsonify({"error": "Invalid image"}), 400

#     try:
#         # Predict quadrilateral points using your model (ensure model is defined elsewhere)
#         # The model is expected to return a structure where you can access .obb.xyxyxyxy[0].numpy()
#         tensor_data = numplate_yolo_model.predict(source=original_image, conf=0.05)[0].obb.xyxyxyxy[0].numpy()
#     except Exception as e:
#         return jsonify({"error": f"Error during prediction: {str(e)}"}), 500

#     # Convert to NumPy array with float32 type and reorder points
#     quad_points = np.array(tensor_data, dtype=np.float32)
#     ordered_quad = order_points(quad_points)

#     # Compute bounding box of the ordered quadrilateral
#     x_coords = ordered_quad[:, 0]
#     y_coords = ordered_quad[:, 1]
#     w = int(np.max(x_coords) - np.min(x_coords))
#     h = int(np.max(y_coords) - np.min(y_coords))

#     # Load the logo image with alpha channel from a fixed path
#     logo_path = "logo.png"
#     logo = cv2.imread(logo_path, cv2.IMREAD_UNCHANGED)
#     if logo is None:
#         return jsonify({"error": "Logo image not found"}), 500

#     # Resize logo to fit within the bounding box while maintaining aspect ratio
#     logo_height, logo_width = logo.shape[:2]
#     scale = min(w / logo_width, h / logo_height)
#     new_width = int(logo_width * scale)
#     new_height = int(logo_height * scale)
#     resized_logo = cv2.resize(logo, (new_width, new_height), interpolation=cv2.INTER_AREA)

#     # Ensure the logo image has an alpha channel
#     if resized_logo.shape[2] == 3:
#         resized_logo = cv2.cvtColor(resized_logo, cv2.COLOR_BGR2BGRA)

#     # Create a white background with the same size as the bounding box (RGBA)
#     white_image = np.zeros((h, w, 4), dtype=np.uint8)
#     white_image[:, :] = [255, 255, 255, 255]

#     # Center the resized logo on the white background
#     x_center = (w - new_width) // 2
#     y_center = (h - new_height) // 2
#     white_image[y_center:y_center+new_height, x_center:x_center+new_width] = resized_logo

#     # Define source points (corners of the white image)
#     src_pts = np.array([
#         [0, 0],
#         [w, 0],
#         [w, h],
#         [0, h]
#     ], dtype=np.float32)

#     # Compute the perspective transform matrix from the white image to the quadrilateral
#     M = cv2.getPerspectiveTransform(src_pts, ordered_quad)
#     warped_logo = cv2.warpPerspective(white_image, M, (original_image.shape[1], original_image.shape[0]))

#     # Blend the warped logo onto the original image
#     warped_alpha = warped_logo[:, :, 3]
#     original_image_rgba = cv2.cvtColor(original_image, cv2.COLOR_BGR2BGRA)
#     mask = warped_alpha > 0
#     original_image_rgba[mask] = warped_logo[mask]
#     result_image = cv2.cvtColor(original_image_rgba, cv2.COLOR_BGRA2BGR)

#     # Encode the result image to JPEG in memory
#     success, encoded_image = cv2.imencode('.jpg', result_image)
#     if not success:
#         return jsonify({"error": "Image encoding failed"}), 500
#     io_buf = io.BytesIO(encoded_image.tobytes())
#     io_buf.seek(0)

#     return send_file(io_buf, mimetype="image/jpeg", as_attachment=True,
#                      download_name=f"processed_{original_filename}")


# if __name__ == "__main__":
#     app.run(host='0.0.0.0', port=8080)

