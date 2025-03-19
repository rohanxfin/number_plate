import base64
import requests
import cv2
import numpy as np
import os

def test_numberplate_removal(image_paths, logo_path=None, endpoint_url="https://number-plate-app-341598202610.us-central1.run.app/numberplate-removal"):
    """
    Tests the /numberplate-removal endpoint.
    
    Parameters:
      image_paths (list): List of file paths for the images to be processed.
      logo_path (str): Optional file path for the custom logo.
      endpoint_url (str): URL of the endpoint.
    
    The function sends a JSON payload to the endpoint, receives the processed images,
    decodes them from base64, and displays them.
    """
    # Build the payload with images.
    images_payload = []
    for image_path in image_paths:
        if not os.path.exists(image_path):
            print(f"Image not found: {image_path}")
            continue
        with open(image_path, "rb") as f:
            image_data = f.read()
            image_b64 = base64.b64encode(image_data).decode("utf-8")
            filename = os.path.basename(image_path)
            images_payload.append({"filename": filename, "base64": image_b64})
    
    data = {"images": images_payload}

    print(data)
    
    # If a logo path is provided, add it to the payload.
    if logo_path:
        if not os.path.exists(logo_path):
            print(f"Logo not found: {logo_path}")
        else:
            with open(logo_path, "rb") as f:
                logo_data = f.read()
                logo_b64 = base64.b64encode(logo_data).decode("utf-8")
            data["logo"] = {"base64": logo_b64}


    
    # Send the POST request.
    response = requests.post(endpoint_url, json=data)
    if response.status_code != 200:
        print("Error:", response.text)
        return

    
    print(response.json())
    # Process the JSON response.
    results = response.json().get("images", [])


    if not results:
        print("No images returned.")
        return
    
    for result in results:
        filename = result.get("filename", "unknown")
        status = result.get("status", "unknown")
        print(f"Processing result for {filename} - Status: {status}")
        image_b64 = result.get("base64")
        if not image_b64:
            print(f"No image data returned for {filename}.")
            continue
        
        # Decode the base64 image.
        image_bytes = base64.b64decode(image_b64)
        np_img = np.frombuffer(image_bytes, dtype=np.uint8)
        image = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
        if image is None:
            print(f"Could not decode image for {filename}.")
            continue
        
        # Display the image.
        window_title = f"{filename} - {status}"
        cv2.imshow(window_title, image)
        cv2.waitKey(0)
        cv2.destroyWindow(window_title)

if __name__ == "__main__":
    # Example usage:
    # Replace these with the correct paths on your machine.
    image_paths = [
    r"C:\Users\rohan\OneDrive\Desktop\n1\1739522365_8104.jpg",
    r"C:\Users\rohan\OneDrive\Desktop\n1\1739522365_9357.jpg",
    r"C:\Users\rohan\OneDrive\Desktop\n1\1739522366_4658.jpg",
    r"C:\Users\rohan\OneDrive\Desktop\n1\1739522366_6165.jpg",
    r"C:\Users\rohan\OneDrive\Desktop\n1\1739522366_7087.jpg",
    r"C:\Users\rohan\OneDrive\Desktop\n1\1739522574_2322.jpg",
    r"C:\Users\rohan\OneDrive\Desktop\n1\1739522574_4130.jpg",
    r"C:\Users\rohan\OneDrive\Desktop\n1\1739522574_6457.jpg",
    r"C:\Users\rohan\OneDrive\Desktop\n1\1739522574_9944.jpg",
    r"C:\Users\rohan\OneDrive\Desktop\n1\1739522575_2758.jpg",
    r"C:\Users\rohan\OneDrive\Desktop\n1\1739522707_2758.jpg",
    r"C:\Users\rohan\OneDrive\Desktop\n1\1739522707_5367.jpg",
    r"C:\Users\rohan\OneDrive\Desktop\n1\1739522707_6738.jpg",
    r"C:\Users\rohan\OneDrive\Desktop\n1\1739522708_7303.jpg",
    r"C:\Users\rohan\OneDrive\Desktop\n1\1739522708_8120.jpg"
    ]

    logo_path = r"C:\Users\rohan\OneDrive\Desktop\swiggy-download.jpg"  # Optional; set to None if not needed.
    
    test_numberplate_removal(image_paths, logo_path)
