===== Testing locally ======

docker build -t my-olx-app .    
docker run -p 8080:8080 my-olx-app

===== Pushing to GCR ======

docker build -t us-central1-docker.pkg.dev/ai-projects-453411/number-plate-repo/number-plate-app .                                                                                                                                    
docker push us-central1-docker.pkg.dev/ai-projects-453411/number-plate-repo/number-plate-app                                                                                                                                          
 gcloud run deploy number-plate-app --image us-central1-docker.pkg.dev/ai-projects-453411/number-plate-repo/number-plate-app --platform managed --region us-central1 --allow-unauthenticated --memory 8Gi --cpu 2 --timeout 300 --execution-environment=gen2 --max-instances=2

======== Sample payload ==========

Below are the input and output payload formats for each endpoint for your reference:

============= URL =================

https://number-plate-app-341598202610.us-central1.run.app

====================================

### 1. Root Endpoint (`/`)

- **Method:** GET  
- **Input:**  
  - *None* (simple GET request with no payload)

- **Output:**  
  - **Content-Type:** `text/plain`  
  - **Body:**  
    ```
    application number plater removal is up and running
    ```

---

### 2. Test Endpoint (`/test`)

- **Method:** POST  
- **Input:**  
  - **Content-Type:** `multipart/form-data`  
  - **Fields:**  
    - `images`:  
      - **Type:** File upload (any image format)  
      - **Description:** The image to be processed.  
    - `logo` *(Optional)*:  
      - **Type:** File upload (any image format)  
      - **Description:** Custom logo to overlay (if not provided, the default logo is used).

- **Output:**  
  - **Content-Type:** `image/jpeg`  
  - **Body:**  
    - Binary JPEG image representing the processed image.

---

### 3. Numberplate Removal Endpoint (`/numberplate-removal`)

- **Method:** POST  
- **Input:**  
  - **Content-Type:** `application/json`  
  - **JSON Format Example:**
    ```json
    {
      "images": [
        {
          "filename": "example1.jpg",
          "base64": "base64_encoded_string_of_image1"
        },
        {
          "filename": "example2.jpg",
          "base64": "base64_encoded_string_of_image2"
        }
      ],
      "logo": { 
        "base64": "base64_encoded_string_of_logo"  // Optional
      }
    }
    ```
  - **Description:**  
    - The `images` key contains an array of image objects where each object must include:
      - `filename`: The name of the image file.
      - `base64`: The image content encoded in base64.
    - The `logo` key is optional and should contain:
      - `base64`: The logo image content encoded in base64.

- **Output:**  
  - **Content-Type:** `application/json`  
  - **JSON Format Example:**
    ```json
    {
      "images": [
        {
          "filename": "example1.jpg",
          "base64": "base64_encoded_string_of_processed_image1",
          "status": "success"
        },
        {
          "filename": "example2.jpg",
          "base64": "base64_encoded_string_of_original_image2",
          "status": "failed"
        }
      ]
    }
    ```
  - **Description:**  
    - The response includes an `images` array where each image object contains:
      - `filename`: The name of the processed image.
      - `base64`: The processed image content in base64 format (or the original image if processing failed).
      - `status`: Indicates `"success"` if processing was successful, or `"failed"` if an error occurred.

