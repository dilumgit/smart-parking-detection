# Smart Parking Detection AI

An AI-powered smart parking detection system that uses YOLO11n for vehicle detection and OpenCV-based spatial analysis to determine parking-space occupancy. The system provides a FastAPI REST backend and an interactive Streamlit interface for parking configuration, image analysis, and occupancy visualization.

---

## 1. Overview

**Smart Parking Detection AI** is an intelligent parking lot monitoring and vehicle detection solution designed for fixed camera setups. It bridges modern deep-learning object detection with classic computational geometry:

1. **Pretrained Object Detection**: Leverages **Ultralytics YOLO11n** to detect vehicles in parking lot imagery.
2. **Geometric Occupancy Mapping**: Maps detected vehicle centroids against predefined 4-corner polygon slot boundaries using OpenCV point-in-polygon tests.
3. **Decoupled Architecture**: Features a robust **FastAPI** REST API backend for scalable inference and an interactive **Streamlit** dashboard for live visual configuration and occupancy tracking.

---

## 2. Features

- **Vehicle Detection**: Accurately detects common vehicle categories (`car`, `motorcycle`, `bus`, `truck`) using a pretrained YOLO11n neural network.
- **Custom Parking Layouts**: Configure, label, and persist multiple named parking areas with 4-corner polygon coordinates per spot.
- **Interactive Web Annotator**: Mark parking spot corners directly onto reference imagery inside the Streamlit dashboard via an embedded HTML5 Canvas component.
- **Occupancy Analysis Pipeline**: Automatically determines whether each slot is `OCCUPIED` or `EMPTY` based on vehicle bounding box centers.
- **Occupancy Metrics**: Computes lot-wide metrics including total spaces, occupied count, available count, and overall occupancy percentage.
- **RESTful API**: Full-featured FastAPI service exposing endpoints for health monitoring, parking layout management, and multipart image analysis.
- **Automated Test Suite**: Comprehensive unit and integration test suite with 54 passing tests covering schemas, geometry, detector, API endpoints, and UI state.

---

## 3. System Architecture

```text
User / Fixed Camera Still
           │
           ▼
   Streamlit Frontend (Port 8501)
   ├── Mode 1: Check Parking Availability
   └── Mode 2: Interactive Parking Area Setup
           │
           │ HTTP REST Requests
           ▼
    FastAPI REST Backend (Port 8000)
    ├── GET  /health
    ├── GET  /api/v1/parking/areas
    ├── POST /api/v1/parking/areas
    └── POST /api/v1/parking/analyze
           │
           ├──► [ai.detector.VehicleDetector]
           │        └── YOLO11n Inference (Ultralytics)
           │        └── Target Classes: car, bus, truck, motorcycle
           │        └── Bounding Boxes: [x1, y1, x2, y2]
           │
           └──► [ai.parking_analyzer.ParkingAnalyzer]
                    └── Calculate Vehicle Centroids: ((x1+x2)/2, (y1+y2)/2)
                    └── Load Parking Slot Polygons (JSON)
                    └── cv2.pointPolygonTest (Centroid inside Polygon?)
                    └── Space Status: OCCUPIED (🔴) / EMPTY (🟢)
                    └── Occupancy Rate (%) & Statistics
           │
           ▼
    Structured JSON Response
           │
           ▼
   Streamlit Metrics & Visual Table
```

### Component Responsibilities

- **Streamlit Frontend (`streamlit_app/app.py`)**: Interacts with the user, collects uploaded images, provides the click-to-annotate canvas, and renders metrics cards and detailed status tables.
- **FastAPI Backend (`app/main.py`)**: Serves as the central API gateway. Handles image decoding, orchestrates AI inference, validates payloads, and serializes responses.
- **VehicleDetector (`ai/detector.py`)**: Encapsulates Ultralytics YOLO11n, validates confidence thresholds, and filters detections to relevant vehicle categories.
- **ParkingAnalyzer (`ai/parking_analyzer.py`)**: Executes geometric center-point calculations and OpenCV polygon containment tests to establish slot occupancy.
- **ParkingConfig & ParkingAreaManager (`ai/parking_config.py`)**: Validates JSON polygon schemas and persists parking layout configurations in `data/parking_configs/`.
- **Visualizer (`ai/visualizer.py`)**: OpenCV utilities for drawing bounding boxes, class labels, and semi-transparent parking spot status overlays.

---

## 4. How It Works

### YOLO Vehicle Detection
- The system uses **YOLO11n** as a pretrained object detection model (trained on the COCO dataset).
- *Note:* The model is **not** trained from scratch, fine-tuned, or trained to recognize parking spaces directly.
- Instead, YOLO detects general vehicle classes:
  - `car`
  - `motorcycle`
  - `bus`
  - `truck`
- Each detection produces a bounding box `[x1, y1, x2, y2]` along with a confidence score.

### Parking Occupancy Logic
1. **Vehicle Centroid Calculation**: The center point $(C_x, C_y)$ of each detected vehicle bounding box is computed:
   $$C_x = \frac{x_1 + x_2}{2}, \quad C_y = \frac{y_1 + y_2}{2}$$
2. **Polygon Definition**: Each parking space is represented as a 4-point quadrilateral polygon:
   $$[[x_1, y_1], [x_2, y_2], [x_3, y_3], [x_4, y_4]]$$
3. **Point-in-Polygon Test**: The system evaluates OpenCV's `cv2.pointPolygonTest`:
   ```python
   dist = cv2.pointPolygonTest(polygon_pts, (center_x, center_y), measureDist=False)
   ```
4. **Occupancy Decision**: If `dist >= 0`, the vehicle center lies inside or on the contour of the parking space, marking it as **`OCCUPIED`**. If no vehicle center resides within the polygon, the spot is marked **`EMPTY`**.
5. **Statistical Aggregation**: The system tallies total, occupied, and empty spaces, and calculates the lot occupancy rate:
   $$\text{Occupancy Rate (\%)} = \left(\frac{\text{Occupied Spaces}}{\text{Total Spaces}}\right) \times 100$$

---

## 5. Technology Stack

- **Core**: Python 3.10+
- **Deep Learning**: Ultralytics YOLO11n (PyTorch)
- **Computer Vision**: OpenCV (`opencv-python`)
- **Web API**: FastAPI, Uvicorn, Pydantic v2, Python-Multipart
- **Frontend**: Streamlit, HTML5 Canvas, Pandas
- **Testing**: Pytest, HTTPX

---

## 6. Project Structure

```text
smart-parking-ai/
├── ai/                              # Core AI and computer vision modules
│   ├── detector.py                  # YOLO vehicle detection engine
│   ├── parking_analyzer.py          # Point-in-polygon occupancy logic
│   ├── parking_config.py            # Parking area config schema & manager
│   ├── visualizer.py                # OpenCV polygon & bbox drawing utils
│   ├── tracker.py                   # Multi-frame tracking interface (placeholder)
│   └── __init__.py
│
├── app/                             # FastAPI REST API backend
│   ├── main.py                      # FastAPI application entrypoint & routing
│   ├── api/                         # API route endpoints
│   │   ├── parking.py               # Occupancy analysis & parking area endpoints
│   │   ├── health.py                # Service health check endpoint
│   │   └── detection.py             # Detection router (placeholder)
│   ├── schemas/                     # Pydantic data validation models
│   │   └── parking.py               # Request and response schemas
│   └── services/                    # Service layer placeholders
│       ├── detection_service.py
│       ├── parking_service.py
│       └── video_service.py
│
├── streamlit_app/                   # Streamlit web dashboard
│   ├── app.py                       # Main Streamlit dashboard application
│   └── components/                  # Custom Streamlit components
│       └── image_annotator/         # Interactive HTML5 Canvas click annotator
│           ├── __init__.py
│           └── index.html
│
├── data/                            # Project assets & configurations
│   ├── images/                      # Reference and test images (e.g. 1.jpg)
│   ├── parking_configs/             # Saved parking area JSON configurations
│   └── videos/                      # Video sample storage (.gitkeep)
│
├── scripts/                         # Standalone utilities & verification pipelines
│   ├── analyze_parking.py           # Offline image occupancy analysis CLI
│   ├── annotate_parking.py          # OpenCV desktop click annotation tool
│   ├── process_image.py             # Image detection and annotation script
│   ├── test_detection.py            # Standalone YOLO detection tester
│   └── verify_click_workflow.py     # End-to-end verification script
│
├── tests/                           # Automated test suite (54 tests)
│   ├── test_api.py                  # FastAPI route tests
│   ├── test_detector.py             # YOLO detector unit tests
│   ├── test_parking_analyzer.py     # Point-in-polygon analyzer tests
│   ├── test_parking_areas.py        # Parking area manager & persistence tests
│   ├── test_parking_config.py       # Configuration schema & loading tests
│   ├── test_streamlit_app.py        # Streamlit helper & client tests
│   ├── test_visualizer.py           # Drawing utility tests
│   └── test_parking.py              # General placeholder test
│
├── outputs/                         # Output artifacts (.gitkeep)
├── requirements.txt                 # Project dependencies
├── yolo11n.pt                       # Lightweight pretrained YOLO11n model weights (~5.35 MB)
├── .env.example                     # Environment variable template
├── .gitignore                       # Git ignore specifications
└── README.md                        # Project documentation
```

---

## 7. Model File (`yolo11n.pt`)

The project uses the lightweight **YOLO11n** model weights (`yolo11n.pt`, approximately **5.35 MB**).

- **Location**: Placed directly in the project root (`yolo11n.pt`).
- **Storage**: At ~5.35 MB, the weights file is well within GitHub's standard 100 MB file limit and is tracked directly in the repository without requiring Git LFS.
- **Automatic Fallback**: If the file is ever removed or absent, Ultralytics will automatically download the official `yolo11n.pt` asset upon first execution.

---

## 8. Installation

### 1. Clone the Repository
```bash
git clone https://github.com/dilumgit/smart-parking-detection.git
cd smart-parking-detection
```

### 2. Create and Activate a Virtual Environment

**Windows (PowerShell):**
```powershell
python -m venv .venv
.venv\Scripts\activate
```

**Linux / macOS:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 9. Running the Application

To run the complete system, start the FastAPI backend server first, then launch the Streamlit frontend.

### Step 1: Start the FastAPI Backend
```bash
python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```
- **Root URL**: [http://127.0.0.1:8000](http://127.0.0.1:8000)
- **Health Check**: [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health)
- **Interactive Swagger Docs**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **ReDoc Documentation**: [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)

### Step 2: Start the Streamlit Frontend
In a new terminal window (with `.venv` activated):
```bash
python -m streamlit run streamlit_app/app.py
```
- Open your browser at: [http://localhost:8501](http://localhost:8501)
- The Streamlit dashboard automatically connects to the FastAPI backend at `http://127.0.0.1:8000`.

---

## 10. Using the Application

### Mode 1: Check Parking Availability
1. Ensure the FastAPI backend is running (sidebar indicator will show `🟢 API Connected`).
2. Select an active parking area layout from the dropdown (e.g., `default` or any previously saved parking area).
3. Upload an image of the parking lot (JPEG or PNG).
4. Click **🚀 Analyze Parking Lot**.
5. View the occupancy metrics:
   - **Total Spaces**
   - **Occupied Spaces**
   - **Empty Spaces**
   - **Occupancy Rate (%)**
6. Review the itemized slot table showing status (`OCCUPIED` vs `EMPTY`), vehicle counts, detected vehicle classes, and confidence scores.

### Mode 2: Parking Area Setup (Interactive Click-to-Mark)
1. In the sidebar navigation, select **📐 Parking Area Setup**.
2. Enter a **Parking Area Name** (e.g., `West Campus Lot`) and a unique **Parking Area ID** (e.g., `west_campus_lot`).
3. Upload a clear reference photograph of the parking area from the fixed camera angle.
4. Click 4 corners directly on the image to mark each parking space. The slot polygon will automatically close on the 4th click and advance to the next spot (`P1`, `P2`, etc.).
5. Use the control buttons if needed:
   - **Undo Last Point**: Removes the last clicked corner.
   - **Delete Last Space**: Removes the most recently completed parking space.
   - **Reset All**: Clears all current points and spaces.
6. Click **💾 Save Parking Area Configuration** to persist the layout to `data/parking_configs/<parking_area_id>.json` via the FastAPI backend.

---

## 11. REST API Documentation

The FastAPI service exposes the following REST endpoints:

### Core Endpoints

#### 1. Service Root
- **`GET /`**
- Returns service status and links to documentation and endpoints.

#### 2. Service Health
- **`GET /health`**
- Returns `{ "status": "healthy", "service": "smart-parking-ai" }`.

#### 3. List Saved Parking Areas
- **`GET /api/v1/parking/areas`**
- Returns a list of all configured parking area summaries, including area IDs, display names, reference images, and total space counts.

#### 4. Get Parking Area by ID
- **`GET /api/v1/parking/areas/{parking_area_id}`**
- Returns the complete configuration for a specific parking area, including all defined 4-point polygon coordinates.

#### 5. Create Saved Parking Area
- **`POST /api/v1/parking/areas`**
- Validates and saves a new named parking area configuration with its defined space polygons to disk.

#### 6. Analyze Parking Occupancy
- **`POST /api/v1/parking/analyze`**
- Accepts `multipart/form-data`:
  - `file`: Parking lot image file (JPEG, PNG).
  - `parking_area_id` *(Optional query parameter)*: ID of the saved layout to evaluate against (defaults to `parking.json` / `parking_area_001`).
- Runs YOLO11n vehicle detection and OpenCV point-in-polygon analysis.
- Returns `ParkingAnalysisResponse` JSON containing total spaces, occupied count, empty count, occupancy rate, and slot-by-slot results.

---

## 12. Parking Area Configuration Schema

Parking area polygon layouts are saved as JSON files in `data/parking_configs/`. Example schema:

```json
{
  "parking_area_id": "parking_area_001",
  "name": "Main Parking Lot",
  "reference_image": "1.jpg",
  "spaces": [
    {
      "space_id": "P1",
      "polygon": [
        [45, 123],
        [110, 237],
        [213, 212],
        [138, 105]
      ]
    },
    {
      "space_id": "P2",
      "polygon": [
        [140, 104],
        [212, 214],
        [312, 188],
        [230, 87]
      ]
    }
  ]
}
```

---

## 13. Testing

The repository includes a comprehensive automated test suite built with `pytest`:

```bash
python -m pytest -q
```

### Verified Test Suite Result
```text
54 passed, 1 warning in 10.44s
```

The test suite validates:
- Vehicle detector loading and target vehicle class filtering.
- Point-in-polygon calculations and centroid extraction.
- Edge cases in polygon occupancy mapping (boundary, empty, multi-vehicle).
- Configuration loading, schema validation, and persistence.
- FastAPI endpoints (`/health`, `/areas`, `/analyze`).
- Streamlit client utilities and error handling.
- OpenCV visualizer functions.

---

## 14. Important Limitation: Fixed Camera View

> **Crucial Note:** Saved parking-space configurations are strictly tied to the camera angle and viewpoint from which they were marked. A different camera angle, zoom level, or significant perspective shift requires creating a new parking area configuration.

---

## 15. Current Limitations

- **Image-Based Analysis**: The current implementation analyzes static images rather than streaming real-time RTSP/CCTV video feeds.
- **Manual Slot Annotation**: Parking space polygons must be manually calibrated once per camera view rather than automatically segmented by an AI model.
- **Perspective & Angles**: Pretrained COCO weights perform best on standard perspective angles; extreme top-down or heavily occluded views may experience reduced vehicle recall.
- **No Historical Analytics**: Occupancy results reflect the uploaded snapshot; no persistent database is included for historical trend analytics.

---

## 16. Future Improvements

The following features represent potential architectural extensions:
- **Fine-Tuned YOLO**: Fine-tune YOLO on specialized aerial and parking lot vehicle datasets for superior detection under extreme angles.
- **Automated Space Detection**: Train a segmentation model to automatically detect parking line demarcations.
- **Live Stream Integration**: Implement an RTSP/video ingestion worker to process live camera feeds continuously.
- **Vehicle Tracking**: Track vehicles across video frames using DeepSORT or ByteTrack to monitor parking duration and flow.
- **Analytics Database**: Integrate PostgreSQL/TimescaleDB to log occupancy events and visualize peak usage trends.

---

## 17. License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
