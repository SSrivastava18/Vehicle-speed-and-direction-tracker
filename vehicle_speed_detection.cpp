#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>

using namespace cv;
using namespace dnn;
using namespace std;

// Function to get output layer names
vector<String> getOutputLayersNames(const Net& net) {
    vector<String> layerNames = net.getLayerNames();
    vector<String> outputLayers;
    for (auto i : net.getUnconnectedOutLayers())
        outputLayers.push_back(layerNames[i - 1]);
    return outputLayers;
}

int main() {
    // Paths to YOLO files
    string weightsPath = "./yolo/yolov4.weights";   // Update with your YOLO weights path
    string configPath = "./yolo/yolov4.cfg";        // Update with your YOLO config path
    string namesPath = "./yolo/coco.names";         // Update with your COCO class labels file path
    string videoPath = "D:/Vehicle speed detection/12691893_3840_2160_30fps.mp4"; // Update with your video path

    // Load YOLO model
    Net net = readNet(weightsPath, configPath);
    net.setPreferableBackend(DNN_BACKEND_OPENCV);
    net.setPreferableTarget(DNN_TARGET_CPU);

    // Load COCO class labels
    vector<string> classes;
    ifstream ifs(namesPath.c_str());
    string line;
    while (getline(ifs, line)) {
        classes.push_back(line);
    }

    // Load video
    VideoCapture cap(videoPath);
    if (!cap.isOpened()) {
        cerr << "Error: Could not open video." << endl;
        return -1;
    }

    // Get video properties
    double fps = cap.get(CAP_PROP_FPS);
    cout << "Video FPS: " << fps << endl;

    // Define a known distance (in meters) for speed calculation
    const double knownDistance = 150.0; // Adjust this based on your real-world reference

    // Initialize tracking variables
    vector<Rect> previousFrameBoxes;

    Mat frame;
    while (cap.read(frame)) {
        // Resize frame for better visibility
        const int resizeWidth = 1280, resizeHeight = 720;
        resize(frame, frame, Size(resizeWidth, resizeHeight));
        int width = frame.cols, height = frame.rows;

        // Create blob from frame
        Mat blob;
        blobFromImage(frame, blob, 0.00392, Size(416, 416), Scalar(), true, false);
        net.setInput(blob);

        // Forward pass
        vector<Mat> outs;
        net.forward(outs, getOutputLayersNames(net));

        // Process detections
        vector<int> classIds;
        vector<float> confidences;
        vector<Rect> boxes;
        for (size_t i = 0; i < outs.size(); ++i) {
            float* data = (float*)outs[i].data;
            for (int j = 0; j < outs[i].rows; ++j, data += outs[i].cols) {
                Mat scores = outs[i].row(j).colRange(5, outs[i].cols);
                Point classIdPoint;
                double confidence;
                minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);
                int classId = classIdPoint.x;

                if (confidence > 0.5 &&
                    (classes[classId] == "car" || classes[classId] == "truck" ||
                     classes[classId] == "bus" || classes[classId] == "motorbike")) {
                    int centerX = int(data[0] * width);
                    int centerY = int(data[1] * height);
                    int w = int(data[2] * width);
                    int h = int(data[3] * height);
                    int x = centerX - w / 2;
                    int y = centerY - h / 2;

                    boxes.push_back(Rect(x, y, w, h));
                    confidences.push_back((float)confidence);
                    classIds.push_back(classId);
                }
            }
        }

        // Non-Maximum Suppression
        vector<int> indices;
        NMSBoxes(boxes, confidences, 0.5, 0.4, indices);

        // Process the detections
        vector<Rect> currentFrameBoxes;
        for (int idx : indices) {
            Rect box = boxes[idx];
            Point vehicleCenter(box.x + box.width / 2, box.y + box.height / 2);

            // Match with previous frame vehicles
            bool matched = false;
            for (Rect prevBox : previousFrameBoxes) {
                Point prevCenter(prevBox.x + prevBox.width / 2, prevBox.y + prevBox.height / 2);
                double distancePixels = norm(vehicleCenter - prevCenter);

                if (distancePixels < 50) { // Adjust threshold
                    matched = true;
                    double timeSeconds = 1.0 / fps;
                    double speedMps = (distancePixels / width) * knownDistance / timeSeconds;
                    double speedKmph = speedMps * 3.6;

                    putText(frame, format("Speed: %.2f km/h", speedKmph),
                            Point(box.x, box.y - 10), FONT_HERSHEY_SIMPLEX, 0.5,
                            Scalar(0, 255, 255), 2);
                    cout << "Vehicle Speed: " << speedKmph << " km/h" << endl;
                }
            }

            currentFrameBoxes.push_back(box);

            // Draw bounding box
            rectangle(frame, box, Scalar(0, 255, 0), 2);
            putText(frame, format("%s: %.2f", classes[classIds[idx]].c_str(), confidences[idx]),
                    Point(box.x, box.y - 10), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0), 2);
        }

        previousFrameBoxes = currentFrameBoxes;

        // Display the frame
        imshow("Vehicle Speed Detection", frame);

        if (waitKey(1) == 'q') break; // Exit on 'q'
    }

    cap.release();
    destroyAllWindows();

    return 0;
}
