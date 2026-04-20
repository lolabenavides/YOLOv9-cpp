#include "YOLOv9.h"
#include <opencv2/dnn.hpp> 
#include <chrono>

YOLOv9::YOLOv9(const std::string& model_path) //Constructor
 : env(ORT_LOGGING_LEVEL_WARNING, "YOLOv9"),
    session_options(),
    session(env, model_path.c_str(), session_options)
{
    session_options.SetIntraOpNumThreads(1);
}

cv::Mat letterbox(const cv::Mat& img, int new_w, int new_h, LetterboxInfo& info)
{
    int w = img.cols;
    int h = img.rows;

    float scale = std::min((float)new_w / w, (float)new_h / h);

    int resized_w = int(w * scale);
    int resized_h = int(h * scale);

    cv::Mat resized;
    cv::resize(img, resized, cv::Size(resized_w, resized_h));

    int pad_x = (new_w - resized_w) / 2;
    int pad_y = (new_h - resized_h) / 2;

    cv::Mat output;
    cv::copyMakeBorder(
        resized,
        output,
        pad_y,
        new_h - resized_h - pad_y,
        pad_x,
        new_w - resized_w - pad_x,
        cv::BORDER_CONSTANT,
        cv::Scalar(114, 114, 114) // gris típico YOLO
    );

    info.scale = scale;
    info.pad_x = pad_x;
    info.pad_y = pad_y;

    return output;
}


std:: vector<float> YOLOv9::preprocess(const cv::Mat& img, LetterboxInfo& info)
{
    //Resize image: 640x640
    cv::Mat boxed = letterbox(img, 640, 640, info);

    //BGR to RGB
    cv::Mat rgb;
    cv::cvtColor(boxed, rgb, cv::COLOR_BGR2RGB);

    //Float + norm
    rgb.convertTo(rgb, CV_32F, 1.0 / 255.0);

    //HWC to CHW
    std::vector<cv::Mat> channels(3);
    cv::split(rgb, channels);

    std::vector<float> processed_data;
    processed_data.reserve(input_width * input_height * 3);

    for (int c = 0; c < 3; c++)
    {
        processed_data.insert(processed_data.end(),(float*)channels[c].datastart, (float*)channels[c].dataend);
    }

    return processed_data;
}


std:: vector<Detection> YOLOv9::inference(std::vector<float>& processed_data,const cv::Mat& img, LetterboxInfo& info, double& ms) 
{
    //Define input tensor [1, 3, 640, 640]
    std::array<int64_t, 4> input_shape = {1, 3, 640, 640};

    //Create CPU memory info and input tensor from processed data
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(
        OrtArenaAllocator, OrtMemTypeDefault);

    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info,
        processed_data.data(),
        processed_data.size(),
        input_shape.data(),
        input_shape.size()
    );

    //Model input/output names (.onnx)
    std::vector<const char*> input_names  = {"images"};
    std::vector<const char*> output_names = {"output0"};

    //Start time counter
    auto start = std::chrono::high_resolution_clock::now();
    
    //Run inference
    auto output_tensors = session.Run(
        Ort::RunOptions{nullptr},
        input_names.data(),
        &input_tensor, 1,
        output_names.data(), 1
    );

    //End time counter
    auto end = std::chrono::high_resolution_clock::now();
    ms = std::chrono::duration<double, std::milli>(end - start).count();

    //Access output data and extract shape info (classes and boxes)
    float* output_data = (float*)output_tensors[0].GetTensorMutableData<void>();
    auto shape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();

    int num_classes = shape[1] - 4;
    int num_boxes = shape[2];   

    //Convert raw data into detections
    std::vector<Detection> detections;
    float conf_threshold = 0.25f;

    for (int i = 0; i < num_boxes; i++)
    {
        //Box coordinates (cx, cy, w, h)
        float cx = output_data[0 * num_boxes + i];
        float cy = output_data[1 * num_boxes + i];
        float w = output_data[2 * num_boxes + i];
        float h = output_data[3 * num_boxes + i];

        //Search for the class with the highest confidence
        float max_conf = 0.0f;
        int   max_cls  = 0;
        for (int c = 0; c < num_classes; c++)
        {
            float conf = output_data[(4 + c) * num_boxes + i];
            if (conf > max_conf) {
                max_conf = conf;
                max_cls  = c;
            }
        }

        if (max_conf < conf_threshold) {
            continue;
        }

        //Postprocess: Undo letterbox
        float x1 = (cx - w / 2 - info.pad_x) / info.scale;
        float y1 = (cy - h / 2 - info.pad_y)  / info.scale;
        float x2 = (cx + w / 2 - info.pad_x) / info.scale;
        float y2 = (cy + h / 2 - info.pad_y)  / info.scale;

        //Adjust the bounding box coordinates to the image boundaries
        x1 = std::max(0.0f, x1);
        y1 = std::max(0.0f, y1);
        x2 = std::min((float)img.cols, x2);
        y2 = std::min((float)img.rows, y2);

        //Convert cx,cy,w,h to x,y,w,h (top-left corner format)
        Detection det;
        det.class_id   = max_cls;
        det.confidence = max_conf;
        det.box = cv::Rect(
            cv::Point(
            (int)x1, 
            (int)y1), 
            cv::Point(
            (int)x2, 
            (int)y2));
        detections.push_back(det);
    }

    //Detect and delete duplicates (NMS)
    //Variables for NMS 
    std::vector<cv::Rect> boxes; 
    std::vector<float> confidences; 
    std::vector<int> class_ids; 
    
    for (const Detection& d : detections) { 
    boxes.push_back(d.box); 
    confidences.push_back(d.confidence); 
    class_ids.push_back(d.class_id); 
    } 
    
    //Apply NMS 
    std::vector<int> indices; 
    float nms_threshold = 0.45f;  // ajustable, típico en YOLO 
    cv::dnn::NMSBoxes(boxes, confidences, conf_threshold, nms_threshold, indices); 
    
    //Final vector using only the remaining detections 
    std::vector<Detection> final_detections; 
    for (int idx : indices) { 
    final_detections.push_back(detections[idx]); 
    } 
    
    return final_detections;
}
