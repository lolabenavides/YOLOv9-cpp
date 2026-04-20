#pragma once 
#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
#include <iostream> 

struct Detection {
    int class_id;
    float confidence;
    cv::Rect box;
};

struct LetterboxInfo {
    float scale;
    int pad_x;
    int pad_y;
};

class YOLOv9 {
public:
    YOLOv9(const std::string& model_path); //Constructor de clase

    std::vector<float> preprocess(const cv::Mat& img, LetterboxInfo& info);
    std::vector<Detection> inference(std::vector<float>& processed_data, const cv::Mat& img, LetterboxInfo& info, double& ms);
    
private:
    Ort::Env env;
    Ort::SessionOptions session_options;
    Ort::Session session;
    int input_width = 640;
    int input_height = 640;
};