#include "YOLOv9.h"
#include <filesystem>

int main() {

    //Init variables
    std::string onnx_path = "../yolov9-m-converted.onnx";
    std::string input_folder  = "../images/inputs/assets/";
    std::string output_folder = "../images/outputs/";
    double total_time = 0.0;
    

    //Create output folder if it does not exist
    std::filesystem::create_directories(output_folder);

    //COCO dataset classes
    std::vector<std::string> class_names =  {
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
    "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv",
    "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush"
    };

    //Initialise the model and load weights (.onnx)
    YOLOv9 model (onnx_path);

    //Valid image extensions
    std::vector<std::string> valid_ext = {".jpg", ".jpeg", ".png", ".bmp"};

    int processed_images = 0; 

    //Loop through all the files in the folder
    for (const auto& file : std::filesystem::directory_iterator(input_folder))
    {
        //Check if file extension is valid
        std::string ext = file.path().extension().string();
        bool is_image = false;
        for (const auto& e : valid_ext)
        {
            if (ext == e) { 
                is_image = true; 
                processed_images++; 
                break; 
            }
        }
        //If file is not an image, skip the rest of the code
        if (!is_image) 
        {
            continue;
        }

        //Set paths
        std::string file_name = file.path().filename().string();
        std::string input_path  = file.path().string();
        std::string output_path = output_folder + file_name;

        std::cout << "Processing: " << file_name << std::endl;
        
        //Save image in "img"
        cv::Mat img = cv::imread(input_path); 
        //Check if image is empty
        if (img.empty()) {
            std::cout << "Error: empty image. Skipping to the next image." << std::endl;
            continue;
        }

        //Letterbox info for image preprocessing and postprocessing
        LetterboxInfo info;
        //Preprocess
        std::vector<float> processed_data = model.preprocess(img, info);
        //Inference + postprocess
        double ms = 0.0; //Time in ms for each image
        std::vector<Detection> detections = model.inference(processed_data, img, info, ms);
        
        total_time += ms;

        //Calculate the scale factor based on the image
        float scale_factor = std::min(img.cols, img.rows) / 640.0f;
        float font_scale  = 0.6f * scale_factor;
        int   thickness   = std::max(1, (int)(2 * scale_factor));
        int   rect_thickness = std::max(1, (int)(2 * scale_factor));

        //Draw each detection
        for (const Detection& det : detections)
        {
            //Rectangle
            cv::rectangle(img, det.box, cv::Scalar(0, 255, 0), rect_thickness);

            //Label
            std::string label = class_names[det.class_id] + " " + std::to_string((int)(det.confidence * 100)) + "%";

            int baseline = 0;
            cv::Size text_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 
            font_scale, thickness, &baseline);

            cv::rectangle(img,
            cv::Point(det.box.x, det.box.y - text_size.height - 5),
            cv::Point(det.box.x + text_size.width, det.box.y),
            cv::Scalar(0, 255, 0), cv::FILLED);

            cv::putText(img, label,
            cv::Point(det.box.x, det.box.y - 3),
            cv::FONT_HERSHEY_SIMPLEX, font_scale,
            cv::Scalar(0, 0, 0), thickness);
        }
        cv::imwrite(output_path, img);
        std::cout << "Saved in: " << output_path << " (" << detections.size() << " detections). Latency: " << ms << " ms | FPS: "<< 1000.0 / ms <<std::endl;
    }
    if (processed_images == 0)
    std::cout << "No images were found in: " << input_folder << std::endl;
    else
    std::cout << "Finished." << processed_images << " processed images." << std::endl;
    std::cout << "Total time: " << total_time/1000 << " s." << "\n";
    return 0;

}