#include "config_reader.h"

ConfigReader configparameter("test.ini");

ConfigReader::~ConfigReader()
{
    delete[] cam_list_;
}

void ConfigReader::get_config_info()
{
    if (!ini_reader_.readIniFile())
    {
        std::cout << "Error: Failed to read config file: " << file_path_ << std::endl;
    }
    else
    {
        std::cout << "Successfully read config file: " << file_path_ << std::endl;
        cam_num_ = atoi(ini_reader_.getValue("config", "Number_Of_Camera").c_str());
        width_ = atoi(ini_reader_.getValue("config", "Width").c_str());
        height_ = atoi(ini_reader_.getValue("config", "Height").c_str());
        cam_list_ = new cam_info_t[cam_num_];
    }
}

void ConfigReader::get_cam_info()
{
    for (int i = 0; i < cam_num_; i++)
    {
        std::string section = "cam" + std::to_string(i);
        strcpy_s(cam_list_[i].name, ini_reader_.getValue(section, "Name").c_str());
        strcpy_s(cam_list_[i].type, ini_reader_.getValue(section, "Type").c_str());
        strcpy_s(cam_list_[i].url, ini_reader_.getValue(section, "Url").c_str());
        cam_list_[i].id = atoi(ini_reader_.getValue(section, "Id").c_str());
        cam_list_[i].width = width_;
        cam_list_[i].height = height_;
    }
}

void ConfigReader::get_yolov5_info()
{
    yolov5_config_.input_width = atoi(ini_reader_.getValue("yolov5", "Input_Width").c_str());
    yolov5_config_.input_height = atoi(ini_reader_.getValue("yolov5", "Input_Height").c_str());
    yolov5_config_.score_threshold = atof(ini_reader_.getValue("yolov5", "Score_Threshold").c_str());
    yolov5_config_.nms_threshold = atof(ini_reader_.getValue("yolov5", "Nms_Threshold").c_str());
    yolov5_config_.confidence_threshold = atof(ini_reader_.getValue("yolov5", "Confidence_Threshold").c_str());
    strcpy_s(yolov5_config_.class_list, ini_reader_.getValue("yolov5", "Class_List").c_str());
    yolov5_config_.batch_size = atoi(ini_reader_.getValue("yolov5", "Batch_Size").c_str());
    if (yolov5_config_.batch_size != cam_num_)
    {
        std::cout << "Error: Batch size is not equal camera number!" << std::endl;
        yolov5_config_.batch_size = cam_num_;
    }
    else
    {
        std::cout << "Batch size is equal to camera number!" << std::endl;
    }
    strcpy_s(yolov5_config_.model_path, ini_reader_.getValue("yolov5", "Model_Path").c_str());

}

void ConfigReader::print_cam_info()
{
    for (int i = 0; i < cam_num_; i++)
    {
        std::cout << "\n\nCam " << i << " info:" << std::endl;
        std::cout << "Name: " << cam_list_[i].name << std::endl;
        std::cout << "Type: " << cam_list_[i].type << std::endl;
        std::cout << "Url: " << cam_list_[i].url << std::endl;
        std::cout << "Id: " << cam_list_[i].id << std::endl;
        std::cout << "Width: " << cam_list_[i].width << std::endl;
        std::cout << "Height: " << cam_list_[i].height << std::endl;
        std::cout << "Fps: " << cam_list_[i].fps << std::endl;
    }
}

void ConfigReader::print_yolov5_info()
{
    std::cout << "\n\nYolov5 info:" << std::endl;
    std::cout << "Input_Width: " << yolov5_config_.input_width << std::endl;
    std::cout << "Input_Height: " << yolov5_config_.input_height << std::endl;
    std::cout << "Score_Threshold: " << yolov5_config_.score_threshold << std::endl;
    std::cout << "Nms_Threshold: " << yolov5_config_.nms_threshold << std::endl;
    std::cout << "Confidence_Threshold: " << yolov5_config_.confidence_threshold << std::endl;
    std::cout << "Class_List: " << yolov5_config_.class_list << std::endl;
    std::cout << "Batch_Size: " << yolov5_config_.batch_size << std::endl;
    std::cout << "Model_Path: " << yolov5_config_.model_path << std::endl;
}