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