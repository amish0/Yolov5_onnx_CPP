#ifndef CONFIG_READER_H
#define CONFIG_READER_H

#include <iostream>
#include <stdlib.h>
#include <string.h>
#include "ini_reader.h"

struct cam_info_t
{
    char name[20];
    char type[20];
    char url[100];
    int id;
    int width;
    int height;
    int fps = -1;
};

class ConfigReader
{
private:
    std::string file_path_;
    IniReaderNested ini_reader_;
public:
    ConfigReader(const std::string &file_path) : file_path_(file_path), ini_reader_(file_path_)
    {
        get_config_info();
        get_cam_info();
    }
    ~ConfigReader();
    void get_config_info();
    void get_cam_info();
    void print_cam_info();
    
    int cam_num_;
    int width_;
    int height_;
    cam_info_t *cam_list_;
};

extern ConfigReader configparameter;
#endif // CAM_META_INFO_H
