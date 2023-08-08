#include "ini_reader.h"

bool IniReader::readIniFile()
{
    std::ifstream file(file_path_);
    if (!file.is_open())
    {
        std::cout << "Error: File not found: " << file_path_ << std::endl;
        return false;
    }

    std::string line;
    while (std::getline(file, line))
    {
        if (!line.empty() && line[0] != '#')
        {
            std::istringstream iss(line);
            std::string key, value;
            if (std::getline(iss, key, ':') && std::getline(iss, value))
            {
                config_[trim(key)] = trim(value);
            }
        }
    }

    file.close();
    return true;
}

std::string IniReader::getValue(const std::string &key) const
{
    auto it = config_.find(key);
    return (it != config_.end()) ? it->second : "";
}

std::string IniReader::trim(const std::string &str)
{
    size_t first = str.find_first_not_of(" \t\n\r\f\v");
    size_t last = str.find_last_not_of(" \t\n\r\f\v");
    return (first != std::string::npos && last != std::string::npos) ? str.substr(first, last - first + 1) : "";
}

bool IniReaderNested::readIniFile()
{
    std::ifstream file(file_path_);
    if (!file.is_open())
    {
        std::cout << "Error: File not found: " << file_path_ << std::endl;
        return false;
    }

    std::string current_section;
    std::string line;
    while (std::getline(file, line))
    {
        if (!line.empty() && line[0] != '#')
        {
            if (line[0] == '[' && line.back() == ']')
            {
                current_section = line.substr(1, line.size() - 2);
            }
            else
            {
                std::istringstream iss(line);
                std::string key, value;
                if (std::getline(iss, key, ':') && std::getline(iss, value))
                {
                    key = trim(key);
                    value = trim(value);
                    config_[current_section][key] = value;
                }
            }
        }
    }
    file.close();
    return true;
}

std::string IniReaderNested::getValue(const std::string &section, const std::string &key) const
{
    auto it = config_.find(section);
    if (it != config_.end())
    {
        auto it2 = it->second.find(key);
        if (it2 != it->second.end())
        {
            return it2->second;
        }
    }
    return "";
}

std::string IniReaderNested::trim(const std::string &str)
{
    size_t first = str.find_first_not_of(" \t\n\r\f\v");
    size_t last = str.find_last_not_of(" \t\n\r\f\v");
    return (first != std::string::npos && last != std::string::npos) ? str.substr(first, last - first + 1) : "";
}
