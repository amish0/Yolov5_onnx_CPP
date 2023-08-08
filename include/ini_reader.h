#ifndef INI_READER_H
#define INI_READER_H
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <unordered_map>

class IniReader
{
public:
    IniReader(const std::string &file_path) : file_path_(file_path) {}

    bool readIniFile();

    std::string getValue(const std::string &key) const;

private:
    std::string file_path_;
    std::unordered_map<std::string, std::string> config_;

    std::string trim(const std::string &str);
};

class IniReaderNested
{
public:
    IniReaderNested(const std::string &file_path) : file_path_(file_path) {}

    bool readIniFile();

    std::string getValue(const std::string &section, const std::string &key) const;
private:
    std::string file_path_;
    std::unordered_map<std::string, std::unordered_map<std::string, std::string>> config_;

    std::string trim(const std::string &str);
};

#endif // !READINI_H