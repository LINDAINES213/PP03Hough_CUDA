#ifndef PGMIMAGE_H
#define PGMIMAGE_H

#include <string>
#include <fstream>
#include <iostream>
#include <sstream>

class PGMImage {
public:
    int x_dim;        // width of the image
    int y_dim;        // height of the image
    int maxVal;       // maximum value of a pixel
    unsigned char* pixels;  // pixel data

    PGMImage(const char* filename) {
        std::ifstream file(filename, std::ios::binary);
        if (!file) {
            std::cerr << "Cannot open file: " << filename << std::endl;
            exit(1);
        }

        std::string magic;
        file >> magic;
        
        if (magic != "P5") {
            std::cerr << "Invalid magic number, must be P5" << std::endl;
            exit(1);
        }

        // Skip comments
        char line[1024];
        while (file.peek() == '#') {
            file.getline(line, 1024);
        }

        file >> x_dim >> y_dim >> maxVal;
        file.get(); // Skip whitespace

        pixels = new unsigned char[x_dim * y_dim];
        file.read(reinterpret_cast<char*>(pixels), x_dim * y_dim);
        file.close();
    }

    ~PGMImage() {
        delete[] pixels;
    }

    // Prevent copying to avoid double deletion of pixels
    PGMImage(const PGMImage&) = delete;
    PGMImage& operator=(const PGMImage&) = delete;
};

#endif // PGMIMAGE_H