#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <cstring>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

using namespace std;
using namespace cv;

class OccupancyMap {
    public:
        OccupancyMap(const string& image_path, int num_mip=3);
        void turn_img_to_bit_array(const string& image_path);
        void pack_bit_array_to_om();
        vector<int> bitfield(int n);
        Mat unpack_om_to_bit_array(Mat om);
        Mat create_next_level_mipmap(Mat prev_mip, int mip_size);
        void generate_mipmaps(int num_mip);
        int initialize_step(int dir);
        int get_query_bits(int a_min, int a_max, int b_start, int b_end);
        bool check_visibility(Point2f query_start_uv, Point2f query_end_uv);
    private:
        vector<Mat> bit_imgs;
        Mat sequence_base_img;
        Mat om;
        vector<Mat> mipmaps;
        int res;
        Point2f query_start_pos, query_end_pos;
};

OccupancyMap::OccupancyMap(const string& image_path, int num_mip) {
    turn_img_to_bit_array(image_path);
    pack_bit_array_to_om();
    generate_mipmaps(num_mip);
}

void OccupancyMap::turn_img_to_bit_array(const string& image_path) {
    Mat img = imread(image_path, IMREAD_COLOR);
    cvtColor(img, img, COLOR_BGR2GRAY);
    img.convertTo(img, CV_32FC1, 1/255.0);

    img = 1 - (img > 0.01);

    bit_imgs.push_back(img.clone());
    sequence_base_img = img.clone();

    for (int i = 0; i < sequence_base_img.rows; i++) {
        for (int j = 0; j < sequence_base_img.cols; j++) {
            if (sequence_base_img.at<float>(i, j) >= 1) {
                sequence_base_img.at<float>(i, j) = 10;
            }
        }
    }

    res = img.rows;
}

void OccupancyMap::pack_bit_array_to_om() {
    vector<uint32_t> om_x(4);
    vector<vector<uint32_t>> om(res, om_x);

    for (int i = 0; i < res; i++) {
        for (int j = 0; j < 4; j++) {
            uint32_t bits = 0;
            for (int k = 0; k < 32; k++) {
                bits += (bit_imgs[0].at<float>(i, j * 32 + k) != 0) << k;
            }
            om[i][j] = bits;
        }
    }

    om.convertTo(om, CV_32SC4);
    this->om = om;
}

vector<int> OccupancyMap::bitfield(int n) {
    vector<int> bits(32);

    for (int i = 0; i < 32; i++) {
        bits[i] = (n >> i) & 1;
    }

    return bits;
}

Mat OccupancyMap::unpack_om_to_bit_array(Mat om) {
    vector<float> bit_img_x(128);
    vector<vector<float>> bit_img(om.rows, bit_img_x);

    for (int i = 0; i < om.rows; i++) {
        for (int j = 0; j < 4; j++) {
           
