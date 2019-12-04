#include <sstream>
#include <iostream>
#define _USE_MATH_DEFINES
#include <cmath>
#include <algorithm>
#include <string>
#include <vector>
#include "Image.h"
#include "opencv2/opencv.hpp"

using namespace cv;

namespace imgrecog {
	Image::Image(const std::string& path)
		: _path_templates({
					"../templates/0.png",
					"../templates/1.png",
					"../templates/2.png",
					"../templates/3.png",
					"../templates/4.png",
					"../templates/5.png",
					"../templates/6.png",
					"../templates/7.png",
					"../templates/8.png",
					"../templates/9.png"
		}),
		_template_image({
			cv::imread(_path_templates[0], cv::IMREAD_GRAYSCALE),
			cv::imread(_path_templates[1], cv::IMREAD_GRAYSCALE),
			cv::imread(_path_templates[2], cv::IMREAD_GRAYSCALE),
			cv::imread(_path_templates[3], cv::IMREAD_GRAYSCALE),
			cv::imread(_path_templates[4], cv::IMREAD_GRAYSCALE),
			cv::imread(_path_templates[5], cv::IMREAD_GRAYSCALE),
			cv::imread(_path_templates[6], cv::IMREAD_GRAYSCALE),
			cv::imread(_path_templates[7], cv::IMREAD_GRAYSCALE),
			cv::imread(_path_templates[8], cv::IMREAD_GRAYSCALE),
			cv::imread(_path_templates[9], cv::IMREAD_GRAYSCALE)
		}),
		_path(path),
		_image(cv::imread(path, cv::IMREAD_COLOR)),
		_edge(_image.clone()),
		_edgePoint(),
		_gradient(_edge.rows, std::vector<double>(_edge.cols, 0)),
		_direction(_edge.rows, std::vector<double>(_edge.cols, 0))
	{
		if (!_image.data) {
			std::stringstream ss;
			ss << "Error: Could not open or find the image: " << _path;
			throw std::invalid_argument(ss.str());
		}
		edgeDetection();
	}

	void Image::edgeGrayscale() noexcept
	{
		uint8_t* pixelPtr = (uint8_t*)_edge.data;
		int cn = _edge.channels();
		cv::Scalar_<uint8_t> bgrPixel;

		for (int y = 0; y < _edge.rows; ++y)
			for (int x = 0; x < _edge.cols; ++x) {
				bgrPixel.val[0] = pixelPtr[y * _edge.cols * cn + x * cn + 0]; // B
				bgrPixel.val[1] = pixelPtr[y * _edge.cols * cn + x * cn + 1]; // G
				bgrPixel.val[2] = pixelPtr[y * _edge.cols * cn + x * cn + 2]; // R

				uint8_t grayscale = .2126 * (double)bgrPixel.val[2] + .7152 * (double)bgrPixel.val[1] + .0722 * (double)bgrPixel.val[0];

				pixelPtr[y * _edge.cols * cn + x * cn + 0] = grayscale; // B
				pixelPtr[y * _edge.cols * cn + x * cn + 1] = grayscale; // G
				pixelPtr[y * _edge.cols * cn + x * cn + 2] = grayscale; // R
			}

	}

	void Image::edgeBlur() noexcept
	{
		std::vector<std::vector<double>> kernel = {
			{1. / 256., 4. / 256., 6. / 256., 4. / 256., 1. / 256.},
			{4. / 256., 16. / 256., 24. / 256., 16. / 256., 4. / 256.},
			{6. / 256., 24. / 256., 36. / 256., 24. / 256., 6. / 256.},
			{4. / 256., 16. / 256., 24. / 256., 16. / 256., 4. / 256.},
			{1. / 256., 4. / 256., 6. / 256., 4. / 256., 1. / 256.}
		};

		int offset = kernel.size() / 2;
		auto copy = _edge.clone();
		uint8_t* pixelCopyPtr = (uint8_t*)copy.data;
		uint8_t* pixelPtr = (uint8_t*)_edge.data;
		int cn = _edge.channels();
		cv::Scalar_<uint8_t> bgrPixel;
		for (int y = 0; y < _edge.rows; ++y)
			for (int x = 0; x < _edge.cols; ++x) {
				double acc = 0;
				for (int b = 0; b < kernel.size(); ++b)
					for (int a = 0; a < kernel[b].size(); ++a) {
						int xn = x + a - offset;
						if (xn > _edge.cols - 1)
							xn = _edge.cols - 1;
						if (xn < 0)
							xn = 0;
						int yn = y + b - offset;
						if (yn > _edge.rows - 1)
							yn = _edge.rows - 1;
						if (yn < 0)
							yn = 0;
						acc += static_cast<double>(pixelCopyPtr[yn * _edge.cols * cn + xn * cn])* kernel[a][b];
					}
				pixelPtr[y * _edge.cols * cn + x * cn + 0] = static_cast<int>(acc); // B
				pixelPtr[y * _edge.cols * cn + x * cn + 1] = static_cast<int>(acc); // G
				pixelPtr[y * _edge.cols * cn + x * cn + 2] = static_cast<int>(acc); // R
			}
	}

	void Image::edgeGradient() noexcept
	{
		uint8_t* pixelPtr = (uint8_t*)_edge.data;
		int cn = _edge.channels();
		for (int y = 0; y < _gradient.size(); ++y)
			for (int x = 0; x < _gradient[y].size(); ++x) {
				if (0 < x && x < _gradient[y].size() - 1 && 0 < y && y < _gradient.size() - 1) {
					auto val1 = static_cast<double>(pixelPtr[y * _edge.cols * cn + (x + 1) * cn]);
					auto val2 = static_cast<double>(pixelPtr[y * _edge.cols * cn + (x - 1) * cn]);
					double magx = val1 - val2;
					val1 = static_cast<double>(pixelPtr[(y + 1) * _edge.cols * cn + x * cn]);
					val2 = static_cast<double>(pixelPtr[(y - 1) * _edge.cols * cn + x * cn]);
					double magy = val1 - val2;
					_gradient[y][x] = sqrt(magx * magx + magy * magy);
					_direction[y][x] = atan2(magy, magx);
				}
			}
	}

	void Image::edgeRemoveNonMaxGradient() noexcept
	{
		for (int y = 1; y < _gradient.size() - 1; ++y)
			for (int x = 1; x < _gradient[y].size() - 1; ++x) {
				double angle = _direction[y][x] >= 0 ? _direction[y][x] : _direction[y][x] + M_PI;
				double rangle = round(angle / (M_PI_4));
				double mag = _gradient[y][x];
				if ((rangle == 0 || rangle == 4) && (_gradient[y][x + 1] > mag || _gradient[y][x + 1] > mag)
					|| (rangle == 1 && (_gradient[y - 1][x - 1] > mag || _gradient[y + 1][x + 1] > mag))
					|| (rangle == 2 && (_gradient[y - 1][x] > mag || _gradient[y + 1][x] > mag))
					|| (rangle == 3 && (_gradient[y - 1][x + 1] > mag || _gradient[y + 1][x - 1] > mag)))
					_gradient[y][x] = 0;
			}
	}

	void Image::edgeFilter(int low, int high) noexcept
	{
		int cn = _edge.channels();
		cv::Scalar_<uint8_t> bgrPixel;
		uint8_t* pixelPtr = (uint8_t*)_edge.data;
		std::set<std::pair<int, int>> iter;
		for (int y = 0; y < _edge.rows; ++y)
			for (int x = 0; x < _edge.cols; ++x) {
				if (_gradient[y][x] > high) {
					pixelPtr[y * _edge.cols * cn + x * cn + 0] = 255;
					pixelPtr[y * _edge.cols * cn + x * cn + 1] = 255;
					pixelPtr[y * _edge.cols * cn + x * cn + 2] = 255;
					iter.insert(std::pair<int, int>(x, y));
					_edgePoint.insert(std::pair<int, int>(x, y));
				}
				else {
					pixelPtr[y * _edge.cols * cn + x * cn + 0] = 0;
					pixelPtr[y * _edge.cols * cn + x * cn + 1] = 0;
					pixelPtr[y * _edge.cols * cn + x * cn + 2] = 0;
				}
			}
		std::vector<std::pair<int, int>> neighbors =
		{ {-1, -1}, {-1, 0}, {-1, 1}, {0, -1}, {0, 1}, {1, -1}, {1, 0}, {1, 1} };
		while (iter.size()) {
			std::set<std::pair<int, int>> tmp;
			for (auto& pos : iter) {
				int x = pos.first;
				int y = pos.second;
				for (auto& neighbor : neighbors) {
					int a = neighbor.first;
					int b = neighbor.second;
					if (!(y + b < 0 || y + b > _gradient.size() || x + a < 0 || x + a > _gradient[y + b].size())
						&& _gradient[y + b][x + a] > low
						&& pixelPtr[(y + b) * _edge.cols * cn + (x + a) * cn + 0] == 0)
						tmp.insert(std::make_pair<int, int>(x + a, y + b));
				}
			}

			for (auto& pos : tmp) {
				int x = pos.first;
				int y = pos.second;
				pixelPtr[y * _edge.cols * cn + x * cn + 0] = 255;
				pixelPtr[y * _edge.cols * cn + x * cn + 1] = 255;
				pixelPtr[y * _edge.cols * cn + x * cn + 2] = 255;
				_edgePoint.insert(std::pair<int, int>(x, y));
			}
			iter = tmp;
		}
	}

	void Image::cutImage()
	{
		cv::Mat _temp;
		cv::Mat _temp2;
		cv::Rect firstROI((double)_image.cols * 0.09, (double)_image.rows * 0.53, (double)_image.cols * 0.81, (double)_image.rows * 0.11); // cut number part
		_temp = _edge(firstROI);
		_temp2 = _image(firstROI);

		int top = -1;
		int bottom = -1;
		int left = -1;
		int right = -1;
		Vec3b *pixel;

		for (int i = 0; i < _temp.rows; ++i){			// find borders
			pixel = _temp.ptr<Vec3b>(i);
			for (int j = 0; j < _temp.cols; ++j){
				if (pixel[j][0] + pixel[j][1] + pixel[j][2] > 384){
					if (top == -1)
						top = i;
					else
						bottom = i;
					if (left == -1 || j < left)
						left = j;
					else if (j > right)
						right = j;
				}
			}
		}

		if (top == -1 || bottom == -1 || bottom - top <= 1){	// cannot find any number in the image, error
			std::stringstream ss;
			ss << "Error: Could not find numbers in the picture " << _path;
			throw std::invalid_argument(ss.str());
		}

		int allowance = (bottom - top) * 0.25;

		cv::Rect secondROI(left, top, right - left, bottom - top); // cut number part
		_cut = _temp2(secondROI);
	}

	void Image::binarization() noexcept
	{
		int allowance = _cut.rows * 0.25;		// add some border around text for good individual cut
		copyMakeBorder(_cut, _cut, allowance, allowance, allowance, allowance, cv::BORDER_REFLECT, cv::Scalar(255, 255, 255));

		Vec3b *pixel;

		for (int i = 0; i < _cut.rows; ++i){			// find borders
			pixel = _cut.ptr<Vec3b>(i);
			for (int j = 0; j < _cut.cols; ++j){
				if (pixel[j][0] + pixel[j][1] + pixel[j][2] > 384){
					pixel[j][0] = 0;
					pixel[j][1] = 0;
					pixel[j][2] = 0;
				} else {
					pixel[j][0] = 255;
					pixel[j][1] = 255;
					pixel[j][2] = 255;
				}
			}
		}
	}

	void Image::getResult() noexcept
	{
		cv::Mat _temp;

		for (int i = 0; i < 19; ++i){
			if (i == 4 || i == 9 || i == 14){
				std::cout << " ";
			} else {
				cv::Rect ROI((i * _cut.cols) / 19, 0, _cut.cols / 19, _cut.rows);
				// here cut individual numbers out
				_temp = _cut(ROI);
				_temp = resizeNumber(_temp);
				std::cout << templateMatching(_temp);
			}
		}
		std::cout << std::endl;
	}

	cv::Mat Image::resizeNumber(cv::Mat& input) noexcept
	{
		cv::Mat output;
		int y1 = 0;
		int y2 = 0;
		int x1 = 0;
		bool found = false;
		Vec3b* pixel;

		for (int y = 0; y < input.rows && !found; ++y) {
			pixel = input.ptr<Vec3b>(y);
			for (int x = 0; x < input.cols; ++x) {
				if (!pixel[x][0]) {
					y1 = y;
					found = true;
					break;
				}
			}
		}
		found = false;
		for (int y = input.rows - 1; y >= 0 && !found; --y) {
			pixel = input.ptr<Vec3b>(y);
			for (int x = 0; x < input.cols; ++x) {
				if (!pixel[x][0]) {
					y2 = y;
					found = true;
					break;
				}
			}
		}
		x1 = input.cols;
		for (int y = input.rows - 1; y >= 0; --y) {
			pixel = input.ptr<Vec3b>(y);
			for (int x = 0; x < x1; ++x) {
				if (!pixel[x][0]) {
					x1 = x;
					break;
				}
			}
		}
		int ysize = (y2 - y1) + 1;
		double ratio = 49.0 / ysize;
		int xsize = 32.0 / ratio;
		if (xsize + x1 >= input.cols) {
			xsize = input.cols - x1;
		}
		auto tmp = input(cv::Rect(x1, y1, xsize, ysize));
		cv::resize(tmp, output, cv::Size(), ratio, ratio);
		return output;
	}

	int Image::templateMatching(cv::Mat &input) noexcept
	{
		int crop_width = _template_image[0].cols; //32
		int crop_height = _template_image[0].rows; //49
		std::vector<int> sumB(10, 0);
		std::vector<int> sumW(10, 0);
		int nBlack = 0;
		std::vector<int> nBlack2(10, 0);
		std::vector<int> nWhite2(10, 0);
		std::vector<float> ratio(10, 0);

		for (int k = 0; k < 10; k++) {
			for (int i = 0; i < input.rows && i < crop_height; ++i) {
				auto pixel = input.ptr<Vec3b>(i);
				auto pixel2 = (_template_image[k].ptr<uchar>)(i);
				for (int j = 0; j < input.cols && j < crop_width; ++j) {
					if ((pixel[j][0] < 127 && pixel2[j] < 127)) {
						sumB[k]++;
					}
					if ((pixel[j][0] > 127 && pixel2[j] > 127)) {
						sumW[k]++;
					}
					if ((pixel2[j] < 127)) {
						nBlack2[k]++;
					}
					else {
						nWhite2[k]++;
					}
				}
			}
		}
		float total = crop_height * crop_width;
		for (int i = 0; i < 10; i++) {
			if (nBlack2[i] && nWhite2[i])
				ratio[i] = (float)sumB[i] / (float)total + (float)sumW[i] / (float)total;
		}
		return std::max_element(ratio.begin(), ratio.end()) - ratio.begin();
	}

	void Image::edgeDetection() noexcept
	{
		edgeGrayscale();
		edgeBlur();
		edgeGradient();
		edgeRemoveNonMaxGradient();
		edgeFilter(20, 25);
		cutImage();
		binarization();
		getResult();
	}

	void Image::show() noexcept
	{
		cv::namedWindow(_path, cv::WINDOW_AUTOSIZE);
		cv::imshow(_path, _cut);
		cv::waitKey(0);
	}
}
