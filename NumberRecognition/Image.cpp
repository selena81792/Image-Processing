#include <sstream>
#include <iostream>
#define _USE_MATH_DEFINES
#include <cmath>
#include <string>
#include "Image.h"
#include "opencv2/opencv.hpp"

using namespace cv;

namespace imgrecog {
	Image::Image(const std::string& path)
		: _path(path),
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

				uint8_t grayscale = .2126 * bgrPixel.val[2] + .7152 * bgrPixel.val[1] + .0722 * bgrPixel.val[0];

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

		size_t offset = kernel.size() / 2;
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
		cv::Rect firstROI(_image.cols * 0.09, _image.rows * 0.53, _image.cols * 0.81, _image.rows * 0.11); // cut number part
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
				std::cout << templateMatching(_temp);
			}
		}
		std::cout << std::endl;
	}

	int Image::templateMatching(cv::Mat input) noexcept
	{
		// template function to match the font and return the correct number

		imshow("input", input);

		return 6;
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
		// cv::namedWindow("Show image", cv::WINDOW_AUTOSIZE);
		// cv::imshow("Show image", _image);
		// cv::namedWindow("Show edge", cv::WINDOW_AUTOSIZE);
		// cv::imshow("Show edge", _edge);

		cv::namedWindow(_path, cv::WINDOW_AUTOSIZE);
		cv::imshow(_path, _cut);
		cv::waitKey(0);
	}
}
