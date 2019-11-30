#include <sstream>
#include "Image.h"

namespace imgrecog {
	Image::Image(const std::string& path)
		: _path(path),
		_image(cv::imread(path, cv::IMREAD_COLOR))
	{
		if (!_image.data) {
			std::stringstream ss;
			ss << "Error: Could not open or find the image: " << _path;
			throw std::invalid_argument(ss.str());
		}
	}

	void Image::show() noexcept
	{
		cv::namedWindow("Show image", cv::WINDOW_AUTOSIZE);
		cv::imshow("Show image", _image);
		cv::waitKey(0);
	}
}
