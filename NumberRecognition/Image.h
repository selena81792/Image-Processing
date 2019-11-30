#pragma once
#include <string>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

namespace imgrecog {
	class Image {
		const std::string _path;
		cv::Mat _image;

	public:
		Image(const std::string& path);
		~Image() noexcept = default;

		void show() noexcept;
	};
}
