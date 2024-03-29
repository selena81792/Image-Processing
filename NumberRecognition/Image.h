#pragma once
#include <string>
#include <set>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

namespace imgrecog {
	class Image {
		const std::vector<std::string> _path_templates;
		const std::vector<cv::Mat> _template_image;

		const std::string _path;
		cv::Mat _image;
		cv::Mat _edge;
		cv::Mat _cut;
		std::set<std::pair<int, int>> _edgePoint;
		std::vector<std::vector<double>> _gradient;
		std::vector<std::vector<double>> _direction;

		void edgeDetection() noexcept;
		void edgeGrayscale() noexcept;
		void edgeBlur() noexcept;
		void edgeGradient() noexcept;
		void edgeRemoveNonMaxGradient() noexcept;
		void edgeFilter(int low, int high) noexcept;
		void cutImage();
		void binarization() noexcept;
		void getResult() noexcept;
		cv::Mat resizeNumber(cv::Mat &input) noexcept;
		int templateMatching(cv::Mat &input) noexcept;
	public:
		Image(const std::string& path);
		~Image() noexcept = default;

		void show() noexcept;
	};
}
