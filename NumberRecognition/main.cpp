#include <iostream>
#include <filesystem>
#include "Image.h"

int main(int ac, char** av)
{
	if (ac != 2) {
		std::cerr
			<< "Usage:" << std::endl
			<< av[0] << " path_image" << std::endl;
		return EXIT_FAILURE;
	}
	try {
		imgrecog::Image image(av[1]);
		image.show();
	}
	catch (std::exception &e) {
		std::cerr << "Error: " << e.what() << std::endl;
		return EXIT_FAILURE;
	}
	return EXIT_SUCCESS;
}
