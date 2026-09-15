#include "../source_file_realtime_v1_async/communication/inc/FileOutputSpikeDriver.h"
#include "../source_file_realtime_v1_async/Event/inc/Spike/Spike.h"
#include <filesystem>
#include <stdexcept>

FileOutputSpikeDriver::FileOutputSpikeDriver(const char* filename) : filename(filename) {
	std::filesystem::path file_path(filename);
	std::filesystem::path parent_path = file_path.parent_path();
	if (!parent_path.empty()) {
		std::filesystem::create_directories(parent_path);
	}

	this->Handler = fopen(filename, "wt");
	if (this->Handler == nullptr) {
		throw std::runtime_error(std::string("Failed to open output file: ") + filename);
	}
}

FileOutputSpikeDriver::~FileOutputSpikeDriver() {
	if (this->Handler) {
		fclose(this->Handler);
		this->Handler = NULL;
	}
}

void FileOutputSpikeDriver::WriteSpike(Spike* NewSpike, float basetimestep) {
#pragma omp critical (FileOutputSpikeDriver)
	{
		if (this->Handler == nullptr) {
			throw std::runtime_error(std::string("Output spike file handle is null: ") + this->filename);
		}
		fprintf(this->Handler, "%d\t%li\n", NewSpike->getTime(), NewSpike->SourceNeuron->Neuron_index);
	}
}

bool FileOutputSpikeDriver::IsBuffered() {
	return false;
}


void FileOutputSpikeDriver::FlushBuffers() {
	return;
}
