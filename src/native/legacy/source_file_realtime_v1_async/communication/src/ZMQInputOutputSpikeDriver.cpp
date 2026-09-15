#include "../source_file_realtime_v1_async/communication/inc/ZMQInputOutputSpikeDriver.h"

#include "../source_file_realtime_v1_async/Event/inc/Spike/InputSpike.h"

#include <cstddef>
#include <cstring>
#include <stdexcept>

namespace {

// Keep the ZMQ payload bounded so a malformed peer cannot allocate unbounded memory.
static const uint32_t kMaxSpikesPerBatch = 10000000u;

}  // namespace

ZMQInputOutputSpikeDriver::ZMQInputOutputSpikeDriver(enum DriverType Type,
                                                     std::string server_address,
                                                     unsigned short tcp_port)
    : connection_type(Type) {
    if (Type == SERVER) {
        socket_ = new ZmqSocket(ZmqSocket::Mode::REPLY, "", static_cast<int>(tcp_port));
    } else if (Type == CLIENT) {
        socket_ = new ZmqSocket(ZmqSocket::Mode::REQUEST, server_address, static_cast<int>(tcp_port));
    }
    isFinished = false;
}

ZMQInputOutputSpikeDriver::~ZMQInputOutputSpikeDriver() {
    delete socket_;
}

void ZMQInputOutputSpikeDriver::LoadInputSpike(EventQueue* eventQueue, Network* network, int time) {
    uint32_t time_step = static_cast<uint32_t>(time);
    uint32_t output_count = static_cast<uint32_t>(OutputBuffer.size());

    if (OutputBuffer.size() > kMaxSpikesPerBatch) {
        throw std::runtime_error("Spike count exceeds kMaxSpikesPerBatch");
    }

    std::size_t request_size = sizeof(time_step) + sizeof(output_count);
    if (output_count > 0) {
        request_size += output_count * sizeof(OutputSpikeIO);
    }

    std::vector<char> request_buf(request_size);
    char* ptr = request_buf.data();

    std::memcpy(ptr, &time_step, sizeof(time_step));
    ptr += sizeof(time_step);
    std::memcpy(ptr, &output_count, sizeof(output_count));
    ptr += sizeof(output_count);

    if (output_count > 0) {
        std::memcpy(ptr, OutputBuffer.data(), output_count * sizeof(OutputSpikeIO));
    }

    socket_->sendBuffer(request_buf.data(), static_cast<int>(request_size));

    uint32_t input_count = 0;
    int bytes = socket_->receiveBuffer(&input_count, sizeof(input_count));
    if (bytes != static_cast<int>(sizeof(input_count))) {
        throw std::runtime_error("Failed to receive input spike count");
    }
    if (input_count > kMaxSpikesPerBatch) {
        throw std::runtime_error("Input spike batch too large");
    }

    if (input_count <= 0) {
        return;
    }

    std::vector<OutputSpikeIO> input_spikes(input_count);
    std::size_t data_size = input_count * sizeof(OutputSpikeIO);
    bytes = socket_->receiveBuffer(input_spikes.data(), static_cast<int>(data_size));
    if (static_cast<std::size_t>(bytes) != data_size) {
        throw std::runtime_error("Incomplete input spike data");
    }

    // ZMQ input spikes are translated back into regular input-spike events.
    for (std::size_t spike_index = 0; spike_index < input_spikes.size(); ++spike_index) {
        const OutputSpikeIO& input_spike = input_spikes[spike_index];
        if (input_spike.neuron < 0 || input_spike.neuron >= network->neuronsNum) {
            continue;
        }
        Neuron* target_neuron = &(network->neurons[input_spike.neuron]);
        int event_time = static_cast<int>(input_spike.time / input_spike.basetimestepl);
        InputSpike* spike = new InputSpike(target_neuron, event_time, target_neuron->Queue_index);
        eventQueue->Insert_a_Event(spike, spike->getIndex());
    }
}

void ZMQInputOutputSpikeDriver::LoadInputSpike(EventQueue* eventQueue,
                                               Network* network,
                                               int spikenum,
                                               const int* Times,
                                               const int* neuron_index) {
    return;
}

void ZMQInputOutputSpikeDriver::WriteSpike(Spike* NewSpike, float basetimesteps) {
    if (!NewSpike) {
        return;
    }

    OutputSpikeIO out_spike(NewSpike->SourceNeuron->Neuron_index, NewSpike->getTime(), basetimesteps);

#pragma omp critical(ZMQInputOutputSpikeDriver_WriteSpike)
    {
        OutputBuffer.push_back(out_spike);
    }
}

bool ZMQInputOutputSpikeDriver::IsBuffered() {
    return true;
}

void ZMQInputOutputSpikeDriver::FlushBuffers() {
    OutputBuffer.clear();
}
