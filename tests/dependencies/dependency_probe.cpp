#include <pinocchio/multibody/model.hpp>
#include <zmq.h>

#include <iostream>

int main() {
    pinocchio::Model model;
    int major = 0;
    int minor = 0;
    int patch = 0;
    zmq_version(&major, &minor, &patch);

    if (model.nq != 0 || major != ZMQ_VERSION_MAJOR ||
        minor != ZMQ_VERSION_MINOR || patch != ZMQ_VERSION_PATCH) {
        return 1;
    }

    std::cout << "Pinocchio model construction: OK\n";
    std::cout << "ZeroMQ version: " << major << '.' << minor << '.' << patch
              << '\n';
    return 0;
}
