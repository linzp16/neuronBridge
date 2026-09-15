#ifndef NPGR_GENERATED_NEURON_CATALOG_H
#define NPGR_GENERATED_NEURON_CATALOG_H

#include "neuron_model/NeuronModelCatalog.h"

namespace npgr {

// The build always links either the generated implementation or the stable
// empty fallback, so Catalog initialization never depends on optional files.
void RegisterGeneratedNeuronCatalogEntries(
    std::vector<NeuronModelCatalogEntry>* entries);

}  // namespace npgr

#endif
