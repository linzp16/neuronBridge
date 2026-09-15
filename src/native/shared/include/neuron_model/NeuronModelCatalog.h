#ifndef NPGR_NEURON_MODEL_CATALOG_H
#define NPGR_NEURON_MODEL_CATALOG_H

#include <string>
#include <vector>

namespace npgr {

enum class NeuronBackend {
    LegacyCpu,
    LegacyGpu,
    DenseGpu,
};

enum class NeuronModelCategory {
    Core,
    Input,
    Helper,
    Special,
};

enum class NeuronModelRole {
    OrdinaryCore,
    InputSpike,
    InputCurrent,
    TriggerRelay,
    Special,
    DenseOnlyHelper,
};

struct NeuronModelBackendBinding {
    NeuronBackend backend = NeuronBackend::LegacyCpu;
    std::string implementation_name;
    bool supported = false;
    bool preferred = false;
};

struct NeuronModelCatalogEntry {
    std::string canonical_name;
    std::vector<std::string> aliases;
    std::vector<NeuronModelBackendBinding> backends;
    NeuronModelCategory category = NeuronModelCategory::Core;
    NeuronModelRole role = NeuronModelRole::OrdinaryCore;
    bool public_user_model = true;
};

// Central semantic model catalog shared by legacy CPU/GPU and dense runtimes.
// Factories remain responsible for constructing concrete implementation
// classes; this catalog owns names, aliases, categories, and backend support.
class NeuronModelCatalog {
public:
    static NeuronModelCatalog& Instance();

    const std::vector<NeuronModelCatalogEntry>& Entries() const noexcept;
    const NeuronModelCatalogEntry* Resolve(const std::string& name) const;
    std::string ResolveCanonicalName(const std::string& name) const;
    bool IsSupported(const std::string& name, NeuronBackend backend) const;
    std::string ResolveImplementationName(const std::string& name,
                                          NeuronBackend backend,
                                          std::string* reason = nullptr) const;
    bool IsPublicUserModel(const std::string& name) const;
    NeuronModelCategory CategoryOf(const std::string& name) const;
    NeuronModelRole RoleOf(const std::string& name) const;
    bool IsInputModel(const std::string& name) const;
    bool IsDenseOnlyHelperModel(const std::string& name) const;
    bool IsOrdinaryCoreModel(const std::string& name) const;
    std::vector<std::string> PublicModelNames() const;
    std::vector<std::string> PublicOrdinaryCoreModelNames() const;
    std::vector<std::string> PublicInputModelNames() const;
    std::vector<std::string> PublicSpecialModelNames() const;
    std::vector<std::string> PublicDenseOnlyHelperModelNames() const;
    std::vector<std::string> SupportedModelNames(NeuronBackend backend) const;

private:
    NeuronModelCatalog();

    void RegisterGeneratedEntries(
        const std::vector<NeuronModelCatalogEntry>& entries);
    void ValidateEntryOrThrow(const NeuronModelCatalogEntry& candidate) const;

    std::vector<NeuronModelCatalogEntry> entries_;
};

}  // namespace npgr

#endif
