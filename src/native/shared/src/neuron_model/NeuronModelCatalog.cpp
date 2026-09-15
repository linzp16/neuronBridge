#include "neuron_model/GeneratedNeuronCatalog.h"
#include "neuron_model/NeuronModelCatalog.h"

#include <algorithm>
#include <sstream>
#include <stdexcept>
#include <unordered_set>

namespace npgr {
namespace {

NeuronModelBackendBinding Binding(NeuronBackend backend,
                                  const char* implementation_name,
                                  bool supported,
                                  bool preferred = false) {
    NeuronModelBackendBinding binding;
    binding.backend = backend;
    binding.implementation_name = implementation_name != nullptr ? implementation_name : "";
    binding.supported = supported;
    binding.preferred = preferred;
    return binding;
}

bool NameMatches(const NeuronModelCatalogEntry& entry, const std::string& name) {
    if (entry.canonical_name == name) {
        return true;
    }
    return std::find(entry.aliases.begin(), entry.aliases.end(), name) != entry.aliases.end();
}

const NeuronModelBackendBinding* FindBinding(const NeuronModelCatalogEntry& entry,
                                             NeuronBackend backend) {
    for (std::size_t index = 0; index < entry.backends.size(); ++index) {
        if (entry.backends[index].backend == backend) {
            return &entry.backends[index];
        }
    }
    return nullptr;
}

const char* BackendName(NeuronBackend backend) {
    switch (backend) {
        case NeuronBackend::LegacyCpu:
            return "legacy CPU";
        case NeuronBackend::LegacyGpu:
            return "legacy GPU";
        case NeuronBackend::DenseGpu:
            return "dense GPU";
        default:
            return "unknown";
    }
}

}  // namespace

NeuronModelCatalog& NeuronModelCatalog::Instance() {
    static NeuronModelCatalog catalog;
    return catalog;
}

NeuronModelCatalog::NeuronModelCatalog() {
    entries_ = {
        {
            "TimeDrivenLIF_Exponential_double",
            {"TimeDrivenLIF_Exponential_double_GPU"},
            {
                Binding(NeuronBackend::LegacyCpu, "TimeDrivenLIF_Exponential_double", true, true),
                Binding(NeuronBackend::LegacyGpu, "TimeDrivenLIF_Exponential_double_GPU", true, true),
                Binding(NeuronBackend::DenseGpu, "TimeDrivenLIF_Exponential_double", true, true),
            },
            NeuronModelCategory::Core,
            NeuronModelRole::OrdinaryCore,
            true,
        },
        {
            "TimeDrivenLIF_Exponential_Decay",
            {"TimeDrivenLIF_Exponential_Decay_GPU"},
            {
                Binding(NeuronBackend::LegacyCpu, "TimeDrivenLIF_Exponential_Decay", true, true),
                Binding(NeuronBackend::LegacyGpu, "TimeDrivenLIF_Exponential_Decay_GPU", true, true),
                Binding(NeuronBackend::DenseGpu, "TimeDrivenLIF_Exponential_Decay", true, true),
            },
            NeuronModelCategory::Core,
            NeuronModelRole::OrdinaryCore,
            true,
        },
        {
            "TimeDrivenIzhikevic_Exponential_Decay",
            {"TimeDrivenIzhikevic_I_Exponential_Decay",
             "TimeDrivenIzhikevic_I_Exponential_Decay_GPU",
             "TimeDrivenIzhikevic_Exponential_Decay_GPU"},
            {
                Binding(NeuronBackend::LegacyCpu, "TimeDrivenIzhikevic_I_Exponential_Decay", true, true),
                Binding(NeuronBackend::LegacyGpu, "TimeDrivenIzhikevic_Exponential_Decay_GPU", true, true),
                Binding(NeuronBackend::DenseGpu, "TimeDrivenIzhikevic_Exponential_Decay", true, true),
            },
            NeuronModelCategory::Core,
            NeuronModelRole::OrdinaryCore,
            true,
        },
        {
            "PoissonRate",
            {"PoissonRate_GPU"},
            {
                Binding(NeuronBackend::LegacyCpu, "PoissonRate", true, true),
                Binding(NeuronBackend::LegacyGpu, "PoissonRate_GPU", true, true),
                Binding(NeuronBackend::DenseGpu, "PoissonRate", true, true),
            },
            NeuronModelCategory::Core,
            NeuronModelRole::OrdinaryCore,
            true,
        },
        {
            "TimeDrivenLIF_Voltage_jump",
            {"TimeDrivenLIF_Voltage_jump_GPU"},
            {
                Binding(NeuronBackend::LegacyCpu, "TimeDrivenLIF_Voltage_jump", true, true),
                Binding(NeuronBackend::LegacyGpu, "TimeDrivenLIF_Voltage_jump_GPU", true, true),
                Binding(NeuronBackend::DenseGpu, "TimeDrivenLIF_Voltage_jump", true, true),
            },
            NeuronModelCategory::Core,
            NeuronModelRole::OrdinaryCore,
            true,
        },
        {
            "TimeDrivenLIF_Exponential_triple",
            {"TimeDrivenLIF_Exponential_triple_GPU"},
            {
                Binding(NeuronBackend::LegacyCpu, "TimeDrivenLIF_Exponential_triple", true, true),
                Binding(NeuronBackend::LegacyGpu, "TimeDrivenLIF_Exponential_triple_GPU", true, true),
                Binding(NeuronBackend::DenseGpu, "TimeDrivenLIF_Exponential_triple", true, true),
            },
            NeuronModelCategory::Core,
            NeuronModelRole::OrdinaryCore,
            true,
        },
        {
            "InputSpikeNeuronModel",
            {},
            {
                Binding(NeuronBackend::LegacyCpu, "InputSpikeNeuronModel", true, true),
                Binding(NeuronBackend::LegacyGpu, "InputSpikeNeuronModel", true, true),
                Binding(NeuronBackend::DenseGpu, "", false),
            },
            NeuronModelCategory::Input,
            NeuronModelRole::InputSpike,
            true,
        },
        {
            "InputCurrentNeuronModel",
            {},
            {
                Binding(NeuronBackend::LegacyCpu, "InputCurrentNeuronModel", true, true),
                Binding(NeuronBackend::LegacyGpu, "InputCurrentNeuronModel", true, true),
                Binding(NeuronBackend::DenseGpu, "", false),
            },
            NeuronModelCategory::Input,
            NeuronModelRole::InputCurrent,
            true,
        },
        {
            "TriggerRelayNeuronModel",
            {"TriggerRelayNeuron",
             "TriggerRelayNeuronModel_GPU",
             "DenseTriggerRelayNeuron",
             "DenseTriggerRelayNeuronModel"},
            {
                Binding(NeuronBackend::LegacyCpu, "TriggerRelayNeuronModel", true, true),
                Binding(NeuronBackend::LegacyGpu, "TriggerRelayNeuronModel", true, true),
                Binding(NeuronBackend::DenseGpu, "TriggerRelayNeuronModel", true, true),
            },
            NeuronModelCategory::Core,
            NeuronModelRole::TriggerRelay,
            true,
        },
        {
            "HandwritingTimeDrivenModel",
            {},
            {
                Binding(NeuronBackend::LegacyCpu, "HandwritingTimeDrivenModel", true, true),
                Binding(NeuronBackend::LegacyGpu, "HandwritingTimeDrivenModel", true, true),
                Binding(NeuronBackend::DenseGpu, "", false),
            },
            NeuronModelCategory::Special,
            NeuronModelRole::Special,
            true,
        },
        {
            "EDLUTLikeLIF",
            {"EDLUTLikeLIF_GPU"},
            {
                Binding(NeuronBackend::LegacyCpu, "", false),
                Binding(NeuronBackend::LegacyGpu, "EDLUTLikeLIF_GPU", true, true),
                Binding(NeuronBackend::DenseGpu, "", false),
            },
            NeuronModelCategory::Special,
            NeuronModelRole::Special,
            true,
        },
    };

    std::vector<NeuronModelCatalogEntry> generated_entries;
    RegisterGeneratedNeuronCatalogEntries(&generated_entries);
    RegisterGeneratedEntries(generated_entries);
}

const std::vector<NeuronModelCatalogEntry>& NeuronModelCatalog::Entries() const noexcept {
    return entries_;
}

void NeuronModelCatalog::RegisterGeneratedEntries(
    const std::vector<NeuronModelCatalogEntry>& entries) {
    for (const NeuronModelCatalogEntry& entry : entries) {
        ValidateEntryOrThrow(entry);
        entries_.push_back(entry);
    }
}

void NeuronModelCatalog::ValidateEntryOrThrow(
    const NeuronModelCatalogEntry& candidate) const {
    if (candidate.canonical_name.empty()) {
        throw std::runtime_error("generated neuron catalog entry has an empty canonical name");
    }

    std::unordered_set<std::string> candidate_names;
    candidate_names.insert(candidate.canonical_name);
    for (const std::string& alias : candidate.aliases) {
        if (alias.empty() || !candidate_names.insert(alias).second) {
            throw std::runtime_error(
                "generated neuron catalog entry has an empty or duplicate alias: " +
                candidate.canonical_name);
        }
    }

    bool has_supported_backend = false;
    std::unordered_set<int> candidate_backends;
    for (const NeuronModelBackendBinding& binding : candidate.backends) {
        const int backend = static_cast<int>(binding.backend);
        if (!candidate_backends.insert(backend).second) {
            throw std::runtime_error(
                "generated neuron catalog entry has duplicate backend bindings: " +
                candidate.canonical_name);
        }
        if (binding.supported && binding.implementation_name.empty()) {
            throw std::runtime_error(
                "generated neuron catalog entry has an empty supported implementation: " +
                candidate.canonical_name);
        }
        has_supported_backend = has_supported_backend || binding.supported;
    }
    if (!has_supported_backend) {
        throw std::runtime_error(
            "generated neuron catalog entry has no supported backend: " +
            candidate.canonical_name);
    }

    for (const NeuronModelCatalogEntry& existing : entries_) {
        std::unordered_set<std::string> existing_names;
        existing_names.insert(existing.canonical_name);
        existing_names.insert(existing.aliases.begin(), existing.aliases.end());
        for (const std::string& name : candidate_names) {
            if (existing_names.find(name) != existing_names.end()) {
                throw std::runtime_error(
                    "generated neuron catalog entry conflicts with an existing name: " + name);
            }
        }
        for (const NeuronModelBackendBinding& candidate_binding : candidate.backends) {
            if (!candidate_binding.supported) {
                continue;
            }
            for (const NeuronModelBackendBinding& existing_binding : existing.backends) {
                if (existing_binding.supported &&
                    existing_binding.backend == candidate_binding.backend &&
                    existing_binding.implementation_name == candidate_binding.implementation_name) {
                    throw std::runtime_error(
                        "generated neuron catalog entry conflicts with an existing implementation: " +
                        candidate_binding.implementation_name);
                }
            }
        }
    }
}

const NeuronModelCatalogEntry* NeuronModelCatalog::Resolve(const std::string& name) const {
    for (std::size_t index = 0; index < entries_.size(); ++index) {
        if (NameMatches(entries_[index], name)) {
            return &entries_[index];
        }
    }
    return nullptr;
}

std::string NeuronModelCatalog::ResolveCanonicalName(const std::string& name) const {
    const NeuronModelCatalogEntry* entry = Resolve(name);
    return entry != nullptr ? entry->canonical_name : name;
}

bool NeuronModelCatalog::IsSupported(const std::string& name, NeuronBackend backend) const {
    const NeuronModelCatalogEntry* entry = Resolve(name);
    if (entry == nullptr) {
        return false;
    }
    const NeuronModelBackendBinding* binding = FindBinding(*entry, backend);
    return binding != nullptr && binding->supported;
}

std::string NeuronModelCatalog::ResolveImplementationName(const std::string& name,
                                                         NeuronBackend backend,
                                                         std::string* reason) const {
    const NeuronModelCatalogEntry* entry = Resolve(name);
    if (entry == nullptr) {
        if (reason != nullptr) {
            *reason = "unknown neuron model: " + name;
        }
        return "";
    }
    const NeuronModelBackendBinding* binding = FindBinding(*entry, backend);
    if (binding == nullptr || !binding->supported || binding->implementation_name.empty()) {
        if (reason != nullptr) {
            std::ostringstream oss;
            oss << "neuron model " << entry->canonical_name
                << " is not supported by " << BackendName(backend);
            *reason = oss.str();
        }
        return "";
    }
    return binding->implementation_name;
}

bool NeuronModelCatalog::IsPublicUserModel(const std::string& name) const {
    const NeuronModelCatalogEntry* entry = Resolve(name);
    return entry != nullptr && entry->public_user_model;
}

NeuronModelCategory NeuronModelCatalog::CategoryOf(const std::string& name) const {
    const NeuronModelCatalogEntry* entry = Resolve(name);
    return entry != nullptr ? entry->category : NeuronModelCategory::Special;
}

NeuronModelRole NeuronModelCatalog::RoleOf(const std::string& name) const {
    const NeuronModelCatalogEntry* entry = Resolve(name);
    return entry != nullptr ? entry->role : NeuronModelRole::Special;
}

bool NeuronModelCatalog::IsInputModel(const std::string& name) const {
    const NeuronModelRole role = RoleOf(name);
    return role == NeuronModelRole::InputSpike ||
           role == NeuronModelRole::InputCurrent;
}

bool NeuronModelCatalog::IsDenseOnlyHelperModel(const std::string& name) const {
    return RoleOf(name) == NeuronModelRole::DenseOnlyHelper;
}

bool NeuronModelCatalog::IsOrdinaryCoreModel(const std::string& name) const {
    return RoleOf(name) == NeuronModelRole::OrdinaryCore;
}

std::vector<std::string> NeuronModelCatalog::PublicModelNames() const {
    std::vector<std::string> names;
    for (std::size_t index = 0; index < entries_.size(); ++index) {
        if (entries_[index].public_user_model) {
            names.push_back(entries_[index].canonical_name);
        }
    }
    return names;
}

std::vector<std::string> NeuronModelCatalog::PublicOrdinaryCoreModelNames() const {
    std::vector<std::string> names;
    for (std::size_t index = 0; index < entries_.size(); ++index) {
        if (entries_[index].public_user_model &&
            entries_[index].role == NeuronModelRole::OrdinaryCore) {
            names.push_back(entries_[index].canonical_name);
        }
    }
    return names;
}

std::vector<std::string> NeuronModelCatalog::PublicInputModelNames() const {
    std::vector<std::string> names;
    for (std::size_t index = 0; index < entries_.size(); ++index) {
        if (!entries_[index].public_user_model) {
            continue;
        }
        if (entries_[index].role == NeuronModelRole::InputSpike ||
            entries_[index].role == NeuronModelRole::InputCurrent) {
            names.push_back(entries_[index].canonical_name);
        }
    }
    return names;
}

std::vector<std::string> NeuronModelCatalog::PublicSpecialModelNames() const {
    std::vector<std::string> names;
    for (std::size_t index = 0; index < entries_.size(); ++index) {
        if (entries_[index].public_user_model &&
            entries_[index].role == NeuronModelRole::Special) {
            names.push_back(entries_[index].canonical_name);
        }
    }
    return names;
}

std::vector<std::string> NeuronModelCatalog::PublicDenseOnlyHelperModelNames() const {
    std::vector<std::string> names;
    for (std::size_t index = 0; index < entries_.size(); ++index) {
        if (entries_[index].public_user_model &&
            entries_[index].role == NeuronModelRole::DenseOnlyHelper) {
            names.push_back(entries_[index].canonical_name);
        }
    }
    return names;
}

std::vector<std::string> NeuronModelCatalog::SupportedModelNames(NeuronBackend backend) const {
    std::vector<std::string> names;
    for (std::size_t index = 0; index < entries_.size(); ++index) {
        const NeuronModelBackendBinding* binding = FindBinding(entries_[index], backend);
        if (binding != nullptr && binding->supported) {
            names.push_back(entries_[index].canonical_name);
        }
    }
    return names;
}

}  // namespace npgr
