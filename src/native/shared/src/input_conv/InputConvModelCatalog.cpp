#include "input_conv/InputConvModelCatalog.h"

#include <algorithm>

namespace npgr {
namespace {

bool NameMatches(const InputConvModelCatalogEntry& entry, const std::string& name) {
    if (entry.canonical_name == name || entry.implementation_name == name) {
        return true;
    }
    return std::find(entry.aliases.begin(), entry.aliases.end(), name) != entry.aliases.end();
}

}  // namespace

InputConvModelCatalog& InputConvModelCatalog::Instance() {
    static InputConvModelCatalog catalog;
    return catalog;
}

InputConvModelCatalog::InputConvModelCatalog() {
    entries_ = {
        {
            "InputConvV1",
            {},
            "InputConvV1",
            true,
            true,
            true,
        },
    };
}

const InputConvModelCatalogEntry* InputConvModelCatalog::Resolve(const std::string& name) const {
    for (std::size_t index = 0; index < entries_.size(); ++index) {
        if (NameMatches(entries_[index], name)) {
            return &entries_[index];
        }
    }
    return nullptr;
}

std::string InputConvModelCatalog::ResolveCanonicalName(const std::string& name) const {
    const InputConvModelCatalogEntry* entry = Resolve(name);
    return entry != nullptr ? entry->canonical_name : name;
}

std::string InputConvModelCatalog::ResolveImplementationName(const std::string& name,
                                                            std::string* reason) const {
    const InputConvModelCatalogEntry* entry = Resolve(name);
    if (entry == nullptr) {
        if (reason != nullptr) {
            *reason = "unknown input conv model: " + name;
        }
        return "";
    }
    if (entry->implementation_name.empty()) {
        if (reason != nullptr) {
            *reason = "input conv model has no implementation: " + entry->canonical_name;
        }
        return "";
    }
    return entry->implementation_name;
}

bool InputConvModelCatalog::IsPublicUserModel(const std::string& name) const {
    const InputConvModelCatalogEntry* entry = Resolve(name);
    return entry != nullptr && entry->public_user_model;
}

bool InputConvModelCatalog::ValidateOutputTarget(const std::string& name,
                                                 bool target_is_dense_subnetwork,
                                                 std::string* reason) const {
    const InputConvModelCatalogEntry* entry = Resolve(name);
    if (entry == nullptr) {
        if (reason != nullptr) {
            *reason = "unknown input conv model: " + name;
        }
        return false;
    }
    if (target_is_dense_subnetwork && !entry->supports_dense_subnetwork_output) {
        if (reason != nullptr) {
            *reason = "input conv model does not support dense subnetwork output: " +
                      entry->canonical_name;
        }
        return false;
    }
    if (!target_is_dense_subnetwork && !entry->supports_main_network_output) {
        if (reason != nullptr) {
            *reason = "input conv model does not support main network output: " +
                      entry->canonical_name;
        }
        return false;
    }
    return true;
}

std::vector<std::string> InputConvModelCatalog::PublicModelNames() const {
    std::vector<std::string> names;
    for (std::size_t index = 0; index < entries_.size(); ++index) {
        if (entries_[index].public_user_model) {
            names.push_back(entries_[index].canonical_name);
        }
    }
    return names;
}

}  // namespace npgr
