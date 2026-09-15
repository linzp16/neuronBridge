#ifndef NPGR_INPUT_CONV_MODEL_CATALOG_H
#define NPGR_INPUT_CONV_MODEL_CATALOG_H

#include <string>
#include <vector>

namespace npgr {

struct InputConvModelCatalogEntry {
    std::string canonical_name;
    std::vector<std::string> aliases;
    std::string implementation_name;
    bool supports_main_network_output = true;
    bool supports_dense_subnetwork_output = true;
    bool public_user_model = true;
};

// Central semantic catalog for InputConv models.
//
// The catalog owns user-facing names and aliases. Concrete factories still own
// object construction, so simulation setup does not depend on model-specific
// allocation details.
class InputConvModelCatalog {
public:
    static InputConvModelCatalog& Instance();

    const InputConvModelCatalogEntry* Resolve(const std::string& name) const;
    std::string ResolveCanonicalName(const std::string& name) const;
    std::string ResolveImplementationName(const std::string& name,
                                          std::string* reason = nullptr) const;
    bool ValidateOutputTarget(const std::string& name,
                              bool target_is_dense_subnetwork,
                              std::string* reason = nullptr) const;
    bool IsPublicUserModel(const std::string& name) const;
    std::vector<std::string> PublicModelNames() const;

private:
    InputConvModelCatalog();

    std::vector<InputConvModelCatalogEntry> entries_;
};

}  // namespace npgr

#endif
