#pragma once

#include "core/parameter.hpp"

#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace cugpt::core
{
    // Composite modules expose leaf Parameter objects through one flat list for the
    // optimizer and keep a separate named module/parameter tree for serialization
    // The full parameter name is derived from the module hierarchy, just like in
    // Python implementation; Parameter itself does not own the full name
    class Module
    {
    public:
        using NamedParameter = std::pair<std::string, Parameter *>;
        using ConstNamedParameter = std::pair<std::string, const Parameter *>;

        virtual ~Module() = default;

        virtual void zeroGrad(const CudaContext &ctx)
        {
            for (Parameter *p : params_)
            {
                p->zeroGrad(ctx);
            }
        }

        // Flat parameter list used by the optimizer
        const std::vector<Parameter *> &parameters() const noexcept { return params_; }

        // Returns Python-compatible fully-qualified parameter names
        std::vector<NamedParameter> namedParameters(const std::string &prefix = "")
        {
            std::vector<NamedParameter> result;
            collectNamedParameters(prefix, result);
            return result;
        }

        std::vector<ConstNamedParameter> namedParameters(const std::string &prefix = "") const
        {
            std::vector<ConstNamedParameter> result;
            collectNamedParameters(prefix, result);
            return result;
        }

    protected:
        void registerParameter(const std::string &name, Parameter &parameter)
        {
            validateMemberName(name);
            ensureUniqueMemberName(name);

            parameter_members_.emplace_back(name, &parameter);
            params_.push_back(&parameter);
        }

        void registerModule(const std::string &name, Module &module)
        {
            validateMemberName(name);
            ensureUniqueMemberName(name);

            module_members_.emplace_back(name, &module);
            params_.insert(params_.end(), module.parameters().begin(), module.parameters().end());
        }

        std::vector<Parameter *> params_;

    private:
        std::vector<std::pair<std::string, Parameter *>> parameter_members_;
        std::vector<std::pair<std::string, Module *>> module_members_;

        static void validateMemberName(const std::string &name)
        {
            if (name.empty())
            {
                throw std::invalid_argument("Module member name cannot be empty");
            }
        }

        void ensureUniqueMemberName(const std::string &name) const
        {
            for (const auto &[existing_name, unused] : parameter_members_)
            {
                if (existing_name == name)
                {
                    throw std::invalid_argument("Duplicate module member name: " + name);
                }
            }

            for (const auto &[existing_name, unused] : module_members_)
            {
                if (existing_name == name)
                {
                    throw std::invalid_argument("Duplicate module member name: " + name);
                }
            }
        }

        template <typename NamedParameterT>
        void collectNamedParameters(const std::string &prefix, std::vector<NamedParameterT> &result) const
        {
            for (const auto &[name, parameter] : parameter_members_)
            {
                const std::string full_name = prefix.empty() ? name : prefix + "." + name;
                result.emplace_back(full_name, parameter);
            }

            for (const auto &[name, module] : module_members_)
            {
                const std::string full_name = prefix.empty() ? name : prefix + "." + name;
                module->collectNamedParameters(full_name, result);
            }
        }
    };
} // namespace cugpt::core
