#define SAFETENSORS_CPP_IMPLEMENTATION 1
#include "safetensors.hh"

int main(int argc, char **argv)
{
    safetensors::safetensors_t st;

    std::string warn, err;

    safetensors::load_from_file("model.safetensors", &st, &warn, &err);

    for (size_t i = 0; i < st.tensors.size(); i++)
    {
        std::string key = st.tensors.keys()[i];
        safetensors::tensor_t tensor;
        st.tensors.at(i, &tensor);

        std::cout << key << ": "
                  << safetensors::get_dtype_str(tensor.dtype) << " ";
        std::cout << "[";
        for (size_t i = 0; i < tensor.shape.size(); i++)
        {
            if (i > 0)
            {
                std::cout << ", ";
            }
            std::cout << std::to_string(tensor.shape[i]);
        }
        std::cout << "]\n";

        std::cout << "  data_offsets["
                  << std::to_string(tensor.data_offsets[0]) << ", "
                  << std::to_string(tensor.data_offsets[1]) << "]\n";
    }

    return 0;
}