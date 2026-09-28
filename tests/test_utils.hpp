#pragma once

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

inline void expectNear(const std::vector<float> &actual,
                       const std::vector<float> &expected,
                       float atol,
                       const std::string &name)
{
    if (actual.size() != expected.size())
    {
        std::cerr << name << ": size mismatch\n";
        std::exit(EXIT_FAILURE);
    }

    float max_error = 0.0f;
    for (std::size_t i = 0; i < actual.size(); ++i)
    {
        max_error = std::fmax(max_error, std::fabs(actual[i] - expected[i]));
    }
    
    std::cout << name << " max_abs_error=" << max_error << '\n';
    if (max_error > atol)
    {
        std::cerr << name << ": tolerance exceeded\n";
        std::exit(EXIT_FAILURE);
    }
}
