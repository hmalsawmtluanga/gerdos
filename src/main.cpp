#include <iostream>

#include <gerdos/version.hpp>

int main()
{
    std::cout << "GERDOS "
              << GERDOS_VERSION_MAJOR << '.'
              << GERDOS_VERSION_MINOR << '.'
              << GERDOS_VERSION_PATCH << '\n';

    std::cout
        << "Global Execution Runtime for Dynamic Open Systems\n";

    return 0;
}
