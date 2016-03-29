//  Copyright (c) 2014 University of Oregon
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include "utils.hpp"
#include <sstream>
#include <cstring>
#if defined(__GNUC__)
#include <cxxabi.h>
#endif

namespace apex {

/* Idea borrowed from:
 * http://stackoverflow.com/questions/931827/stdstring-comparison-check-whether-string-begins-with-another-string */
bool starts_with(const std::string& input, const std::string& match)
{
        return input.size() >= match.size()
                    && std::equal(match.begin(), match.end(), input.begin());
}

/* Idea borrowed from:
 * http://stackoverflow.com/questions/236129/split-a-string-in-c
 */
std::vector<std::string> &split(const std::string &s, char delim, std::vector<std::string> &elems) {
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, delim)) {
        // ignore duplicate delimiters.
        if (item.size() > 0) {
            elems.push_back(item);
        }
    }
    return elems;
}

std::string demangle(const std::string& timer_name) {
  std::string demangled(timer_name);
#if defined(__GNUC__)
  int     status;
  char *realname = abi::__cxa_demangle(timer_name.c_str(), 0, 0, &status);
  if (status == 0) {
    char* index = strstr(realname, "<");
    if (index != NULL) {
      *index = 0; // terminate before templates for brevity
    }
    demangled = std::string(realname);
    free(realname);
  }
#endif
  return demangled;
}

};

