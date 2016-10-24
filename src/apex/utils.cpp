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
// for setting thread affinity
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
// for removing directories
#include <dirent.h>
#include <sys/stat.h>

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
  if (status == 0 && realname != NULL) {
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

#define handle_error_en(en, msg) \
               do { errno = en; perror(msg); exit(EXIT_FAILURE); } while (0)

void set_thread_affinity(void) {
#if !defined(__APPLE__)
    int s, j;
    cpu_set_t cpuset;
    pthread_t thread;

    thread = pthread_self();

    /* Set affinity mask to include CPUs 0 to 7 */

    CPU_ZERO(&cpuset);
    j = my_hardware_concurrency() - 1;
    CPU_SET(j, &cpuset);

    s = pthread_setaffinity_np(thread, sizeof(cpu_set_t), &cpuset);
    if (s != 0) handle_error_en(s, "pthread_setaffinity_np");

    /* Check the actual affinity mask assigned to the thread */

    /*
    s = pthread_getaffinity_np(thread, sizeof(cpu_set_t), &cpuset);
    if (s != 0) handle_error_en(s, "pthread_getaffinity_np");

    printf("Set returned by pthread_getaffinity_np() contained:\n");
    for (j = 0; j < CPU_SETSIZE; j++) {
        if (CPU_ISSET(j, &cpuset)) {
            printf("    CPU %d\n", j);
        }
    }
	*/
#endif
    return;
}

void remove_path(const char *pathname) {
    struct dirent *entry = NULL;
    DIR *dir = NULL;
	struct stat sb;
	if (stat(pathname, &sb) == 0 && S_ISDIR(sb.st_mode)) {
    	dir = opendir(pathname);
    	while((entry = readdir(dir)) != NULL) {
        	DIR *sub_dir = NULL;
        	FILE *file = NULL;
        	char abs_path[100] = {0};
        	if(*(entry->d_name) != '.') {
            	sprintf(abs_path, "%s/%s", pathname, entry->d_name);
            	sub_dir = opendir(abs_path);
            	if(sub_dir != NULL) {
                	closedir(sub_dir);
                	remove_path(abs_path);
            	} else {
                	file = fopen(abs_path, "r");
                	if(file != NULL) {
                    	fclose(file);
                    	//printf("Removing: %s\n", abs_path);
                    	remove(abs_path);
                	}
            	}
        	}
    	}
        //printf("Removing: %s\n", pathname);
    	remove(pathname);
	}
}

};

