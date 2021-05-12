/*
 * Copyright (c) 2014-2021 Kevin Huck
 * Copyright (c) 2014-2021 University of Oregon
 *
 * Distributed under the Boost Software License, Version 1.0. (See accompanying
 * file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
 */

#include <mutex>
#include <unordered_map>
#include <iostream>
#include <fstream>
#include "task_identifier.hpp"

namespace apex {

namespace dependency {

class Node {
    private:
        task_identifier* data;
        Node* parent;
        size_t count;
        double calls;
        double accumulated;
        double min;
        double max;
        double sumsqr;
        size_t index;
        std::unordered_map<task_identifier, Node*> children;
        static std::mutex treeMutex;
        static size_t nodeCount;
    public:
        Node(task_identifier* id, Node* p) :
            data(id), parent(p), count(1), calls(0), accumulated(0), index(++nodeCount) {
        }
        ~Node() {
            treeMutex.lock();
            for (auto c : children) {
                delete c.second;
            }
            treeMutex.unlock();
        }
        Node* appendChild(task_identifier* c);
        Node* replaceChild(task_identifier* old_child, task_identifier* new_child);
        task_identifier* getData() { return data; }
        Node* getParent() { return parent; }
        size_t getCount() { return count; }
        size_t getCalls() { return calls; }
        double getAccumulated() { return accumulated; }
        void addAccumulated(double value, bool is_resume);
        size_t getIndex() { return index; };
        void writeNode(std::ofstream& outfile, double total);
        void writeNodeASCII(double total, size_t indent);
};

} // dependency_tree

} // apex