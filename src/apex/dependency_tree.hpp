/*
 * Copyright (c) 2014-2021 Kevin Huck
 * Copyright (c) 2014-2021 University of Oregon
 *
 * Distributed under the Boost Software License, Version 1.0. (See accompanying
 * file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
 */

#pragma once
#include <mutex>
#include <unordered_map>
#include <iostream>
#include <fstream>
#include <atomic>
#include <set>
#include <map>
#include "apex_types.h"
#include "task_identifier.hpp"

namespace apex {

namespace dependency {

class Node {
    private:
        task_identifier* data;
        Node* parent;
        size_t count;
        apex_profile prof;
        //double calls;
        //double accumulated;
        //double min;
        //double max;
        //double sumsqr;
        double inclusive;
        size_t index;
        std::set<uint64_t> thread_ids;
        std::unordered_map<task_identifier, Node*> children;
        // map for arbitrary metrics
        std::map<std::string, double> metric_map;
        static std::mutex treeMutex;
        static std::atomic<size_t> nodeCount;
        static std::set<std::string> known_metrics;
    public:
        Node(task_identifier* id, Node* p) :
            data(id), parent(p), count(1), inclusive(0),
            index(nodeCount.fetch_add(1, std::memory_order_relaxed)) {
            prof.calls = 0.0;
            prof.accumulated = 0.0;
            prof.minimum = 0.0;
            prof.maximum = 0.0;
            prof.sum_squares = 0.0;
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
        inline double& getCalls() { return prof.calls; }
        inline double& getAccumulated() { return prof.accumulated; }
        inline double& getMinimum() { return prof.minimum; }
        inline double& getMaximum() { return prof.maximum; }
        inline double& getSumSquares() { return prof.sum_squares; }
        void addAccumulated(double value, double incl, bool is_resume, uint64_t thread_id);
        size_t getIndex() { return index; };
        std::string getName() { return data->get_name(); };
        void writeNode(std::ofstream& outfile, double total);
        double writeNodeASCII(std::ofstream& outfile, double total, size_t indent);
        double writeNodeCSV(std::stringstream& outfile, double total, int node_id);
        double writeNodeJSON(std::ofstream& outfile, double total, size_t indent);
        void writeTAUCallpath(std::ofstream& outfile, std::string prefix);
        static size_t getNodeCount() {
            return nodeCount;
        }
        void addMetrics(std::map<std::string, double>& metric_map);
        static std::set<std::string>& getKnownMetrics() {
            return known_metrics;
        }
};

} // dependency_tree

} // apex
