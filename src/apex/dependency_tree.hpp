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

class metricStorage {
public:
    apex_profile prof;
    std::map<double, size_t> distribution;
    metricStorage(double value) {
        prof.accumulated = value;
        prof.maximum = value;
        prof.minimum = value;
        prof.sum_squares = value*value;
        distribution[value] = 1;
    }
    void increment(double value) {
        prof.accumulated += value;
        prof.maximum = std::max<double>(prof.maximum, value);
        prof.minimum = std::min<double>(prof.minimum, value);
        prof.sum_squares += value*value;
        if (distribution.find(value) == distribution.end()) {
            distribution[value] = 1;
        } else {
            distribution[value] += 1;
        }
    }
};

class Node : public std::enable_shared_from_this<Node> {
    private:
        task_identifier* data;
        std::vector<std::shared_ptr<Node>> parents;
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
        std::unordered_map<task_identifier, std::shared_ptr<Node>> children;
        // map for arbitrary metrics
        std::map<std::string, metricStorage> metric_map;
        static std::mutex treeMutex;
        static std::atomic<size_t> nodeCount;
        static std::set<std::string> known_metrics;
    public:
        Node(task_identifier* id, std::shared_ptr<Node> p) :
            data(id), count(1), inclusive(0),
            index(nodeCount.fetch_add(1, std::memory_order_relaxed)) {
            parents.push_back(p);
            prof.calls = 0.0;
            prof.accumulated = 0.0;
            prof.minimum = 0.0;
            prof.maximum = 0.0;
            prof.sum_squares = 0.0;
            memset(prof.papi_metrics, 0, sizeof(double)*8);
        }
        // nothing to destruct? because we used smart pointers for parents/children...
        ~Node() { }
        std::shared_ptr<Node> appendChild(task_identifier* c, std::shared_ptr<Node> existing);
        std::shared_ptr<Node> replaceChild(task_identifier* old_child, task_identifier* new_child);
        task_identifier* getData() { return data; }
        size_t getCount() { return count; }
        inline double& getCalls() { return prof.calls; }
        inline double& getAccumulated() { return prof.accumulated; }
        inline double getThreads() { return (double)thread_ids.size(); }
        inline double& getMinimum() { return prof.minimum; }
        inline double& getMaximum() { return prof.maximum; }
        inline double& getSumSquares() { return prof.sum_squares; }
        void addAccumulated(double value, double incl, bool is_resume, uint64_t thread_id,
            double values[8], int num_papi_counters);
        size_t getIndex() { return index; };
        std::string getName() const { return data->get_name(); };
        void writeNode(std::ofstream& outfile, double total);
        double writeNodeASCII(std::ofstream& outfile, double total, size_t indent);
        double writeNodeCSV(std::stringstream& outfile, double total, int node_id, int num_papi_counters);
        double writeNodeJSON(std::ofstream& outfile, double total, size_t indent);
        void writeTAUCallpath(std::ofstream& outfile, std::string prefix);
        static size_t getNodeCount() {
            return nodeCount;
        }
        void addMetrics(std::map<std::string, double>& metric_map);
        static std::set<std::string>& getKnownMetrics() {
            return known_metrics;
        }
        // required for using this class as a key in a map, vector, etc.
        static bool compareNodeByParentName (const std::shared_ptr<Node> lhs, const std::shared_ptr<Node> rhs) {
            if (lhs->parents[0]->index < rhs->parents[0]->index) {
                return true;
            }
            if (lhs->getName().compare(rhs->getName()) < 0) {
                return true;
            }
            return false;
        }

};

} // dependency_tree

} // apex
