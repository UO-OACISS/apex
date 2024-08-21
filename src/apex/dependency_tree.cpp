/*
 * Copyright (c) 2014-2021 Kevin Huck
 * Copyright (c) 2014-2021 University of Oregon
 *
 * Distributed under the Boost Software License, Version 1.0. (See accompanying
 * file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
 */

#include "dependency_tree.hpp"
#include "utils.hpp"
#include <iomanip>
#include <sstream>
#include <math.h>
#include "apex_assert.h"
#include "apex.hpp"

namespace apex {

namespace dependency {

// declare an instance of the statics
std::mutex Node::treeMutex;
std::atomic<size_t> Node::nodeCount{0};
std::set<std::string> Node::known_metrics;

Node* Node::appendChild(task_identifier* c, Node* existing) {
    treeMutex.lock();
    auto iter = children.find(*c);
    if (iter == children.end()) {
        if (existing != nullptr) {
            existing->parents.push_back(this);
            children.insert(std::make_pair(*c,existing));
            treeMutex.unlock();
            return existing;
        } else {
            auto n = new Node(c,this);
            //std::cout << "Inserting " << c->get_name() << std::endl;
            children.insert(std::make_pair(*c,n));
            treeMutex.unlock();
            return n;
        }
    }
    iter->second->count++;
    treeMutex.unlock();
    return iter->second;
}

Node* Node::replaceChild(task_identifier* old_child, task_identifier* new_child) {
    treeMutex.lock();
    auto olditer = children.find(*old_child);
    // not found? shouldn't happen...
    if (olditer == children.end()) {
        auto n = new Node(new_child,this);
        //std::cout << "Inserting " << new_child->get_name() << std::endl;
        children.insert(std::make_pair(*new_child,n));
        treeMutex.unlock();
        return n;
    }
    olditer->second->count--;
    // if no more references to this node, delete it.
    if (olditer->second->count == 0) {
        children.erase(*old_child);
    }
    auto newiter = children.find(*new_child);
    // not found? shouldn't happen...
    if (newiter == children.end()) {
        auto n = new Node(new_child,this);
        //std::cout << "Inserting " << new_child->get_name() << std::endl;
        children.insert(std::make_pair(*new_child,n));
        treeMutex.unlock();
        return n;
    }
    treeMutex.unlock();
    return newiter->second;
}

void Node::writeNode(std::ofstream& outfile, double total) {
    static std::set<Node*> processed;
    if (processed.count(this)) return;
    processed.insert(this);
    static size_t depth = 0;
    // Write out the relationships
    for(auto& parent : parents) {
        if (parent != nullptr) {
            outfile << "  \"" << parent->getIndex() << "\" -> \"" << getIndex() << "\";";
            outfile << std::endl;
        }
    }

    double acc = (data == task_identifier::get_main_task_id() || getAccumulated() == 0.0) ?
        total : getAccumulated();
    node_color * c = get_node_color_visible(acc, 0.0, total, data->get_tree_name());
    double ncalls = (getCalls() == 0) ? 1 : getCalls();

    std::string decoration;
    std::string font;
    // if the node is dark, make the font white for readability
    if (acc > 0.5 * total) {
        font = "; fontcolor=white";
    } else {
        font = "; fontcolor=black";
    }
    // if the node is GPU related, make the box dark blue
    if (isGPUTimer(data->get_tree_name())) {
        decoration = "shape=box; color=blue; style=filled" + font + "; fillcolor=\"#";
    } else {
        decoration = "shape=box; color=firebrick; style=filled" + font + "; fillcolor=\"#";
    }
    // write out the nodes
    outfile << "  \"" << getIndex() <<
            "\" [" << decoration <<
            std::setfill('0') << std::setw(2) << std::hex << c->convert(c->red) <<
            std::setfill('0') << std::setw(2) << std::hex << c->convert(c->green) <<
            std::setfill('0') << std::setw(2) << std::hex << c->convert(c->blue) <<
            "\"; depth=" << std::dec << depth <<
            "; time=" << std::fixed << acc <<
            "; inclusive=" << std::fixed << inclusive <<
            "; label=\"" << data->get_tree_name() <<
            "\\lcalls: " << ncalls << "\\ltime: " <<
            std::defaultfloat << acc << "\\l\" ];" << std::endl;

    // do all the children
    depth++;
    for (auto c : children) {
        c.second->writeNode(outfile, total);
    }
    depth--;
}

bool cmp(std::pair<task_identifier, Node*>& a, std::pair<task_identifier, Node*>& b) {
    return a.second->getAccumulated() > b.second->getAccumulated();
}

double Node::writeNodeASCII(std::ofstream& outfile, double total, size_t indent) {
    APEX_ASSERT(total > 0.0);
    constexpr int precision{3};
    for (size_t i = 0 ; i < indent ; i++) {
        outfile << "| ";
    }
    outfile << "|-> ";
    indent++;
    // write out the inclusive and percent of total
    double acc = (data == task_identifier::get_main_task_id() || getAccumulated() == 0.0) ?
        total : getAccumulated();
    double percentage = (getAccumulated() / total) * 100.0;
    outfile << std::fixed << std::setprecision(precision) << acc << " - "
            << std::fixed << std::setprecision(precision) << percentage << "% [";
    // write the number of calls
    double ncalls = (getCalls() == 0) ? 1 : getCalls();
    outfile << std::fixed << std::setprecision(0) << ncalls << "]";
    // write other stats - min, max, stddev
    double mean = acc / ncalls;
    // avoid -0.0 which will cause a -nan for stddev
    double variance = std::max(0.0,((getSumSquares() / ncalls) - (mean * mean)));
    double stddev = sqrt(variance);
    outfile << " {min=" << std::fixed << std::setprecision(precision) << getMinimum() << ", max=" << getMaximum()
            << ", mean=" << mean << ", var=" << variance
            << ", std dev=" << stddev
            << ", inclusive=" << inclusive
            << ", threads=" << thread_ids.size() << "} ";
    // Write out the name
    outfile << data->get_tree_name() << " ";
    // end the line
    outfile << std::endl;

    // sort the children by accumulated time
    std::vector<std::pair<task_identifier, Node*> > sorted;
    for (auto& it : children) {
        sorted.push_back(it);
    }
    sort(sorted.begin(), sorted.end(), cmp);

    // do all the children
    double remainder = acc;
    for (auto c : sorted) {
        double tmp = c.second->writeNodeASCII(outfile, total, indent);
        remainder = remainder - tmp;
    }
    if (children.size() > 0 && remainder > 0.0) {
        for (size_t i = 0 ; i < indent ; i++) {
            outfile << "| ";
        }
        percentage = (remainder / total) * 100.0;
        outfile << "Remainder: " << remainder << " - " << percentage << "%" << std::endl;
    }
    return acc;
}

/* print something like (for Hatchet):
{
    "frame": {"name": "foo"},
    "metrics": {"time (inc)": 135.0, "time": 0.0},
    "children": [ { ...} ]
} */
double Node::writeNodeJSON(std::ofstream& outfile, double total, size_t indent) {
    APEX_ASSERT(total > 0.0);
    // indent as necessary
    for (size_t i = 0 ; i < indent ; i++) { outfile << " "; }
    indent++;
    // write out the opening brace
    outfile << std::fixed << std::setprecision(6) << "{ ";
    // write out the name
    outfile << "\"frame\": {\"name\": \"" << data->get_name()
            << "\", \"type\": \"function\", \"rank\": "
            << apex::instance()->get_node_id() << "}, ";
    // write out the inclusive
    double acc = (data == task_identifier::get_main_task_id() || getAccumulated() == 0.0) ?
        total : std::min(total, getAccumulated());

    // solve for the exclusive
    double excl = acc;
    for (auto c : children) {
        excl = excl - c.second->getAccumulated();
    }
    if (excl < 0.0) {
        excl = 0.0;
    }

    // Don't write out synchronization events! They confuse the graph.
    if (data->get_tree_name().find("Synchronize") != std::string::npos) acc = 0.0;
    double ncalls = (getCalls() == 0) ? 1 : getCalls();
    outfile << "\"metrics\": {\"time\": " << excl
            << ", \"total time (inc)\": " << acc
            << ", \"time (inc cpu)\": " << (acc / (double)(thread_ids.size()))
            << ", \"time (inc wall)\": " << inclusive
            << ", \"num threads\": " << thread_ids.size()
            << ", \"min (inc)\": " << getMinimum()
            << ", \"max (inc)\": " << getMaximum()
            << ", \"sumsqr (inc)\": " << getSumSquares()
            << ", \"calls\": " << ncalls << "}";

    // if no children, we are done
    if (children.size() == 0) {
        outfile << " }";
        return acc;
    }

    // write the children label
    outfile << ", \"children\": [\n";

    // do all the children
    double children_total = 0.0;
    bool first = true;
    for (auto c : children) {
        if (!first) { outfile << ",\n"; }
        first = false;
        double tmp = c.second->writeNodeJSON(outfile, total, indent);
        children_total = children_total + tmp;
    }
    // close the list
    outfile << "\n";
    for (size_t i = 0 ; i < indent-1 ; i++) { outfile << " "; }
    outfile << "]\n";
    // end the object
    for (size_t i = 0 ; i < indent-1 ; i++) { outfile << " "; }
    outfile << "}";
    return std::max(acc, children_total);
}

void Node::writeTAUCallpath(std::ofstream& outfile, std::string prefix) {
    static size_t depth = 0;

    // if we have no children, and there's no prefix, do nothing.
    if (prefix.size() == 0 && children.size() == 0) { return ; }

    // get the inclusive amount for this timer
    double acc = (getAccumulated() * 1000000) / getThreads(); // stored in seconds, we need to convert to microseconds

    // update the prefix
    if (data->get_name().compare(APEX_MAIN_STR) == 0) {
        prefix.append(".TAU application");
    } else {
        prefix.append(data->get_name(true));
    }

    // only do this if we are in the tree - the flat profile is written already.
    if (depth > 0) {
        // compute our exclusive time
        double child_time = 0;
        double child_calls = 0;
        for (auto c : children) {
            double tmp = (c.second->getAccumulated() * 1000000) / c.second->getThreads();
            child_time = child_time + tmp;
            tmp = c.second->getCalls() / c.second->getThreads();
            child_calls = child_calls + tmp;
        }
        double remainder = 0;
        if (acc < child_time) {
            if (acc == 0.0) {
                acc = child_time;
                remainder = 0.0;
            } else {
                remainder = acc;
            }
        } else {
            remainder = acc - child_time;
        }

        // otherwise, write out this node
        outfile << "\"" << prefix << "\" ";
        // write the number of calls
        double ncalls = (getCalls() == 0) ? 1 : (getCalls() / getThreads());
        outfile << std::fixed << std::setprecision(0) << ncalls << " ";
        // write out subroutines
        outfile << child_calls << " ";
        // write out exclusive
        outfile << std::fixed << std::setprecision(3) << remainder << " ";
        // write out inclusive
        outfile << std::fixed << std::setprecision(3) << acc << " ";
        //outfile << std::fixed << std::setprecision(3) << inclusive << " ";
        // write out profilecalls and group
        outfile << "0 GROUP=\"TAU_CALLPATH|" << data->get_group() << "\" ";
        // end the line
        outfile << std::endl;
    }

    // create a prefix for children
    std::string child_prefix{prefix};
    child_prefix.append(" => ");

    // recursively do a depth-first writing of all the children and subchildren...
    depth++;
    for (auto c : children) {
        c.second->writeTAUCallpath(outfile, child_prefix);
    }
    depth--;

    return;
}

void Node::addAccumulated(double value, double incl, bool is_resume, uint64_t thread_id,
    double values[8], int num_papi_counters) {
    static std::mutex m;
    m.lock();
    if (!is_resume) {
        getCalls()+=1;
        inclusive = inclusive + incl;
    }
    getAccumulated() = getAccumulated() + value;
    if (getMinimum() == 0.0 || value < getMinimum()) { getMinimum() = value; }
    if (value > getMaximum()) { getMaximum() = value; }
    getSumSquares() = getSumSquares() + (value*value);
    thread_ids.insert(thread_id);
    /* Add the papi measurements */
    for (int i = 0 ; i < num_papi_counters ; i++) {
        prof.papi_metrics[i] += values[i];
    }

    m.unlock();
}

double Node::writeNodeCSV(std::stringstream& outfile, double total, int node_id, int num_papi_counters) {
    static size_t depth = 0;
    static std::set<Node*> processed;
    if (processed.count(this)) return getAccumulated();
    processed.insert(this);
    APEX_ASSERT(total > 0.0);
    // write out the node id and graph node index and the name
    outfile << node_id << "," << index << ",\"[";
    std::string delim("");
    for (auto& parent : parents) {
        if (parent != nullptr) {
            outfile << delim << parent->index;
            delim = ",";
        }
    }
    outfile << "]\"," << depth << ",\"";
    outfile << data->get_tree_name() << "\",";
    // write out the accumulated
    double acc = (data == task_identifier::get_main_task_id() || getAccumulated() == 0.0) ?
        total : getAccumulated();
    // write the number of calls
    double ncalls = (getCalls() == 0) ? 1 : getCalls();
    outfile << std::fixed << std::setprecision(0) << ncalls << ",";
    outfile << thread_ids.size() << ",";
    // write other stats - min, max, stddev
    double mean = acc / ncalls;
    outfile << std::setprecision(9);
    outfile << acc << ",";
    outfile << inclusive << ",";
    outfile << getMinimum() << ",";
    outfile << mean << ",";
    outfile << getMaximum() << ",";
    // avoid -0.0 which will cause a -nan for stddev
    double variance = std::max(0.0,((getSumSquares() / ncalls) - (mean * mean)));
    double stddev = sqrt(variance);
    outfile << stddev;
    // write the papi metrics
    for (int m = 0 ; m < num_papi_counters ; m++) {
        outfile << "," << prof.papi_metrics[m];
    }

    // write any available metrics
    for (auto& x : known_metrics) {
        if (metric_map.find(x) == metric_map.end()) {
            outfile << ",,,,,,,";
        } else {
            const auto& value = metric_map.find(x);
            const auto& p = value->second.prof;
            outfile << "," << p.accumulated;
            outfile << "," << p.minimum;
            double mean = p.accumulated/ncalls;
            outfile << "," << mean;
            outfile << "," << p.maximum;
            // compute the standard deviation
            double t1 = p.sum_squares / ncalls;
            double t2 = mean * mean;
            double t3 = t1 - t2;
            variance = std::max(0.0,(t3));
            stddev = sqrt(variance);
            outfile << "," << stddev;
            // find the median
            auto& d = value->second.distribution;
            // how many do we have?
            size_t total = 0;
            double mode = 0;
            double median = 0;
            size_t half = (size_t)(ncalls/2.0);
            size_t most = 0;
            for (auto& node : d) {
                total += node.second;
                if (total >= half) { median = node.first; break; }
            }
            for (auto& node : d) {
                if (node.second > most) { mode = node.first; }
            }
            outfile << "," << median;
            outfile << "," << mode;
        }
    }
    // end the line
    outfile << std::endl;

    // sort the children by name to make tree merging easier (I hope)
    std::vector<Node*> sorted;
    for (auto& it : children) {
        sorted.push_back(it.second);
    }
    sort(sorted.begin(), sorted.end(), Node::compareNodeByParentName);

    // do all the children
    double remainder = acc;
    depth++;
    for (auto c : sorted) {
        //double tmp = c->writeNodeCSV(outfile, total, node_id, num_papi_counters);
        /* OMPT target handling with 0 length events is giving me headaches... */
        double tmp = c->writeNodeCSV(outfile, acc, node_id, num_papi_counters);
        remainder = remainder - tmp;
    }
    depth--;
    return acc;
}

void Node::addMetrics(std::map<std::string, double>& _metric_map) {
    static std::mutex m;
    for (auto& x: _metric_map) {
        std::string name{x.first};
        double value{x.second};
        if (known_metrics.find(name) == known_metrics.end()) {
            m.lock();
            known_metrics.insert(name);
            m.unlock();
        }
        m.lock();
        if (metric_map.find(name) == metric_map.end()) {
            metricStorage newval(value);
            metric_map.emplace(name, std::move(newval));
        } else {
            auto element = metric_map.find(name);
            element->second.increment(value);
        }
        m.unlock();
    }
}

} // dependency_tree

} // apex
