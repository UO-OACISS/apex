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

namespace apex {

namespace dependency {

// declare an instance of the statics
std::mutex Node::treeMutex;
std::atomic<size_t> Node::nodeCount{0};

Node* Node::appendChild(task_identifier* c) {
    treeMutex.lock();
    auto iter = children.find(*c);
    if (iter == children.end()) {
        auto n = new Node(c,this);
        //std::cout << "Inserting " << c->get_name() << std::endl;
        children.insert(std::make_pair(*c,n));
        treeMutex.unlock();
        return n;
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
    static size_t depth = 0;
    // Write out the relationships
    if (parent != nullptr) {
        outfile << "  \"" << parent->getIndex() << "\" -> \"" << getIndex() << "\";";
        outfile << std::endl;
    }

    double acc = (data == task_identifier::get_main_task_id()) ?
        total : accumulated;
    node_color * c = get_node_color_visible(acc, 0.0, total);
    double ncalls = (calls == 0) ? 1 : calls;

    // write out the nodes
    outfile << "  \"" << getIndex() <<
            "\" [shape=box; style=filled; fillcolor=\"#" <<
            std::setfill('0') << std::setw(2) << std::hex << c->convert(c->red) <<
            std::setfill('0') << std::setw(2) << std::hex << c->convert(c->green) <<
            std::setfill('0') << std::setw(2) << std::hex << c->convert(c->blue) <<
            "\"; depth=" << std::dec << depth <<
            "; time=" << std::fixed << acc << "; label=\"" << data->get_tree_name() <<
            "\\l calls: " << ncalls << "\\l time: " <<
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
    for (size_t i = 0 ; i < indent ; i++) {
        outfile << "| ";
    }
    outfile << "|-> ";
    indent++;
    // write out the inclusive and percent of total
    double acc = (data == task_identifier::get_main_task_id() || accumulated == 0.0) ?
        total : accumulated;
    double percentage = (accumulated / total) * 100.0;
    outfile << std::fixed << std::setprecision(5) << acc << " - "
            << std::fixed << std::setprecision(4) << percentage << "% [";
    // write the number of calls
    double ncalls = (calls == 0) ? 1 : calls;
    outfile << std::fixed << std::setprecision(0) << ncalls << "]";
    // write other stats - min, max, stddev
    double mean = acc / ncalls;
    double variance = ((sumsqr / ncalls) - (mean * mean));
    double stddev = sqrt(variance);
    outfile << " {min=" << std::fixed << std::setprecision(4) << min << ", max=" << max
            << ", mean=" << mean << ", var=" << variance
            << ", std dev=" << stddev << "} ";
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

double Node::writeNodeJSON(std::ofstream& outfile, double total, size_t indent) {
    APEX_ASSERT(total > 0.0);
    // indent as necessary
    for (size_t i = 0 ; i < indent ; i++) { outfile << " "; }
    indent++;
    // write out the opening brace
    outfile << std::fixed << std::setprecision(6) << "{ ";
    // write out the name
    outfile << "\"name\": \"" << data->get_tree_name() << "\", ";
    // write out the inclusive
    double acc = (data == task_identifier::get_main_task_id() || accumulated == 0.0) ?
        total : std::min(total, accumulated);
    // Don't write out synchronization events! They confuse the graph.
    if (data->get_tree_name().find("Synchronize") != std::string::npos) acc = 0.0;
    outfile << "\"size\": " << acc;

    // if no children, we are done
    if (children.size() == 0) {
        outfile << " }";
        return acc;
    }

    // write the children label
    outfile << ", \"children\": [\n";

    // sort the children by accumulated time
    std::vector<std::pair<task_identifier, Node*> > sorted;
    for (auto& it : children) {
        sorted.push_back(it);
    }
    sort(sorted.begin(), sorted.end(), cmp);

    // do all the children
    double children_total = 0.0;
    bool first = true;
    for (auto c : sorted) {
        if (!first) { outfile << ",\n"; }
        first = false;
        double tmp = c.second->writeNodeJSON(outfile, total, indent);
        // if we didn't write the child, don't write a comma.
        children_total = children_total + tmp;
    }
    double remainder = acc - children_total;
    /*
    if (remainder > 0.0) {
        if (!first) { outfile << ",\n"; }
        for (size_t i = 0 ; i < indent ; i++) { outfile << " "; }
        outfile << "{ \"name\": \"" << data->get_tree_name() << "\", \"size\": " << remainder << " }";
    }
    */
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
    double acc = accumulated * 1000000; // stored in seconds, we need to convert to microseconds

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
            double tmp = c.second->getAccumulated() * 1000000;
            child_time = child_time + tmp;
            tmp = c.second->getCalls();
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
        double ncalls = (calls == 0) ? 1 : calls;
        outfile << std::fixed << std::setprecision(0) << ncalls << " ";
        // write out subroutines
        outfile << child_calls << " ";
        // write out exclusive
        outfile << std::fixed << std::setprecision(3) << remainder << " ";
        // write out inclusive
        outfile << std::fixed << std::setprecision(3) << acc << " ";
        // write out profilecalls and group
        outfile << "0 GROUP=\"" << data->get_group() << " | TAU_CALLPATH\" ";
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

void Node::addAccumulated(double value, bool is_resume) {
    static std::mutex m;
    m.lock();
    if (!is_resume) { calls+=1; }
    accumulated = accumulated + value;
    if (min == 0.0 || value < min) { min = value; }
    if (value > max) { max = value; }
    sumsqr = sumsqr + (value*value);
    m.unlock();
}

} // dependency_tree

} // apex
