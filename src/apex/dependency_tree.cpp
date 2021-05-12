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

namespace apex {

namespace dependency {

// declare an instance of the statics
std::mutex Node::treeMutex;
size_t Node::nodeCount{0};

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
    // Write out the relationships
    if (parent != nullptr) {
        outfile << "  \"" << parent->getIndex() << "\" -> \"" << getIndex() << "\";";
        outfile << std::endl;
    }

    const std::string apex_main_str("APEX MAIN");
    double acc = (data == task_identifier::get_task_id(apex_main_str)) ?
        total : accumulated;
    node_color * c = get_node_color_visible(acc, 0.0, total);
    double ncalls = (calls == 0) ? 1 : calls;

    // write out the nodes
    outfile << "  \"" << getIndex() <<
    "\" [shape=box; style=filled; fillcolor=\"#" <<
            std::setfill('0') << std::setw(2) << std::hex << c->convert(c->red) <<
            std::setfill('0') << std::setw(2) << std::hex << c->convert(c->green) <<
            std::setfill('0') << std::setw(2) << std::hex << c->convert(c->blue) <<
            "\"; label=\"" << data->get_name() <<
    ":\\ltotal calls: " << ncalls << "\\ltotal time: " << acc << "\" ];" << std::endl;

    // do all the children
    for (auto c : children) {
        c.second->writeNode(outfile, total);
    }
}

bool cmp(std::pair<task_identifier, Node*>& a, std::pair<task_identifier, Node*>& b) {
    return a.second->getAccumulated() > b.second->getAccumulated();
}

double Node::writeNodeASCII(std::ofstream& outfile, double total, size_t indent) {
    for (size_t i = 0 ; i < indent ; i++) {
        outfile << "|   ";
    }
    indent++;
    // Write out the name
    outfile << data->get_short_name() << ": ";
    // write out the inclusive and percent of total
    const std::string apex_main_str("APEX MAIN");
    double acc = (data == task_identifier::get_task_id(apex_main_str)) ?
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
            << ", std dev=" << stddev << "}";
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
        double tmp = c.second->writeNodeASCII(outfile, acc, indent);
        remainder = remainder - tmp;
    }
    if (children.size() > 0 && remainder > 0.0) {
        for (size_t i = 0 ; i < indent ; i++) {
            outfile << "|   ";
        }
        percentage = (remainder / total) * 100.0;
        outfile << "Remainder: " << remainder << " - " << percentage << "%" << std::endl;
    }
    return acc;
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