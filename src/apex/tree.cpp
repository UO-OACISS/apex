#include "tree.h"
#include<thread>
#include<mutex>
#include<atomic>
#include<unordered_map>
#include<utility>
#include<iostream>
#include "apex_assert.h"

namespace apex {
namespace treemerge {

std::atomic<size_t> node::count{0};

node * node::buildTree(std::vector<std::vector<std::string>>& rows,
    node * root) {
    if(rows.size() == 0) return root;
    //size_t rank = stol(rows[0][0]);
    size_t index = stol(rows[0][1]);
    size_t pindex = stol(rows[0][2]);
    //size_t depth = stol(rows[0][3]);
    std::string name = rows[0][4];
    std::unordered_map<size_t, node*> node_map;
    // create the root node, if necessary
    if (root == nullptr) {
        root = new node(name);
    }
    node_map.insert(std::pair<size_t,node*>(index,root));
    // for all other rows, create the children
    for(size_t i = 1 ; i < rows.size() ; i++) {
        //rank = stol(rows[i][0]);
        index = stol(rows[i][1]);
        pindex = stol(rows[i][2]);
        //depth = stol(rows[i][3]);
        name = rows[i][4];
        /*
        std::cout << rank << " " << index << " " << pindex << " "
                  << depth << " " << name << std::endl;
                  */
        APEX_ASSERT(node_map.count(pindex) > 0);
        auto parent = node_map.at(pindex);
        node * newChild = parent->addChild(name);
        node_map.insert(std::pair<size_t,node*>(index,newChild));
        // update the row data with the tree data
        rows[i][1] = std::to_string(newChild->_index);
        rows[i][2] = std::to_string(parent->_index);
    }
    std::mutex foo;
    std::scoped_lock{foo};
    //std::cout << "rank " << rank << " done." << std::endl;
    return root;
}

} // namespace treemerge
} // namespace apex