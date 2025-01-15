#include<thread>
#include<mutex>
#include<atomic>
#include<unordered_map>
#include<iostream>
#include<vector>
#include<mutex>

namespace apex {
namespace treemerge {

class node {
    private:
        std::mutex mtx;
        static std::atomic<size_t> count;
        size_t _index;
        std::string _name;
        std::unordered_map<std::string, node*> children;
    public:
        // delete the default constructor
        node(void) = delete;
        // delete the copy constructors
        node(const node& temp) = delete;
        node& operator=(const node& temp) = delete;
        node(const std::string name) :
            _index(count++), _name(name) { }
        ~node(void) {
            for(auto c : children) {
                delete(c.second);
            }
        }
        bool hasChild(const std::string& name) {
            if(children.count(name) > 0) {
                return true;
            }
            return false;
        }
        node* addChild(const std::string& cname) {
            if (!hasChild(cname)) {
                // child not found, get lock
                std::scoped_lock l{mtx};
                // child may have appeared?
                if (!hasChild(cname)) {
                    //std::cout << "Adding " << _name << " -> " << cname << std::endl;
                    auto child = new node(cname);
                    children.insert(std::pair<std::string, node*>(cname, child));
                }
            }
            return children.at(cname);
        }
        static node * buildTree(std::vector<std::vector<std::string>>& rows, node * root);
        static size_t getSize(void) { return count; }
        static void reset(void) { count = 0; }
};

} // namespace treemerge
} // namespace apex