#include <string>
#include <vector>
#include <istream>

namespace apex {
namespace treemerge {

std::vector<std::string> csv_read_row(std::istream &in, char delimiter);
std::vector<std::string> csv_read_row(std::string &in, char delimiter);
void csv_write(std::vector<std::string> header,
    std::vector<std::vector<std::vector<std::string>>*> &ranks,
    std::string filename);

} // namespace treemerge
} // namespace apex
