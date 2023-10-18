#include "tree.h"
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <sstream>
#include <istream>

namespace apex {
namespace treemerge {

std::vector<std::string> csv_read_row(std::istream &in, char delimiter)
{
    std::stringstream ss;
    bool inquotes = false;
    std::vector<std::string> row;//relying on RVO
    while(in.good())
    {
        char c = in.get();
        if (!inquotes && c=='"') //beginquotechar
        {
            inquotes=true;
            ss << c;
        }
        else if (inquotes && c=='"') //quotechar
        {
            if ( in.peek() == '"')//2 consecutive quotes resolve to 1
            {
                ss << (char)in.get();
            }
            else //endquotechar
            {
                inquotes=false;
                ss << c;
            }
        }
        else if (!inquotes && c==delimiter) //end of field
        {
            row.push_back( ss.str() );
            ss.str("");
        }
        else if (!inquotes && (c=='\r' || c=='\n') )
        {
            if(in.peek()=='\n') { in.get(); }
            row.push_back( ss.str() );
            return row;
        }
        else
        {
            ss << c;
        }
    }
    return row;
}

std::vector<std::string> csv_read_row(std::string &line, char delimiter)
{
    std::stringstream ss(line);
    return csv_read_row(ss, delimiter);
}

void csv_write(std::vector<std::string> header,
    std::vector<std::vector<std::vector<std::string>>*> &ranks,
    std::string filename) {
    std::ofstream outfile(filename);
    {
        bool first{true};
        for (auto col : header) {
            if (!first) {
                outfile << ",";
            } else {
                first = false;
            }
            outfile << col;
        }
        outfile << "\n";
    }
    for (auto rank : ranks) {
        for (auto row : *rank) {
            bool first{true};
            for (auto col : row) {
                if (!first) {
                    outfile << ",";
                } else {
                    first = false;
                }
                outfile << col;
            }
            outfile << "\n";
        }
    }
}

} // namespace treemerge
} // namespace apex
