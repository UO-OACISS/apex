#ifndef SENSOR_DATA_HPP
#define SENSOR_DATA_HPP
#include <string>

namespace apex {

class sensor_data {
    public:
        sensor_data(void);
        ~sensor_data(void);
        std::string get_version(void);
        void read_sensors(void);
};

}

#endif // SENSOR_DATA_HPP
