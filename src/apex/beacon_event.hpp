#ifndef BEACON_EVENT_HPP
#define BEACON_EVENT_HPP

#include <string>
#include <boost/serialization/string.hpp>

namespace apex {

    struct apex_beacon_event {
        bool have_name;
        std::string name;
        apex_function_address function_address;
        double value;
        apex_event_type event_type;

        template <typename Archive>
        void serialize (Archive& ar, const unsigned int version) {
            ar & have_name;
            ar & name;
            ar & function_address;
            ar & value;
            ar & event_type;
            APEX_UNUSED(version);
        }
        apex_beacon_event(void) : 
            have_name(false), 
            name(""), 
            function_address(0L), 
            value(0.0), 
            event_type(APEX_STOP_EVENT) {};
    };
}

#endif // BEACON_EVENT_HPP

