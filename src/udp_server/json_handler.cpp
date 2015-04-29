
#include "json_handler.hpp"
#include "apex_types.h"

using namespace rapidjson;

bool json_handler::Null() { std::cout << "Null()" << std::endl; return true; }
bool json_handler::Bool(bool b) { std::cout << "Bool(" << std::boolalpha << b << ")" << std::endl; return true; }
bool json_handler::Int(int i) { std::cout << "Int(" << i << ")" << std::endl; return true; }
bool json_handler::Uint(unsigned u) { std::cout << "Uint(" << u << ")" << std::endl; return true; }
bool json_handler::Int64(int64_t i) { std::cout << "Int64(" << i << ")" << std::endl; return true; }
bool json_handler::Uint64(uint64_t u) { std::cout << "Uint64(" << u << ")" << std::endl; return true; }
bool json_handler::Double(double d) { 
    //std::cout << "Double(" << d << ")" << std::endl; return true; 
    switch (_value_count) {
        case 0:
            _current_profile->get_profile()->calls = d;
            break;
        case 1:
            _current_profile->get_profile()->accumulated = d;
            break;
        case 2:
            _current_profile->get_profile()->minimum = d;
            break;
        case 3:
            _current_profile->get_profile()->maximum = d;
            break;
        case 4:
            _current_profile->get_profile()->sum_squares = d;
            break;
        default:
            std::cerr << "array parse error: " << _value_count << std::endl;
            return false;
    }
    _value_count++;
    return true;
}
bool json_handler::String(const char* str, SizeType length, bool copy) { 
    //std::cout << "String(" << str << ", " << length << ", " << std::boolalpha << copy << ")" << std::endl;
    APEX_UNUSED(copy);
    switch (_state) {
        case EXPECT_STRING_OR_PROFILE_ARRAY:
            if (_current_key.compare("Profile Format") == 0) {
                if (strncmp(str, "APEX", length) == 0) {
                    _state = EXPECT_ROOT_END_OR_KEY_NAME;
                    return true; 
                } else {
                    std::cerr << "JSON parse error: unsupported format" << std::endl;
                    return false;
                }
            }
        default:
            return false;
    }
}
bool json_handler::StartObject() { 
    switch (_state) {
        case EXPECT_ROOT_START:
            _state = EXPECT_ROOT_END_OR_KEY_NAME;
            return true;
        case EXPECT_PROFILE_START_OR_ARRAY_END:
            _state = EXPECT_PROFILE_NAME;
            return true;
        default:
            std::cerr << "object parse error: " << _state << std::endl;
            return false;
    }
}
bool json_handler::Key(const char* str, SizeType length, bool copy) {
    APEX_UNUSED(copy);
    switch (_state) {
        case EXPECT_ROOT_END_OR_KEY_NAME:
            _state = EXPECT_STRING_OR_PROFILE_ARRAY;
            if (strncmp(str, "timers", length) == 0) {
                _profile_type = APEX_TIMER;
            }
            else if (strncmp(str, "counters", length) == 0) {
                _profile_type = APEX_COUNTER;
            }
            return true; 
        case EXPECT_PROFILE_NAME:
            _current_key = std::string(str, length);
            _current_profile = new apex::profile(0.0, true, _profile_type);
            _state = EXPECT_VALUE_ARRAY;
            return true; 
        default:
            std::cerr << "key parse error" << std::endl;
            return false;
    }
}

bool json_handler::EndObject(SizeType memberCount) { 
    //std::cout << "EndObject(" << memberCount << ")" << std::endl; return true; 
    APEX_UNUSED(memberCount);
    return _state == EXPECT_PROFILE_START_OR_ARRAY_END;
}

bool json_handler::StartArray() { 
    switch (_state) {
        case EXPECT_STRING_OR_PROFILE_ARRAY:
            _state = EXPECT_PROFILE_START_OR_ARRAY_END;
            return true; 
        case EXPECT_VALUE_ARRAY:
            _value_count = 0;
            _state = EXPECT_VALUE_OR_ARRAY_END;
            return true; 
        default:
            std::cerr << "array parse error" << std::endl;
            return false;
    }
}

bool json_handler::EndArray(SizeType elementCount) { 
    switch (_state) {
        case EXPECT_PROFILE_START_OR_ARRAY_END:
            _profile_count = elementCount;
            _state = EXPECT_ROOT_END_OR_KEY_NAME;
            return true; 
        case EXPECT_VALUE_OR_ARRAY_END:
            _state = EXPECT_PROFILE_START_OR_ARRAY_END;
            _value_count = elementCount;
            name_map[_current_key] = _current_profile;
            return true; 
        default:
            std::cerr << "array parse error" << std::endl;
            return false;
    }
}



