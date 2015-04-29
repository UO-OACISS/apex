
#include "apex_types.h"
#include "profile.hpp"
#include "rapidjson/reader.h"
#include "rapidjson/error/en.h"
#include <iostream>
#include <string>
#include <map>

using namespace rapidjson;

typedef enum _handler_states {
    EXPECT_ROOT_START = 0,
    EXPECT_ROOT_END_OR_KEY_NAME,
    EXPECT_PROFILE_NAME,
    EXPECT_STRING_OR_PROFILE_ARRAY,
    EXPECT_VALUE_ARRAY,
    EXPECT_PROFILE_START_OR_ARRAY_END,
    EXPECT_VALUE_OR_ARRAY_END
} handler_states;

struct json_handler {
    handler_states _state;
    std::string _current_key;
    SizeType _profile_count;
    SizeType _value_count;
    apex_profile_type _profile_type;
    apex::profile * _current_profile;
    std::map<std::string, apex::profile *> name_map;
    json_handler(void) : _state(EXPECT_ROOT_START) {}
    bool Null() ;
    bool Bool(bool b) ;
    bool Int(int i) ;
    bool Uint(unsigned u) ;
    bool Int64(int64_t i) ;
    bool Uint64(uint64_t u) ;
    bool Double(double d) ;
    bool String(const char* str, SizeType length, bool copy) ;
    bool StartObject() ;
    bool Key(const char* str, SizeType length, bool copy) ;
    bool EndObject(SizeType memberCount) ; 
    bool StartArray() ;
    bool EndArray(SizeType elementCount) ;
};


