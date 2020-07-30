//  Copyright (c) 2014 University of Oregon
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

namespace apex {

    class cuda_thread_node {
    public:
        uint32_t _device;
        uint32_t _context;
        uint32_t _stream;
        cuda_thread_node(uint32_t device, uint32_t context, uint32_t stream) :
            _device(device), _context(context), _stream(stream) { }
        bool operator==(const cuda_thread_node &rhs) const {
            return (_device==rhs._device &&
                    _context==rhs._context &&
                    _stream==rhs._stream);
        }
        bool operator<(const cuda_thread_node &rhs) const {
            if (_device<rhs._device) {
                return true;
            } else if (_device == rhs._device && _context<rhs._context) {
                return true;
            } else if (_device == rhs._device && _context==rhs._context && _stream<rhs._stream) {
                return true;
            }
            return false;
        }
    };

}

