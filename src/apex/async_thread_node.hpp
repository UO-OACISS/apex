/*
 * Copyright (c) 2014-2021 Kevin Huck
 * Copyright (c) 2014-2021 University of Oregon
 *
 * Distributed under the Boost Software License, Version 1.0. (See accompanying
 * file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
 */

#pragma once

#include <iomanip>
#include <sstream>
#include <string>

namespace apex {

    class base_thread_node {
    public:
        uint32_t _device;
        uint32_t _context;
        uint32_t _stream;
        apex_async_activity_t _activity;
        base_thread_node(uint32_t device, uint32_t context, uint32_t stream,
            apex_async_activity_t activity) :
            _device(device), _context(context), _stream(stream), _activity(activity) { }
        virtual bool operator==(const base_thread_node &rhs) const {
            return (_device   == rhs._device &&
                    _context  == rhs._context &&
                    _stream   == rhs._stream &&
                    _activity == rhs._activity);
        }
        virtual bool operator<(const base_thread_node &rhs) const {
            if (_device<rhs._device) {
                return true;
            } else if (_device == rhs._device && _context < rhs._context) {
                return true;
            } else if (_device == rhs._device && _context == rhs._context &&
                _stream < rhs._stream) {
                return true;
            } else if (_device == rhs._device && _context == rhs._context &&
                _stream == rhs._stream && _activity < rhs._activity &&
                apex_options::use_otf2()) {
                return true;
            }
            return false;
        }
        virtual std::string name () {
            std::stringstream ss;
            ss << "GPU [" << _device << "]";
            std::string tmp{ss.str()};
            return tmp;
        }
        virtual uint32_t sortable_tid () {
            uint32_t tid = ((_device+1) << 28);
            return tid;
        }
    };

    class ompt_thread_node : public base_thread_node {
    public:
        ompt_thread_node(uint32_t device, uint32_t thread,
            apex_async_activity_t activity) :
            base_thread_node(device, 0, thread, activity) { }
        virtual std::string name () {
            std::stringstream ss;
            ss << "GPU [" << _device << ":" << _stream << "]";
            std::string tmp{ss.str()};
            return tmp;
        }
        virtual uint32_t sortable_tid () {
            uint32_t tid = ((_device+1) << 28);
            tid = tid + _stream;
            return tid;
        }
    };


    class cuda_thread_node : public base_thread_node {
    public:
        cuda_thread_node(uint32_t device, uint32_t context, uint32_t stream,
            apex_async_activity_t activity) :
            base_thread_node(device, context, stream, activity) { }
        virtual std::string name () {
            std::stringstream ss;
            ss << "CUDA [" << _device << ":" << _context
               << ":" << std::setfill('0') << std::setw(5) << _stream << "]";
            std::string tmp{ss.str()};
            return tmp;
        }
        virtual uint32_t sortable_tid () {
            uint32_t tid = ((_device+1) << 28);
            tid = tid + (_context << 22);
            tid = tid + _stream;
            return tid;
        }
    };

    class hip_thread_node : public base_thread_node {
    public:
        hip_thread_node(uint32_t device, uint32_t command_queue,
            apex_async_activity_t activity) :
            base_thread_node(device, 0, command_queue, activity) { }
        virtual std::string name () {
            std::stringstream ss;
            ss << "HIP [" << _device
               << ":" << std::setfill('0') << std::setw(5) << _stream << "]";
            std::string tmp{ss.str()};
            return tmp;
        }
        virtual uint32_t sortable_tid () {
            uint32_t tid = ((_device+1) << 28);
            tid = tid + _stream;
            return tid;
        }
    };

}

