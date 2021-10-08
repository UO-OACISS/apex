/**
 * @file gzstream.h
 * @author Chase Geigle
 *
 * Code is adapted from https://github.com/meta-toolkit/meta, on 9/21/2021
 * All files in META are dual-licensed under the MIT and NCSA licenses. For more
 * details, consult LICENSE.txt in root directory.
 */

#pragma once

#include <streambuf>
#include <istream>
#include <ostream>
#include <type_traits>
#include <vector>
#include <zlib.h>

namespace apex { namespace io {

namespace detail
{
template <class StreamBuf>
struct has_bytes_read
{
    template <class T>
    static constexpr auto check(T*) ->
        typename std::is_same<decltype(std::declval<T>().bytes_read()),
                              uint64_t>::type;

    template <class>
    static constexpr auto check(...) -> std::false_type;

    using type = decltype(check<StreamBuf>(nullptr));
    const static constexpr bool value = type::value;
};
}

template <class StreamBase, class ZStreamBuf>
class zfstream : public StreamBase
{
  public:
    using streambuf_type = ZStreamBuf;

    explicit zfstream(const std::string& name, const char* openmode)
        : StreamBase{&buffer_}, buffer_{name.c_str(), openmode}
    {
        if (buffer_.is_open())
            this->clear();
        else
            this->setstate(std::ios::badbit);
    }

    streambuf_type* rdbuf() const
    {
        return const_cast<streambuf_type*>(&buffer_);
    }

    void flush()
    {
        buffer_.sync();
    }

    template <class T = ZStreamBuf>
    typename std::enable_if<detail::has_bytes_read<T>::value
                                && std::is_same<StreamBase,
                                                std::istream>::value,
                            uint64_t>::type
    bytes_read() const
    {
        return buffer_.bytes_read();
    }

    bool is_open() {
        return buffer_.is_open();
    }

    void close() {
        return buffer_.close();
    }

  private:
    streambuf_type buffer_;
};

/**
 * A base class for an input stream that reads from a compressed streambuf
 * object.
 */
template <class ZStreamBuf>
class zifstream : public zfstream<std::istream, ZStreamBuf>
{
  public:
    explicit zifstream(const std::string& name)
        : zfstream<std::istream, ZStreamBuf>(name, "rb")
    {
        // nothing
    }
};

/**
 * A base class for an output stream that writes to a compressed streambuf
 * object.
 */
template <class ZStreamBuf>
class zofstream : public zfstream<std::ostream, ZStreamBuf>
{
  public:
    explicit zofstream(const std::string& name)
        : zfstream<std::ostream, ZStreamBuf>(name, "wb")
    {
        // nothing
    }
};

class gzstreambuf : public std::streambuf
{
  public:
    gzstreambuf(const char* filename, const char* openmode,
                size_t buffer_size = 512);
    ~gzstreambuf();
    int_type underflow() override;
    int_type overflow(int_type ch) override;
    int sync() override;
    bool is_open() const;
    void close(void);
  private:
    std::vector<char> buffer_;
    gzFile file_;
};

/**
 * An ifstream that can read gz compressed files.
 */
using gzifstream = zifstream<gzstreambuf>;

/**
 * An ofstream that can write gz compressed files.
 */
using gzofstream = zofstream<gzstreambuf>;

} // namespace io
} // namespace apex
