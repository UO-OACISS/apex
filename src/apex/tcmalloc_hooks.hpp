#ifdef APEX_HAVE_TCMALLOC

#include <cstddef>
#include <atomic>
#include <unordered_map>
#include <mutex>

namespace apex {
namespace tcmalloc {

class tracker {
public:
    std::atomic<size_t> hostTotalAllocated;
    std::unordered_map<const void*,size_t> hostMemoryMap;
    std::mutex hostMapMutex;
    tracker() : hostTotalAllocated(0) { }
};

bool& inWrapper(void);
void NewHook(const void* ptr, size_t size);
void DeleteHook(const void* ptr);
void init_hook();
void destroy_hook();

}
}
#endif // APEX_HAVE_TCMALLOC
