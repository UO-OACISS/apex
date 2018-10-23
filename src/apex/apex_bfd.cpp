//  Copyright (c) 2014-2018 University of Oregon
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>
#include <set>

#include "apex_bfd.h"
#include "apex.hpp"

// Intel compiler doesn't support this attribute.
#if defined (__INTEL_COMPILER)
#ifndef ATTRIBUTE_RETURNS_NONNULL
# define ATTRIBUTE_RETURNS_NONNULL
#endif
#endif

// bfd.h expects these to be defined, so define them.
// otherwise, we get a compilation error.
#if !defined PACKAGE
#define PACKAGE
#endif
#if !defined PACKAGE_VERSION
#define PACKAGE_VERSION
#endif
#include <bfd.h>
#if APEX_BFD >= 022300
#include <elf-bfd.h>
#endif
#include <dirent.h>
#include <stdint.h>
#include <cctype>
#include <sstream>

#if defined(HAVE_GNU_DEMANGLE) && HAVE_GNU_DEMANGLE
#define HAVE_DECL_BASENAME 1

#include <ansidecl.h>
// some versions of ansidecl.h don't have this macro, some versions of libiberty need it.
#ifndef ATTRIBUTE_RETURNS_NONNULL
# if (GCC_VERSION >= 4009)
#  define ATTRIBUTE_RETURNS_NONNULL __attribute__ ((__returns_nonnull__))
# else
#  define ATTRIBUTE_RETURNS_NONNULL
# endif /* GNUC >= 4.9 */
#endif /* ATTRIBUTE_RETURNS_NONNULL */

#include <demangle.h>
#define DEFAULT_DEMANGLE_FLAGS DMGL_PARAMS | DMGL_ANSI | DMGL_VERBOSE | DMGL_TYPES
#ifdef __PGI
#define DEMANGLE_FLAGS DEFAULT_DEMANGLE_FLAGS | DMGL_ARM
#else
#define DEMANGLE_FLAGS DEFAULT_DEMANGLE_FLAGS
#endif // __PGI
#endif // HAVE_GNU_DEMANGLE

#ifdef __APPLE__
#include <mach-o/dyld.h>
#endif

#if defined(APEX_WINDOWS) && defined(APEX_MINGW)
#include <windows.h>
#include <psapi.h>
#endif

using namespace std;

char const * Apex_bfd_internal_getExecutablePath();

struct ApexBfdModule
{
  ApexBfdModule() :
      bfdImage(nullptr), syms(nullptr), nr_all_syms(0), dynamic(false), bfdOpen(false),
      lastResolveFailed(false), processCode(APEX_BFD_SYMTAB_NOT_LOADED)
  { }

  ~ApexBfdModule() {
    if (bfdImage && bfdOpen)
      bfd_close(bfdImage);
    free(syms);
    syms = nullptr;
  }

#ifdef APEX_INTEL12
  // Meant for consumption by the Intel12 workaround only.
  void markLastResult(bool success) {
    lastResolveFailed = !success;
  }
#endif

  bool apex_loadSymbolTable(char const * path)
  {
#ifdef APEX_INTEL12
    // Nasty hack because Intel 12 is broken with Bfd 2.2x and
    //   requires a complete reset of BFD. The latter's internals
    //   becomes corrupted on a bad address from Intel 12 binaries.
    if (lastResolveFailed) {
      bfd_init();
      bfdOpen = false;
    }
#endif /* APEX_INTEL12 */

    // Executable symbol table is already loaded.
    if (bfdOpen) return true;

    Apex_bfd_initializeBfd();

    //printf("apex_loadSymbolTable: opening [%s]\n", path);
    if (!(bfdImage = bfd_openr(path, 0))) {
      //printf("apex_loadSymbolTable: Failed to open [%s]\n", path);
      return (bfdOpen = false);
    }

#if APEX_BFD >= 022200
    // Decompress sections
    bfdImage->flags |= BFD_DECOMPRESS;
#endif

    if (!bfd_check_format(bfdImage, bfd_object)) {
      //printf("apex_loadSymbolTable: bfd format check failed [%s]\n", path);
      return (bfdOpen = false);
    }

    char **matching;
    if (!bfd_check_format_matches(bfdImage, bfd_object, &matching)) {
      //printf("apex_loadSymbolTable: bfd format mismatch [%s]\n", path);
      if (bfd_get_error() == bfd_error_file_ambiguously_recognized) {
        //printf("apex_loadSymbolTable: Matching formats:");
        for (char ** p = matching; *p; ++p) {
          //printf(" %s", *p);
        }
        //printf("\n");
      }
      free(matching);
    }

    if (!(bfd_get_file_flags(bfdImage) & HAS_SYMS)) {
      //printf("apex_loadSymbolTable: bfd has no symbols [%s]\n", path);
      return (bfdOpen = false);
    }

    size_t size = bfd_get_symtab_upper_bound(bfdImage);
    if (!size) {
      //printf("apex_loadSymbolTable: Retrying with dynamic\n");
      size = bfd_get_dynamic_symtab_upper_bound(bfdImage);
      dynamic = true;
      if (!size) {
        //printf("apex_loadSymbolTable: Cannot get symbol table size [%s]\n", path);
        return (bfdOpen = false);
      }
    }

    syms = (asymbol **)malloc(size);
    if (dynamic) {
      nr_all_syms = bfd_canonicalize_dynamic_symtab(bfdImage, syms);
    } else {
      nr_all_syms = bfd_canonicalize_symtab(bfdImage, syms);
    }
    bfdOpen = nr_all_syms > 0;

    //printf("apex_loadSymbolTable: %s contains %d canonical symbols\n", path,
    //nr_all_syms);

    return bfdOpen;
  }

  bfd *bfdImage;
  asymbol **syms;
  size_t nr_all_syms;
  bool dynamic;

  // For EBS book-keeping
  bool bfdOpen;    // once open, symtabs are loaded and never released
  bool lastResolveFailed;

  // Remember the result of the last process to avoid reprocessing
  int processCode;
};

struct ApexBfdUnit
{
  ApexBfdUnit() : apex_objopen_counter(-1) {
    executablePath = Apex_bfd_internal_getExecutablePath();
    executableModule = new ApexBfdModule;
  }

  void ClearMaps() {
    for (size_t i = 0; i < addressMaps.size(); ++i) {
      if (addressMaps[i]) {
      delete addressMaps[i];
      }
    }
    addressMaps.clear();
  }

  void ClearModules() {
    for (size_t i = 0; i < modules.size(); ++i) {
      delete modules[i];
    }
    modules.clear();
  }

  int apex_objopen_counter;
  char const * executablePath;
  ApexBfdModule * executableModule;
  vector<ApexBfdAddrMap*> addressMaps;
  vector<ApexBfdModule*> modules;

};

struct apex_LocateAddressData
{
  apex_LocateAddressData(ApexBfdModule * _module, ApexBfdInfo & _info) :
      found(false), module(_module), info(_info)
  { }

  bool found;
  ApexBfdModule * module;
  ApexBfdInfo & info;
};

// Internal function prototypes
bool Apex_bfd_internal_loadSymTab(ApexBfdUnit *unit, int moduleIndex);
bool Apex_bfd_internal_loadExecSymTab(ApexBfdUnit *unit);
int Apex_bfd_internal_getModuleIndex(ApexBfdUnit *unit,
    unsigned long probe_addr);
ApexBfdModule * Apex_bfd_internal_getModuleFromIdx(ApexBfdUnit *unit,
    int moduleIndex);
void Apex_bfd_internal_locateAddress(bfd *bfdptr, asection *section,
    void *data ATTRIBUTE_UNUSED);
void Apex_bfd_internal_updateProcSelfMaps(int unit_index);

#if (defined(APEX_BGP) || defined(APEX_BGQ))
void Apex_bfd_internal_updateBGPMaps(ApexBfdUnit *unit);
#endif /* APEX_BGP || APEX_BGQ */

//////////////////////////////////////////////////////////////////////
// Instead of using a global var., use static inside a function  to
// ensure that non-local static variables are initialised before being
// used (Ref: Scott Meyers, Item 47 Eff. C++).
//////////////////////////////////////////////////////////////////////
//typedef std::vector<ApexBfdUnit*> apex_bfd_unit_vector_t;

struct apex_bfd_unit_vector_t : public std::vector<ApexBfdUnit*>
{
    apex_bfd_unit_vector_t() {}
    virtual ~apex_bfd_unit_vector_t() {
        //Wait! We might not be done! Unbelieveable as it may seem, this object
        //could (and does sometimes) get destroyed BEFORE we have resolved the
        //addresses. Bummer.
        apex::finalize();
        //printf("deleting BFD objects\n");
    }
};

apex_bfd_unit_vector_t & apex_ThebfdUnits(void)
{
    // BFD units (e.g. executables and their dynamic libraries)
    static apex_bfd_unit_vector_t internal_bfd_units;
    return internal_bfd_units;
}

void Apex_delete_bfd_units() {
  static bool deleted = false;
  if (!deleted) {
    deleted = true;
    apex_bfd_unit_vector_t units = apex_ThebfdUnits();
    for (std::vector<ApexBfdUnit*>::iterator it = units.begin();
         it != units.end(); ++it) {
      ApexBfdUnit * unit = *it;
      unit->ClearMaps();
      unit->ClearModules();
      delete unit->executableModule;
      delete unit;
    }
    units.clear();
  }
}

typedef int * (*apex_objopen_counter_t)(void);
apex_objopen_counter_t apex_objopen_counter = nullptr;

int get_apex_objopen_counter(void)
{
  if (apex_objopen_counter) {
    return *(apex_objopen_counter());
  }
  return 0;
}

void set_apex_objopen_counter(int value)
{
  if (apex_objopen_counter) {
    *(apex_objopen_counter()) = value;
  }
}

extern "C"
void Apex_bfd_register_apex_objopen_counter(apex_objopen_counter_t handle)
{
  apex_objopen_counter = handle;
}

//
// Main interface functions
//

void Apex_bfd_initializeBfd()
{
  static bool bfdInitialized = false;
  if (!bfdInitialized) {
    bfd_init();
    bfdInitialized = true;
  }
}

apex_bfd_handle_t Apex_bfd_registerUnit()
{
  apex_bfd_handle_t ret = apex_ThebfdUnits().size();
  apex_ThebfdUnits().push_back(new ApexBfdUnit);

  //printf("Apex_bfd_registerUnit: Unit %d registered and initialized\n", ret);

  // Initialize the first address maps for the unit.
  Apex_bfd_updateAddressMaps(ret);

  return ret;
}

bool Apex_bfd_checkHandle(apex_bfd_handle_t handle)
{
  if (handle == APEX_BFD_NULL_HANDLE) {
    //printf("ApexBfd: Warning - attempt to use uninitialized BFD handle\n");
    return false;
  }
  // cast to unsigned to prevent compiler warnings
  if ((unsigned int)(handle) >= apex_ThebfdUnits().size()) {
    //printf("ApexBfd: Warning - invalid BFD unit handle %d, max value %d\n",
    //handle, apex_ThebfdUnits().size());
    return false;
  }
  return (handle >= 0);
}

void Apex_bfd_internal_updateProcSelfMaps(int unit_index)
{
  // *CWL* - This is important! We DO NOT want to use /proc/self/maps on
  //         the BGP because the information acquired comes from the I/O nodes
  //         and not the compute nodes. You could end up with an overlapping
  //         range for address resolution if used!
#if (defined (APEX_BGP) || defined(APEX_BGQ) || (APEX_WINDOWS))
  /* do nothing */
  // *JCL* - Windows has no /proc filesystem, so don't try to use it
#else
  ApexBfdUnit *unit = apex_ThebfdUnits()[unit_index];

  // Note: Linux systems only.
  FILE * mapsfile = fopen("/proc/self/maps", "r");
  if(!mapsfile) {
    return;
  }

  char line[4096];
  unsigned long start, end, offset, device_major, device_minor;
  unsigned filenumber;
  char module[4096];
  char perms[5];
  while (!feof(mapsfile)) {
    if (fgets(line, 4096, mapsfile) == nullptr) { break; }
    //printf("%s", line); fflush(stdout);
    sscanf(line, "%lx-%lx %s %lx %lx:%lx %u %[^\n]", &start, &end, perms,
        &offset, &device_major, &device_minor, &filenumber, module);
    if (*module && ((strcmp(perms, "r-xp") == 0) ||
            (strcmp(perms, "rwxp") == 0)))
    {
      unit->addressMaps.push_back(new ApexBfdAddrMap(start, end,
        offset, module));
      unit->modules.push_back(new ApexBfdModule);
    }
  }
  fclose(mapsfile);
#endif /* APEX_BGP || APEX_BGQ || APEX_WINDOWS */
}

#if (defined(APEX_BGP) || defined(APEX_BGQ))
int Apex_bfd_internal_BGP_dl_iter_callback(struct dl_phdr_info * info,
    size_t size, void * data)
{
  if (strlen(info->dlpi_name) == 0) {
    //printf("Apex_bfd_internal_BGP_dl_iter_callback: Nameless module. Ignored.\n");
    return 0;
  }
  //printf("Apex_bfd_internal_BGP_dl_iter_callback: Processing module [%s]\n",
  //  info->dlpi_name);

  ApexBfdUnit * unit = (ApexBfdUnit *)data;

  // assuming the max of the physical addresses of each segment added to the
  // memory size yields the end of the address range.
  unsigned long max_addr = 0;
  for (int j = 0; j < info->dlpi_phnum; j++) {
    unsigned long local_max = (unsigned long)info->dlpi_phdr[j].p_paddr +
        (unsigned long)info->dlpi_phdr[j].p_memsz;
    if (local_max > max_addr) {
      max_addr = local_max;
    }
  }
  unsigned long start = (unsigned long)info->dlpi_addr;
  ApexBfdAddrMap * map = new ApexBfdAddrMap(start,
    start + max_addr, 0, info->dlpi_name);
  //printf("BG Module: %s, %p-%p (%d)\n", map->name, map->start, map->end,
  //map->offset);
  unit->addressMaps.push_back(map);
  unit->modules.push_back(new ApexBfdModule);
  return 0;
}

void Apex_bfd_internal_updateBGPMaps(ApexBfdUnit *unit)
{
  dl_iterate_phdr(Apex_bfd_internal_BGP_dl_iter_callback, (void *)unit);
}
#endif /* APEX_BGP || APEX_BGQ */

#if defined(APEX_WINDOWS) && defined(APEX_MINGW)
// Executables compiled by MinGW are strange beasts in that
// they use GNU debugger symbols, but are Windows executables.
// BFD support for windows is incomplete (e.g. dl_iterate_phdr
// is not implemented and probably never will be), so we must
// use the Windows API to walk through the PE imports directory
// to discover our external modules (e.g. DLLs).  However, we
// still need BFD to parse the GNU debugger symbols.  In fact,
// the DEBUG PE header of an executable produced by MinGW is
// just an empty table.
void Apex_bfd_internal_updateWindowsMaps(ApexBfdUnit *unit)
{

  // Use Windows Process API to find modules
  // This is preferable to walking the PE file headers with
  // Windows API calls because it provides a more complete
  // and accurate picture of the process memory layout, and
  // memory addresses aren't truncated on 64-bit Windows.

  HMODULE hMod[1024];// Handles for each module
  HANDLE hProc;// A handle on the current process
  DWORD cbNeeded;// Bytes needed to store all handles
  MODULEINFO modInfo;// Information about a module

  // Get the process handle
  hProc = OpenProcess(PROCESS_QUERY_INFORMATION | PROCESS_VM_READ,
      FALSE, GetCurrentProcessId());
  if (hProc == nullptr) {
    //printf("Apex_bfd_internal_updateWindowsMaps: Cannot get process handle.\n");
    return;
  }

  // Get handles on all modules in this process
  if (EnumProcessModules(hProc, hMod, sizeof(hMod), &cbNeeded) == 0) {
    //printf("Apex_bfd_internal_updateWindowsMaps: Cannot enumerate process modules.\n");
    return;
  }

  // Calculate number of handles enumerated
  size_t const nModHandle = cbNeeded / sizeof(HMODULE);

  // Iterate over module handles
  for(size_t i=0; i<nModHandle; ++i) {

    // Get the module information structure
    if(GetModuleInformation(hProc,
        hMod[i], &modInfo, sizeof(modInfo)) == 0) {
      //printf("Apex_bfd_internal_updateWindowsMaps: Cannot get module info
      //(handle 0x%x).\n", hMod[i]);
      continue;
    }

    // Create a new BFD map for this module
    ApexBfdAddrMap * map = new ApexBfdAddrMap;
    map->start = Apex_convert_ptr_to_unsigned_long(modInfo.lpBaseOfDll);
    map->end = map->start + modInfo.SizeOfImage;
    map->offset = 0;

    // Get the full module path name for the map
    if(GetModuleFileNameEx(hProc, hMod[i],
        map->name, sizeof(map->name)) == 0) {
      //printf("Apex_bfd_internal_updateWindowsMaps: Cannot get absolute path
      //to module (handle 0x%x).\n", hMod[i]);
      continue;
    }

    unit->addressMaps.push_back(map);
    unit->modules.push_back(new ApexBfdModule);
  }

  // Release the process handle
  CloseHandle(hProc);
}
#endif /* APEX_WINDOWS && APEX_MINGW */

void Apex_bfd_updateAddressMaps(apex_bfd_handle_t handle)
{
  if (!Apex_bfd_checkHandle(handle)) return;

  ApexBfdUnit * unit = apex_ThebfdUnits()[handle];

  unit->ClearMaps();
  unit->ClearModules();

#if defined(APEX_BGP) || defined(APEX_BGQ)
  Apex_bfd_internal_updateBGPMaps(unit);
#elif defined(APEX_WINDOWS) && defined(APEX_MINGW)
  Apex_bfd_internal_updateWindowsMaps(unit);
#else
  Apex_bfd_internal_updateProcSelfMaps(handle);
#endif

  unit->apex_objopen_counter = get_apex_objopen_counter();

  //printf("Apex_bfd_updateAddressMaps: %d modules discovered\n",
  //unit->modules.size());
}

vector<ApexBfdAddrMap*> const & Apex_bfd_getAddressMaps(
    apex_bfd_handle_t handle) {
  Apex_bfd_checkHandle(handle);
  return apex_ThebfdUnits()[handle]->addressMaps;
}

apex_bfd_module_handle_t Apex_bfd_getModuleHandle(
    apex_bfd_handle_t handle, unsigned long probeAddr)
{
  if (!Apex_bfd_checkHandle(handle)) {
    return APEX_BFD_INVALID_MODULE;
  }
  ApexBfdUnit *unit = apex_ThebfdUnits()[handle];

  int matchingIdx = Apex_bfd_internal_getModuleIndex(unit, probeAddr);
  if (matchingIdx != -1) {
    return (apex_bfd_module_handle_t)matchingIdx;
  }
  return APEX_BFD_NULL_MODULE_HANDLE;
}

ApexBfdAddrMap const * Apex_bfd_getAddressMap(
    apex_bfd_handle_t handle, unsigned long probe_addr)
{
  if (!Apex_bfd_checkHandle(handle)) {
    return nullptr;
  }

  ApexBfdUnit *unit = apex_ThebfdUnits()[handle];
  int matchingIdx = Apex_bfd_internal_getModuleIndex(unit, probe_addr);
  if (matchingIdx == -1) {
    return nullptr;
  }

  return unit->addressMaps[matchingIdx];
}

char const * Apex_bfd_internal_tryDemangle(bfd * bfdImage,
    char const * funcname)
{
  char const * demangled = nullptr;
#if defined(HAVE_GNU_DEMANGLE) && HAVE_GNU_DEMANGLE
  if (funcname && bfdImage) {
    // Some compilers prepend .text. to the symbol name
    if (strncmp(funcname, ".text.", 6) == 0) {
      funcname += 6;
    }

    // Sampling sometimes gives the names as a long branch offset
    char const * substr = strstr(funcname, ".long_branch_r2off.");
    if (substr) {
      char * tmp = strdup(substr+19);
      // Trim offset address from end of name
      char * p = tmp + strlen(tmp) - 1;
      while (p != tmp && isdigit(*p)) --p;
      if (*p == '+') *p = '\0';
      demangled = bfd_demangle(bfdImage, tmp, DEMANGLE_FLAGS);
      free(tmp);
    } else {
      demangled = bfd_demangle(bfdImage, funcname, DEMANGLE_FLAGS);
    }
  }
#else
  APEX_UNUSED(bfdImage);
#endif
  if (demangled) return demangled;
  return funcname;
}

unsigned long apex_getProbeAddr(bfd * bfdImage, unsigned long pc) {
#if APEX_BFD >= 022300
  if (bfd_get_flavour(bfdImage) == bfd_target_elf_flavour) {
    const struct elf_backend_data * bed = get_elf_backend_data(bfdImage);
    bfd_vma sign = (bfd_vma) 1 << (bed->s->arch_size - 1);
    pc &= (sign << 1) - 1;
    if (bed->sign_extend_vma) {
      pc = (pc ^ sign) - sign;
    }
  }
#else
  APEX_UNUSED(bfdImage);
#endif
  return pc;
}

// Probe for BFD information given a single address.
bool Apex_bfd_resolveBfdInfo(apex_bfd_handle_t handle,
    unsigned long probeAddr, ApexBfdInfo & info)
{
  if (!Apex_bfd_checkHandle(handle)) {
    info.secure(probeAddr);
    return false;
  }

  ApexBfdUnit * unit = apex_ThebfdUnits()[handle];
  if (unit == nullptr) {
      return false;
  }
  ApexBfdModule * module;
  unsigned long addr0;
  unsigned long addr1;

  if (unit->apex_objopen_counter != get_apex_objopen_counter()) {
    Apex_bfd_updateAddressMaps(handle);
  }

  // initialize this so we can check it later
  info.lineno = 0;

  // Discover if we are searching in the executable or a module
  int matchingIdx = Apex_bfd_internal_getModuleIndex(unit, probeAddr);
  if (matchingIdx != -1) {
    if (!Apex_bfd_internal_loadSymTab(unit, matchingIdx)) {
      info.secure(probeAddr);
      return false;
    }
    module = Apex_bfd_internal_getModuleFromIdx(unit, matchingIdx);

    // Calculate search addresses for module search
#if defined(APEX_WINDOWS) && defined(APEX_MINGW)
    addr0 = probeAddr;
    addr1 = probeAddr - unit->addressMaps[matchingIdx]->start;
#else
    addr0 = probeAddr;
    addr1 = probeAddr - unit->addressMaps[matchingIdx]->start;
#endif
  } else {
    if (!Apex_bfd_internal_loadExecSymTab(unit)) {
      info.secure(probeAddr);
      return false;
    }
    module = unit->executableModule;

    // Calculate search addresses for executable search
    // Only the first address is valid for the executable
    addr0 = probeAddr;
    addr1 = 0;
  }

  // Search BFD sections for address
  info.probeAddr = apex_getProbeAddr(module->bfdImage, addr0);
  apex_LocateAddressData data(module, info);
  bfd_map_over_sections(module->bfdImage,
    Apex_bfd_internal_locateAddress, &data);

  // If the data wasn't found where we expected and we are searching
  // in a module, try a few more addresses
  if (!data.found && (module != unit->executableModule)) {
    // Try the second address
    if (addr1 && addr0 != addr1) {
      info.probeAddr = apex_getProbeAddr(module->bfdImage, addr1);
      bfd_map_over_sections(module->bfdImage,
        Apex_bfd_internal_locateAddress, &data);
    }
    // Try the executable
    if (!data.found && Apex_bfd_internal_loadExecSymTab(unit)) {
      info.probeAddr = apex_getProbeAddr(module->bfdImage, probeAddr);
      bfd_map_over_sections(unit->executableModule->bfdImage,
        Apex_bfd_internal_locateAddress, &data);
    }
  }

  // We may have the function name but not the file name
  if (info.funcname && !info.filename) {
    if (matchingIdx != -1) {
      info.filename = unit->addressMaps[matchingIdx]->name;
    } else {
      info.filename = unit->executablePath;
    }
  }

  if (data.found && info.funcname) {
#ifdef APEX_INTEL12
    // For Intel 12 workaround. Inform the module that the previous resolve was
    // successful.
    module->markLastResult(true);
#endif /* APEX_INTEL12 */
    info.demangled = Apex_bfd_internal_tryDemangle(module->bfdImage,
        info.funcname);
    return true;
  }

  // Data wasn't found, so check every symbol's address directly to see if it
  // matches. If so, try to get the function name from the symbol name.
  for (asymbol ** s = module->syms; *s; s++) {
    asymbol const & asym = **s;
     // Skip useless symbols (e.g. line numbers)
    if (asym.name && asym.section->size) {
      // See if the addresses match
      unsigned long addr = asym.section->vma + asym.value;
      if (addr == probeAddr) {
        // Get symbol name and massage it
        char const * name = asym.name;
        if (name[0] == '.') {
          char const * mark = strchr(const_cast<char*>(name), '$');
          if (mark) name = mark + 1;
        }
        info.demangled = Apex_bfd_internal_tryDemangle(module->bfdImage, name);
#ifdef APEX_INTEL12
        // For Intel 12 workaround.
        // Inform the module that the previous resolve was successful.
        module->markLastResult(true);
#endif /* APEX_INTEL12 */
        return true;
      }
    }
  }

  // At this point we were unable to resolve the symbol.

#ifdef APEX_INTEL12
  // For Intel 12 workaround. Inform the module that the previous resolve failed.
  module->markLastResult(false);
#endif /* APEX_INTEL12 */

  //printf("result: %p, %s, %s, %d\n", probeAddr,
  //info.funcname, info.filename, info.lineno);

  // we might have partial information, like filename and line number.
  // MOST LIKELY this is an outlined region or some other code that the compiler
  // generated.
  if ((info.funcname == nullptr) && (info.filename != nullptr) && (info.lineno > 0)) {
    info.probeAddr = probeAddr;
    info.funcname = (char*)malloc(32);
    sprintf(const_cast<char*>(info.funcname), "anonymous");
    return true;
  }

  // Couldn't resolve the address so fill in fields as best we can.
  if (info.funcname == nullptr) {
    info.funcname = (char*)malloc(128);
    sprintf(const_cast<char*>(info.funcname), "addr=<%lx>", probeAddr);
  }
  if (info.filename == nullptr) {
    if (matchingIdx != -1) {
      info.filename = unit->addressMaps[matchingIdx]->name;
    } else {
      info.filename = unit->executablePath;
    }
  }
  info.probeAddr = probeAddr;
  info.lineno = 0;

  return false;
}

void Apex_bfd_internal_iterateOverSymtab(ApexBfdModule * module,
    ApexBfdIterFn fn, unsigned long offset)
{
  // Apply the iterator function to all symbols in the table
  for (asymbol ** s = module->syms; *s; s++) {
    asymbol const & asym = **s;

    // Skip useless symbols (e.g. line numbers)
    // It would be easier to use BFD_FLAGS, but those aren't reliable
    // since the debug symbol format is unpredictable
    if (!asym.name || !asym.section->size) {
      continue;
    }

    // Calculate symbol address
    unsigned long addr = asym.section->vma + asym.value;

    // Get apprixmate symbol name
    char const * name = asym.name;
    if (name[0] == '.') {
      char const * mark = strchr(const_cast<char*>(name), '$');
      if (mark) name = mark + 1;
    }

    // Apply the iterator function
    // Names will be resolved and demangled later
    fn(addr + offset, name);
  }
}

int Apex_bfd_processBfdExecInfo(apex_bfd_handle_t handle, ApexBfdIterFn fn)
{
  if (!Apex_bfd_checkHandle(handle)) {
    return APEX_BFD_SYMTAB_LOAD_FAILED;
  }
  ApexBfdUnit * unit = apex_ThebfdUnits()[handle];

  char const * execName = unit->executablePath;
  ApexBfdModule * module = unit->executableModule;

  // Only process the executable once.
  if (module->processCode != APEX_BFD_SYMTAB_NOT_LOADED) {
    printf("Apex_bfd_processBfdExecInfo:\n\t%s already processed (code %d).",
        execName, module->processCode);
    printf("  Will not reprocess.\n");
    return module->processCode;
  }
  //printf("Apex_bfd_processBfdExecInfo: processing executable %s\n", execName);

  // Make sure executable symbol table is loaded
  if (!Apex_bfd_internal_loadExecSymTab(unit)) {
    module->processCode = APEX_BFD_SYMTAB_LOAD_FAILED;
    return module->processCode;
  }

  // Process the symbol table
  Apex_bfd_internal_iterateOverSymtab(module, fn, 0);

  module->processCode = APEX_BFD_SYMTAB_LOAD_SUCCESS;
  return module->processCode;
}

int Apex_bfd_processBfdModuleInfo(apex_bfd_handle_t handle,
    apex_bfd_module_handle_t moduleHandle, ApexBfdIterFn fn)
{
  if (!Apex_bfd_checkHandle(handle)) {
    return APEX_BFD_SYMTAB_LOAD_FAILED;
  }
  ApexBfdUnit * unit = apex_ThebfdUnits()[handle];

  unsigned int moduleIdx = (unsigned int)moduleHandle;
  ApexBfdModule * module = Apex_bfd_internal_getModuleFromIdx(unit, moduleIdx);
  char const * name = unit->addressMaps[moduleIdx]->name;

  // Only process the module once.
  if (module->processCode != APEX_BFD_SYMTAB_NOT_LOADED) {
    printf("Apex_bfd_processBfdModuleInfo: %s already processed (code %d).",
        name, module->processCode);
    printf("  Will not reprocess.\n");
    return module->processCode;
  }
  //printf("Apex_bfd_processBfdModuleInfo: processing module %s\n", name);

  // Make sure symbol table is loaded
  if (!Apex_bfd_internal_loadSymTab(unit, moduleHandle)) {
    module->processCode = APEX_BFD_SYMTAB_LOAD_FAILED;
    return module->processCode;
  }

  unsigned int offset;
#if defined(APEX_WINDOWS) && defined(APEX_MINGW)
  offset = 0;
#else
  offset = unit->addressMaps[moduleIdx]->start;
#endif

  // Process the symbol table
  Apex_bfd_internal_iterateOverSymtab(module, fn, offset);

  module->processCode = APEX_BFD_SYMTAB_LOAD_SUCCESS;
  return module->processCode;
}

bool Apex_bfd_internal_loadSymTab(ApexBfdUnit *unit, int moduleIndex)
{
  if ((moduleIndex == APEX_BFD_NULL_MODULE_HANDLE) ||
      (moduleIndex == APEX_BFD_INVALID_MODULE)) {
    return false;
  }

  char const * name = unit->addressMaps[moduleIndex]->name;
  ApexBfdModule * module =
    Apex_bfd_internal_getModuleFromIdx(unit, moduleIndex);

  return module->apex_loadSymbolTable(name);
}

bool Apex_bfd_internal_loadExecSymTab(ApexBfdUnit *unit)
{
  char const * name = unit->executablePath;
  ApexBfdModule * module = unit->executableModule;

  return module->apex_loadSymbolTable(name);
}

// Internal BFD helper functions
int Apex_bfd_internal_getModuleIndex(ApexBfdUnit *unit,
    unsigned long probe_addr)
{
  if (!unit)
    return -1;
  vector<ApexBfdAddrMap*> const & addressMaps = unit->addressMaps;
  for (unsigned int i = 0; i < addressMaps.size(); i++) {
    if (probe_addr >= addressMaps[i]->start &&
        probe_addr <= addressMaps[i]->end) {
        return i;
    }
  }
  return -1;
}

ApexBfdModule * Apex_bfd_internal_getModuleFromIdx(
    ApexBfdUnit * unit, int moduleIndex)
{
  if (moduleIndex == -1) {
    return unit->executableModule;
  }
  return unit->modules[moduleIndex];
}

#if defined(APEX_BGP)
int Apex_bfd_internal_getBGPExePath(char * path)
{
  DIR * pdir = opendir("/jobs");
  if (!pdir) {
    //printf("APEX: ERROR - Failed to open /jobs\n");
    return -1;
  }

  struct dirent * pent;
  for (int i = 0; i < 3; ++i) {
    pent = readdir(pdir);
    if (!pent) {
      //printf("APEX: ERROR - readdir failed on /jobs (i=%d)\n", i);
      return -1;
    }
  }
  sprintf(path, "/jobs/%s/exe", pent->d_name);
  closedir(pdir);

  //printf("Apex_bfd_internal_getBGPExePath: [%s]\n", path);
  return 0;
}
#endif

char const * Apex_bfd_internal_getExecutablePath()
{
  static char path[4096];
  static bool init = false;

  if (!init) {
    if (!init) {
#if defined(APEX_AIX)
      sprintf(path, "/proc/%d/object/a.out", getpid()); // get PID!
#elif defined(APEX_BGP)
      if (Apex_bfd_internal_getBGPExePath(path) != 0) {
        fprintf(stderr, "Apex_bfd_internal_getExecutablePath: "
            "Warning! Cannot find BG/P executable path [%s], "
            "symbols will not be resolved\n", path);
      }
#elif defined(APEX_BGQ)
      sprintf(path, "%s", "/proc/self/exe");
#elif defined(__APPLE__)
      uint32_t size = sizeof(path);
      _NSGetExecutablePath(path, &size);
#elif defined(APEX_WINDOWS) && defined(APEX_MINGW)
      GetModuleFileName(nullptr, path, sizeof(path));
#else
      // Default: Linux systems
      sprintf(path, "%s", "/proc/self/exe");
#endif
      init = true;
    }
  }

  return path;
}

void Apex_bfd_internal_locateAddress(bfd * bfdptr,
    asection * section, void * dataPtr)
{
  // Assume dataPtr != nullptr because if that parameter is
  // nullptr then we've got bigger problems elsewhere in the code
  apex_LocateAddressData & data = *(apex_LocateAddressData*)dataPtr;

  // Skip this section if we've already resolved the address data
  if (data.found) return;

  // Skip this section if it isn't a debug info section
  if ((bfd_get_section_flags(bfdptr, section) & SEC_ALLOC) == 0) return;

  // Skip this section if the address is before the section start
  bfd_vma vma = bfd_get_section_vma(bfdptr, section);
  if (data.info.probeAddr < vma) return;

  // Skip this section if the address is after the section end
  bfd_size_type size = bfd_get_section_size(section);
  if (data.info.probeAddr >= vma + size) return;

  // The section contains this address, so try to resolve info
  // Note that data.info is a reference, so this call sets the
  // ApexBfdInfo fields without an extra copy.  This also means
  // that the pointers in ApexBfdInfo must never be deleted
  // since they point directly into the module's BFD.
#if (APEX_BFD >= 022200)
  data.found = bfd_find_nearest_line_discriminator(bfdptr, section,
      data.module->syms, (data.info.probeAddr - vma),
      &data.info.filename, &data.info.funcname,
      (unsigned int*)&data.info.lineno, &data.info.discriminator);
#else
  data.found = bfd_find_nearest_line(bfdptr, section,
      data.module->syms, (data.info.probeAddr - vma),
      &data.info.filename, &data.info.funcname,
      (unsigned int*)&data.info.lineno);
#endif
  return;
}

