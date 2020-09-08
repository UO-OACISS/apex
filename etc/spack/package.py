# Copyright 2013-2020 Lawrence Livermore National Security, LLC and other
# Spack Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

from spack import *
import sys

class Apex(CMakePackage):
    """Autonomic Performance Environment for eXascale (APEX)."""

    maintainers = ['khuck']
    homepage = "https://github.com/khuck/xpress-apex"
    url      = "https://github.com/khuck/xpress-apex/archive/v2.2.0.tar.gz"
    git      = "https://github.com/khuck/xpress-apex"

    version('develop', branch='develop')
    version('master', branch='master')
    version('2.2.0', sha256='cd5eddb1f6d26b7dbb4a8afeca2aa28036c7d0987e0af0400f4f96733889c75c')

    # Disable some default dependencies on Darwin/OSX
    darwin_default = False
    if sys.platform != 'darwin':
        darwin_default = True

    # Enable by default
    variant('activeharmony', default=True, description='Enables Active Harmony support')
    variant('binutils', default=True, description='Enables Binutils support')
    variant('otf2', default=True, description='Enables OTF2 support')
    variant('openmp', default=darwin_default, description='Enables OpenMP support')
    variant('papi', default=darwin_default, description='Enables PAPI support')
    variant('gperftools', default=False, description='Enables Google PerfTools TCMalloc support')
    variant('jemalloc', default=False, description='Enables JEMalloc support')

    # Disable by default
    variant('cuda', default=False, description='Enables CUDA support')
    variant('boost', default=False, description='Enables Boost support')

    # Dependencies
    depends_on('cmake')
    depends_on('binutils+libiberty+headers', when='+binutils')
    depends_on('activeharmony@4.6:', when='+activeharmony')
    depends_on('otf2', when='+otf2')
    depends_on('gperftools', when='+gperftools')
    depends_on('jemalloc', when='+jemalloc')
    depends_on('papi', when='+papi')
    depends_on('cuda', when='+cuda')
    depends_on('boost@1.54:', when='+boost')

    # Conflicts
    conflicts('+jemalloc', when='+gperftools')


    def cmake_args(self):
        args = []
        spec = self.spec
        if '+cuda' in spec:
            args.append('-DAPEX_WITH_CUDA=TRUE')
        if '+binutils' in spec:
            args.append('-DBFD_ROOT={0}'.format(spec['binutils'].prefix))
            args.append('-DUSE_BFD=TRUE')
        if '+activeharmony' in spec:
            args.append('-DACTIVEHARMONY_ROOT={0}'.format(spec['activeharmony'].prefix))
            args.append('-DUSE_ACTIVEHARMONY=TRUE')
        if '+otf2' in spec:
            args.append('-DOTF2_ROOT={0}'.format(spec['otf2'].prefix))
            args.append('-DUSE_OTF2=TRUE')
        if '+openmp' in spec:
            args.append('-DUSE_OMPT=TRUE')
        if '+gperftools' in spec:
            args.append('-DGPERFTOOLS_ROOT={0}'.format(spec['gperftools'].prefix))
            args.append('-DUSE_TCMALLOC=TRUE')
        if '+jemalloc' in spec:
            args.append('-DJEMALLOC_ROOT={0}'.format(spec['jemalloc'].prefix))
            args.append('-DUSE_JEMALLOC=TRUE')
        if '+boost' in spec:
            args.append('-DBOOST_ROOT={0}'.format(spec['boost'].prefix))

        return args

