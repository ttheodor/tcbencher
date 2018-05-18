flags = [
    '-x',
    'c++',
    '-std=c++17',
    '-isystem', '/usr/include/c++/8.1.0/',
    '-isystem', '/usr/include/c++/8.1.0/x86_64-pc-linux-gnu',
    '-isystem', '/usr/include/c++/8.1.0/backward',
    '-isystem', '/usr/include/',
    '-isystem', '/opt/cuda/include',
    '-isystem', '/home/theo/tcbencher/third-party/pcg/include',
    '-isystem', '/home/theo/tcbencher/third-party/tc/',
    '-isystem', '/home/theo/tcbencher/build/third-party/tc/proto',
    '-I', '/home/theo/tcbencher/src',
]


def FlagsForFile( filename, **kwargs ):
  return {
    'flags': flags
  }
