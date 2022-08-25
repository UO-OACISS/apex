/* From https://patorjk.com/software/taag/ */

const char * banner_fire = R"(          (            )
   (      )\ )      ( /(
   )\    (()/( (    )\())
((((_)(   /(_)))\  ((_)\
 )\ _ )\ (_)) ((_) __((_)
 (_)_\(_)| _ \| __|\ \/ /
  / _ \  |  _/| _|  >  <
 /_/ \_\ |_|  |___|/_/\_\
)";

const char * banner_doom = R"(  ___  ______ _______   __
 / _ \ | ___ \  ___\ \ / /
/ /_\ \| |_/ / |__  \ V /
|  _  ||  __/|  __| /   \
| | | || |   | |___/ /^\ \
\_| |_/\_|   \____/\/   \/
)";

const char * banner_bloody = R"( ▄▄▄       ██▓███  ▓█████ ▒██   ██▒
▒████▄    ▓██░  ██▒▓█   ▀ ▒▒ █ █ ▒░
▒██  ▀█▄  ▓██░ ██▓▒▒███   ░░  █   ░
░██▄▄▄▄██ ▒██▄█▓▒ ▒▒▓█  ▄  ░ █ █ ▒
 ▓█   ▓██▒▒██▒ ░  ░░▒████▒▒██▒ ▒██▒
 ▒▒   ▓▒█░▒▓▒░ ░  ░░░ ▒░ ░▒▒ ░ ░▓ ░
  ▒   ▒▒ ░░▒ ░      ░ ░  ░░░   ░▒ ░
  ░   ▒   ░░          ░    ░    ░
      ░  ░            ░  ░ ░    ░
)";

#if CMAKE_BUILD_TYPE == 1
#define apex_banner banner_fire
#elif CMAKE_BUILD_TYPE == 2
#define apex_banner banner_doom
#else // == 3
#define apex_banner banner_bloody
#endif
