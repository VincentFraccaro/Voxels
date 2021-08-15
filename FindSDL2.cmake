set(FIND_SDL2_PATHS "${PROJECT_SOURCE_DIR}")

find_library(SDL2_LIBRARY NAMES SDL2main.lib
        PATH_SUFFIXES lib
        PATHS "${FIND_SDL2_PATHS}/lib")